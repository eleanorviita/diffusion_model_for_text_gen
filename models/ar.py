from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import os
from transformers.trainer_utils import get_last_checkpoint
from data.utils import add_control_tokens_and_resize
from transformers.utils import logging


# --- Model loading and training functions ---

def load_ar_model(model_name="gpt2"):
    # load the HuggingFace tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # add the custom control prompt tokens and resize the tokenizer to accommodate them
    add_control_tokens_and_resize(tokenizer, model)

    # use the end-of-sequence token as a padding token (GPT-2 tokenizers often lack a specific 'pad' token)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def train_ar_model(model, tokenizer, train_dataset, eval_dataset, output_dir, **kwargs):
    # fine-tune the base AR model using the relevant dataset for the specified task

    # use the end-of-sequence token as a pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # define the arguments for the training loop so the Trainer can manage training
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=15,
        eval_strategy="epoch",    # evaluate loss after every epoch
        save_strategy="epoch",
        save_total_limit=2,    # save max 2 checkpoints to save hard disk space
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",    # use loss to determine if the model has improved or is over-training
        greater_is_better=False,
        logging_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        dataloader_pin_memory=False,    # memory management for training on GPU (should improve speed)
        eval_steps=None
    )

    # use standard HuggingFace Trainer to manage training loops
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # save checkpoints during training in case of interruption
    latest_checkpoint = None
    if os.path.isdir(output_dir):
        latest_checkpoint = get_last_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# --- Functions for text generation using fine-tuned model ---

def generate_ar_text(prompt, model, tokenizer, max_length=128, knowledge_distillation=False):
    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # prepare input_ids and attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # used the fine-tuned model to generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            temperature=1.0,    #softmax temp
            top_p=0.9,    # probability threshold for candidate tokens
            top_k=50,    # max number of candidate tokens
            min_length=32    # force the model to continue generating until the sequence has at least 32 tokens
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=knowledge_distillation)

@torch.no_grad()
def generate_ar_batch(prompt_list, model, tokenizer, knowledge_distillation=True, micro_batch_size=None):
    # generate a batch of texts using the AR model and tokenizer, given an input prompt

    #disable gradient tracking for inference
    model.eval()

    # prompt_list needs to be a list of prompts - if it is a single string, convert to a list for future handling
    prompts = [prompt_list] if isinstance(prompt_list, str) else list(prompt_list)

    # define a "seed" token in case the prompt is empty. Use EOS / PAD token or a space if necessary
    seed_token = tokenizer.eos_token or tokenizer.pad_token or " "

    # replace empty prompts with seed token
    prompts = [p if (isinstance(p, str) and p.strip()) else seed_token for p in prompts]

    # if model is in decoder-only mode (which it is for generation), ensure left padding
    if not model.config.is_encoder_decoder:
        tokenizer.padding_side='left'
        tokenizer.truncation_side='left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # ensure that the added prompt does not violate overall model sequence length limits
    context = getattr(model.config, "n_positions", getattr(tokenizer, "model_max_length", 1024))
    max_new_tokens = 64
    prompt_max_len = max(8, context - max_new_tokens)

    def gen_chunk(prompts):
        # generate a chunk of texts using a list of prompts

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=prompt_max_len).to(model.device)
        attention_mask = inputs["attention_mask"]

        _prev = logging.get_verbosity()
        # suppress right padding warnings (padding is correct / left padded)
        logging.set_verbosity_error()

        orig_use_cache = getattr(model.config, "use_cache", True)

        # no caching to avoid memory issues
        model.config.use_cache = False

        try:
            # autocast on gpu to reduce memory use
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=32,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=50,
                    tokenizer=tokenizer
                )
        finally:
            model.config.use_cache = orig_use_cache
            logging.set_verbosity(_prev)
        return tokenizer.batch_decode(outputs, skip_special_tokens=knowledge_distillation)

    if micro_batch_size and len(prompts) > micro_batch_size:
        # use micro batches to generate text to avoid hitting memory limits for long prompts
        texts = []
        for i in range(0, len(prompts), micro_batch_size):
            texts.extend(gen_chunk(prompts[i:i+micro_batch_size]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return texts

    return gen_chunk(prompts)
