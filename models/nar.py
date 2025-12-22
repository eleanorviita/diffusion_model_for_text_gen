from transformers import (BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling,
                          DefaultDataCollator, EarlyStoppingCallback)
import torch
import random
import os
from data.utils import add_control_tokens_and_resize
from data.preprocess import (create_nar_distilled_dataset, load_teacher_outputs, generate_prompts_from_dataset,
                             generate_open_ended_prompts_from_dataset, PrefixSamplerConfig)
from pathlib import Path


# --- Custom Trainer (based on HuggingFace Trainer) and loss compute method ---

class MaskedTrainer(Trainer):
    # Customised HF Trainer to handle tokenizer spare vocab tokens of the form [unused123]
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_tokenizer = tokenizer

        # suppress spare vocab tokens with the format [unused123]
        self.unused_token_ids = [self._custom_tokenizer.convert_tokens_to_ids(token) for token in self._custom_tokenizer.vocab
                                 if token.startswith("[unused")]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        # Calculate loss using cross-entropy

        labels = inputs.get("labels").clone()
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.unused_token_ids:
            # suppress "[unused123]" format tokens
            logits[:, :, self.unused_token_ids] = -1e10

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# --- Model loading and training functions ---

def load_nar_model(model_name="bert-base-uncased", add_control_tokens=True):
    # load the trained NAR model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    # add control prompt tokens
    if add_control_tokens:
        add_control_tokens_and_resize(tokenizer, model)

    return model, tokenizer


def run_nar_training_epoch(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
        training_args,
        data_collator,
        epoch=None,
        total_epochs=10,
        do_save=False,
        mlm_probability=0.15
):

    # use the custom MaskedTrainer function to run a single training epoch

    trainer = MaskedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer


def train_nar_model(model, tokenizer, train_dataset, eval_dataset, output_dir, task="controllable", distillation=False,
                    teacher_model_path=None, n_samples=None, mask_prob=0.5, num_epochs=30, batch_size=8,
                    early_stopping=True, patience=5, **kwargs):

    # coordinate training based on task and whether knowledge distillation will be used

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        report_to=["tensorboard"]
    )

    # use HuggingFace data collator for batching and token masking (for controllable generation)
    if distillation:
        data_collator = DefaultDataCollator()

    else:
        if task == "controllable":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,    # masked language modelling
                mlm_probability=0.15
            )
        else:
            data_collator = DefaultDataCollator()

    train_fn = train_nar_with_distillation if distillation else train_nar_standard

    train_fn(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        teacher_model_path=teacher_model_path,
        n_samples=n_samples,
        mask_prob=mask_prob,
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        patience=patience,
        batch_size=batch_size,
        training_args=training_args,
        data_collator=data_collator,
        task=task,
        **kwargs)


def train_nar_with_distillation(model, tokenizer, train_dataset, eval_dataset, output_dir, training_args, data_collator,
                                teacher_model_path, n_samples, mask_prob, num_epochs, batch_size, early_stopping,
                                patience, task, **kwargs):
    # manage setup and training loops for NAR training with knowledge distillation

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # select the relevant teacher example file for the task
    if task == "open_ended":
        teacher_egs_path = Path(teacher_model_path) / "teacher_examples_open_ended.pkl"
    else:
        teacher_egs_path = Path(teacher_model_path) / "teacher_examples_controllable.pkl"

    teacher_outputs = load_teacher_outputs(teacher_egs_path, n_samples)

    # if teacher examples have been saved (correctly) with their associated prompts, use these for training. Otherwise
    # prompts will be generated later on
    if teacher_outputs and isinstance(teacher_outputs[0], dict):
        teacher_prompts = [example.get("prompt") for example in teacher_outputs]
        teacher_texts = [example.get("text") for example in teacher_outputs]

    else:
        teacher_prompts = None
        teacher_texts = teacher_outputs

    SEED = 1234

    start_epoch = 0
    best_eval_loss = float("inf")
    epochs_no_improvement = 0   # for monitoring early stopping

    resume_checkpoint = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # checkpoint save / resume in case of interruption
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        print("Previous checkpoints found")
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        resume_checkpoint = os.path.join(output_dir, latest_checkpoint, "checkpoint.pth")
        if os.path.exists(resume_checkpoint):
            print(f"Resuming training from checkpoint {latest_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]

    try:
        for epoch in range(start_epoch, num_epochs):
            # progressively mask more tokens as training progresses
            start_prob = 0.15
            epoch_mask_prob = min(start_prob + (mask_prob - start_prob) * (epoch / max(1, num_epochs - 1)), mask_prob)

            # prepare masked training data using teacher output prompts if previously saved, or by generating prefixes
            # if needed

            if teacher_prompts is not None:
                prompts = teacher_prompts
            elif task == "controllable":
                prompts = generate_prompts_from_dataset(train_dataset)
            else:
                cfg = PrefixSamplerConfig(max_prefix_k=16, p0=0.6, p_short=0.35, short_lo=3, short_hi=8)
                prompts = generate_open_ended_prompts_from_dataset(train_dataset, tokenizer, n_samples, cfg=cfg,
                                                                   seed=SEED)

            L = min(len(prompts), len(teacher_texts))
            distilled_examples = [
                create_nar_training_example(prompts[i], teacher_texts[i], tokenizer, mask_prob=epoch_mask_prob)
                for i in range(L)
            ]
            train_tokenized = create_nar_distilled_dataset(distilled_examples, tokenizer)
            train_tokenized.shuffle()

            training_args.num_train_epochs = 1

            # run 1 training epoch
            run_nar_training_epoch(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_tokenized,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                training_args=training_args,
                data_collator=data_collator,
                epoch=epoch,
                total_epochs=num_epochs
            )

            # calculate loss after each training epoch
            model.eval()
            eval_loss = 0
            eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()
            avg_eval_loss = eval_loss / len(eval_loader)
            print(f"Epoch {epoch+1}: Avg eval loss: {avg_eval_loss:.4f}")

            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # save the model checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1
            }, os.path.join(checkpoint_dir, "checkpoint.pth"))

            # determine if the model has improved
            if avg_eval_loss < best_eval_loss:
                epochs_no_improvement = 0
                best_eval_loss = avg_eval_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            else:
                epochs_no_improvement += 1

            # trigger early stopping if the model has not improved for x epochs (patience argument)
            if epochs_no_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    finally:
        # save the best model
        best_model_path = os.path.join(output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)


def train_nar_standard(model, tokenizer, train_dataset, eval_dataset, output_dir, training_args, data_collator,
                       early_stopping, **kwargs):

    # checkpoint save / resume in case of interruption
    os.makedirs(output_dir, exist_ok=True)
    resume_checkpoint = None

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        print("Previous checkpoints found")
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        resume_checkpoint = os.path.join(output_dir, latest_checkpoint)
        print("Resuming training from checkpoint", resume_checkpoint)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)] if early_stopping else []

    # use the custom MaskedTrainer (based on HuggingFace Trainer) to manage training loops, loss calculation and
    # early stopping
    trainer = MaskedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # save the best model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


# --- Helper function for knowledge distillation ---

def create_nar_training_example(input_text, output_text, tokenizer, mask_prob=0.5):
    # take the teacher example text and apply random, partial token masking to create a training example for the
    # NAR model

    #encode teacher example as token IDs
    target_ids = tokenizer.encode(output_text, add_special_tokens=False)
    masked_input_ids = []

    # randomly replace tokens with masks
    for id in target_ids:
        if random.random() < mask_prob:
            masked_input_ids.append(tokenizer.mask_token_id)
        else:
            masked_input_ids.append(id)

    # return the masked example (for model training) and original sequence (for supervision)
    return input_text, masked_input_ids, target_ids


# --- Functions for text generation using fine-tuned model ---

def generate_nar_text(prompt, model, tokenizer, max_length=30, mask_token="[MASK]", max_iter=20,
                        confidence_threshold=0.6, sampling=False):
    # Inference method 1, using a fixed confidence threshold to determine token remasking strategy

    model.eval()
    device = model.device

    # exclude [unused123] type tokens
    if not hasattr(tokenizer, "_blocked_token_ids"):
        unused_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.vocab if token.startswith("[unused")]
        special_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
                                                      tokenizer.mask_token, tokenizer.unk_token])
        tokenizer._blocked_token_ids = list(set(unused_ids + special_ids))

    blocked_ids = tokenizer._blocked_token_ids

    # encode the prompt as token IDs
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # create a fixed length sequence, but with just masks after the prompt
    num_masks = max_length - len(prompt_token_ids)
    if num_masks < 0:
        return prompt

    initial_masked_input = prompt + " " + " ".join([mask_token] * num_masks)

    inputs = tokenizer(initial_masked_input, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=max_length).to(device)
    input_ids = inputs["input_ids"]

    prev_input_ids = input_ids.clone()

    # iteratively predict tokens and re-mask low confidence tokens
    for step in range(max_iter):

        mask_positions = torch.nonzero(input_ids[0] == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        if len(mask_positions) == 0:
            break

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, mask_positions]

        if blocked_ids:
            logits[:, blocked_ids] = -1e10

        # use softmax to predict most probable tokens
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # optional sampling if several tokens are plausible - increase variety of output
        if sampling:
            top_pred_id = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            top_prob = torch.gather(probabilities, 1, top_pred_id.unsqueeze(-1)).squeeze(-1)
        else:
            top_prob, top_pred_id = torch.max(probabilities, dim=-1)

        # remask low confidence tokens. If none meet the threshold, keep the ones with highest probability.
        high_conf_mask = top_prob >= confidence_threshold

        if high_conf_mask.any():
            high_conf_indices = mask_positions[high_conf_mask]
            high_conf_predictions = top_pred_id[high_conf_mask]
            input_ids[0, high_conf_indices] = high_conf_predictions

        elif mask_positions.numel() > 0:
            best_idx = torch.argmax(top_prob)
            best_mask_position = mask_positions[best_idx]
            best_prediction_id = top_pred_id[best_idx]
            input_ids[0, best_mask_position] = best_prediction_id

        # if the model can't predict any more tokens, stop iterating
        if torch.equal(prev_input_ids, input_ids):
            break
        prev_input_ids = input_ids.clone()

    token_ids = input_ids.squeeze(0).tolist()

    # filter out any control tokens from the output
    control_token_ids = set(tokenizer.convert_tokens_to_ids(
        [f"[SENT_{s.upper()}]" for s in ["POS", "NEG", "UNSPEC"]] +
        [f"[FORM_{f.upper()}]" for f in ["FORMAL", "INFORMAL", "NEUTRAL", "UNSPEC"]]
    ))

    ids_to_remove = set(blocked_ids) - control_token_ids
    filtered_ids = [token_id for token_id in token_ids if token_id not in ids_to_remove]

    # decode the token ids back to actual tokens
    output_text = tokenizer.decode(filtered_ids)
    output_text = ' '.join(output_text.split()).strip()
    return output_text

'''
# Inference method 2
# Alternative text generation function using proportional masking instead of a fixed threshold (decreasing with each
# remasking iteration so it always fills in some masks even if confidence is very low
# Also suppressed EOS and MASK tokens to encourage longer sequences
# Resulting quality was much lower than previous text generation function

def generate_nar_text(prompt, model, tokenizer, max_length=64, mask_token="[MASK]", max_iter=24,
                        sampling=False, min_len=32, final_temp=0.85):
    model.eval()
    device = model.device

    # exclude [unused123] type tokens
    if not hasattr(tokenizer, "_blocked_token_ids"):
        unused_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.vocab if token.startswith("[unused")]
        special_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
                                                      tokenizer.mask_token, tokenizer.unk_token])
        tokenizer._blocked_token_ids = list(set(unused_ids + special_ids))

    blocked_ids = tokenizer._blocked_token_ids
    
    # encode the prompt as token IDs
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # create a fixed length sequence, but with just masks after the prompt
    num_masks = max_length - len(prompt_token_ids)
    if num_masks < 0:
        return prompt

    initial_masked_input = prompt + " " + " ".join([mask_token] * num_masks)

    inputs = tokenizer(initial_masked_input, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=max_length).to(device)
    input_ids = inputs["input_ids"]

    prev_input_ids = input_ids.clone()

    # iteratively predict tokens and re-mask low confidence tokens
    for step in range(1, max_iter + 1):

        mask_positions = torch.nonzero(input_ids[0] == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        if len(mask_positions) == 0:
            break

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, mask_positions]

        if blocked_ids:
            logits[:, blocked_ids] = -1e10
        
        # block EOS token use before the sequence has a minimum number of unmasked tokens
        logits = suppress_eos_mask_tokens(logits, step, max_iter, min_len=min_len, cur_len=max_length,
                                          eos_id=tokenizer.eos_token_id, mask_id=tokenizer.mask_token_id)

        # at the final step, force a token prediction for all remaining masks, regardless of how low the probability is
        if step == max_iter:
            probabilities = torch.nn.functional.softmax(logits / final_temp, dim=-1)
        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # at the final step, use argmax to identify the most likely token
        if step == max_iter:
            top_pred_id = probabilities.argmax(dim=-1)
            input_ids[0, mask_positions] = top_pred_id
        else:
            K = schedule_remask(mask_positions.numel(), step, max_iter)
            
            # use argmax for predictions per position
            top_prob, top_pred_id = torch.max(probabilities, dim=-1)
            
            #choose highest confidence positions to fix predictions; remask others
            position_order = torch.argsort(top_prob, descending=True)
            commit_positions = position_order[:K]
            commit_mask_positions = mask_positions[commit_positions]
            commit_pred_ids = top_pred_id[commit_positions]
            input_ids[0, commit_mask_positions] = commit_pred_ids

        if torch.equal(prev_input_ids, input_ids):
            pass
        prev_input_ids = input_ids.clone()

    token_ids = input_ids.squeeze(0).tolist()

    # filter out any control tokens from the output
    control_token_ids = set(tokenizer.convert_tokens_to_ids(
        [f"[SENT_{s.upper()}]" for s in ["POS", "NEG", "UNSPEC"]] +
        [f"[FORM_{f.upper()}]" for f in ["FORMAL", "INFORMAL", "NEUTRAL", "UNSPEC"]]
    ))

    ids_to_remove = set(blocked_ids) - control_token_ids
    filtered_ids = [token_id for token_id in token_ids if token_id not in ids_to_remove]

    output_text = tokenizer.decode(filtered_ids)
    output_text = ' '.join(output_text.split()).strip()
    return output_text


# --- helper functions for remasking / decoding ---


def schedule_remask(seq_len, t, T, k_start_frac=0.5, k_end_frac=0.05):
    # remask k tokens each iteration. decrease number of remasks with each iteration (linear)
    fraction = k_start_frac + (k_end_frac - k_start_frac) * (t/T)
    return (max(1, int(fraction*seq_len)))


def suppress_eos_mask_tokens(logits, step, final_step, min_len, cur_len, eos_id, mask_id):
    # suppress EOS token until min_len achieved
    if cur_len < min_len:
        logits[..., eos_id] = -1e-9

    # on the final step, ensure no [MASK] tokens remain
    if step == final_step:
        logits[..., mask_id] = -1e-9

    return logits
'''