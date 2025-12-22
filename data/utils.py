from transformers import AutoModelForCausalLM, AutoTokenizer

def split_dataset(dataset, test_size=0.1, seed=42):
    return dataset.train_test_split(test_size=test_size, seed=seed)


def format_control_prompt(sentiment, formality, include_sep=False):
    # Use the control prompt tags to generate custom prompt tokens which can be added to example texts for training
    # Initial models used a colon separator, but this caused issues for NAR and diffusion models as it was treated as
    # part of the text during training. Removed separator for later models but kept toggle option for backwards
    # compatibility with already trained models

    sentiment_token = ""
    if sentiment == "Positive":
        sentiment_token = "[SENT_POS]"
    elif sentiment == "Negative":
        sentiment_token = "[SENT_NEG]"
    elif sentiment == "unspecified":
        sentiment_token = "[SENT_UNSPEC]"

    formality_token = ""
    if formality == "Formal":
        formality_token = "[FORM_FORMAL]"
    elif formality == "Neutral":
        formality_token = "[FORM_NEUTRAL]"
    elif formality == "Informal":
        formality_token = "[FORM_INFORMAL]"
    elif formality == "unspecified":
        formality_token = "[FORM_UNSPEC]"

    return f"{sentiment_token}{formality_token}{':' if include_sep else ''}"


def add_control_tokens_and_resize(tokenizer, model):
    # Add the custom control prompt tokens to the tokenizer's list of token embeddings and adjust the tokenizer to
    # accommodate them
    tokenizer.add_special_tokens({
        'additional_special_tokens': [
            '[SENT_POS]', '[SENT_NEG]', '[SENT_UNSPEC]',
            '[FORM_FORMAL]', '[FORM_INFORMAL]', '[FORM_NEUTRAL]', '[FORM_UNSPEC]'
        ]
    })
    model.resize_token_embeddings(len(tokenizer))


def parse_control_prompt(prompt):
    # split control prompt for analysing prompt accuracy
    if "[SENT_POS]" in prompt:
        sentiment = "Positive"
    elif "[SENT_NEG]" in prompt:
        sentiment = "Negative"
    else:
        sentiment = "unspecified"

    if "[FORM_FORMAL]" in prompt:
        formality = "Formal"
    elif "[FORM_INFORMAL]" in prompt:
        formality = "Informal"
    elif "[FORM_NEUTRAL]" in prompt:
        formality = "Neutral"
    else:
        formality = "unspecified"

    return sentiment, formality



def convert_safetensors_to_pytorch_model_bin(model_dir, output_dir):
    # convert AR fine-tuned model from safetensors format to pytorch model bin format so that AR weights can be used
    # to initialise diffusion model training

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
