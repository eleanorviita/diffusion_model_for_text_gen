import json
import pickle
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from data.utils import format_control_prompt
from pathlib import Path
import re
import torch
import html
import random
import hashlib
from models.ar import load_ar_model, generate_ar_batch
from dataclasses import dataclass

# --- regex helpers for wikitext ---

HEADING_RE   = re.compile(r"^\s*={2,}[^=].*={2,}\s*$")
CATEGORY_RE  = re.compile(r"\[\[\s*Category:[^\]]+\]\]", re.IGNORECASE)
FILE_RE      = re.compile(r"\[\[\s*(File|Image):[^\]]+\]\]", re.IGNORECASE)
TEMPLATE_RE  = re.compile(r"\{\{[^}]+\}\}")
URL_RE       = re.compile(r"https?://\S+")
CONTROL_RE   = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
ONLY_PUNCT_RE= re.compile(r"^\W+$")


# --- custom dataclass to enable prompt construction from dataset ---

@dataclass
class PrefixSamplerConfig:

    # Configuration for sampling a prefix from a tokenized example text. Used to build prompt+continuation training
    # examples for NAR open-ended training
    # Used to generate prompts of different lengths, biased towards short prompts / no prompt

    max_prefix_k: int = 16    # max length of prefix
    p0: float = 0.6           # probability of empty prefix (no prompt)
    p_short: float = 0.35     # prob of short prefix
    short_lo: int = 3         # min length of 'short' prefix
    short_hi: int = 8         # max length of 'short' prefix

    def sample(self, ids, rng: random.Random):
        # sample a prefix of length k and cut=off point c from a given sequence (mirrors diffusion prefix handling)

        # ensure minimum 2 tokens (at least one prompt token and one continuation token)
        if not ids or len(ids) <2:
            return [], 0

        # cap k so that at least one token in the sequence is available for continuation
        k_max = min(self.max_prefix_k, max(1, len(ids) - 1))
        r = rng.random()

        # if prob < p0, use no prefix (unprompted)
        if r < self.p0:
            k = 0

        # if prob > p0 and < p_short, sample a short prefix of length 3-8 tokens
        elif r < self.p0 + self.p_short:
            lo = min(self.short_lo, k_max)
            hi = min(self.short_hi, k_max)
            if hi >= 1 and hi >= lo:
                k = rng.randint(max(1, lo), hi)
            else:
                k = rng.randint(1, k_max)

        # if prob > p_short, sample a longer prefix
        else:
            lo = max(9, 0)
            if k_max >= lo:
                k = rng.randint(lo, k_max)
            else:
                k = rng.randint(1, k_max)

        # define the cut-off point between prefix and continuation
        c = rng.randint(max(1, k), len(ids) - 1)
        if k > 0:
            prefix_tokens = ids[max(0, c - k): c]
        else:
            prefix_tokens = []

        return prefix_tokens, c


# --- Functions for loading and formatting standard datasets for model training ---

def load_yelp_for_controllable(n_samples=10000):
    # Load the HuggingFace Yelp review polarity dataset for training models in controllable text generation
    # (sentiment control)
    raw_data = load_dataset("yelp_polarity", split=f"train[:{n_samples}]")

    def format_example(example):
        # helper function to format each example to include the sentiment prompt as a tag and within the text itself
        # for model training
        sentiment = "Positive" if example["label"] == 1 else "Negative"
        formality = "unspecified"
        prompt = format_control_prompt(sentiment, formality)
        return {
            "text": f"{prompt} {example['text']}",
            "sentiment": sentiment,
            "formality": formality
        }

    formatted_data = raw_data.map(format_example)
    return formatted_data.remove_columns(["label"])


def load_gyafc_for_controllable(domain="Entertainment_Music", n_samples=10000):
    # Load GYAFC dataset for training models in controllable text generation (formality control)
    raw_data = load_dataset("gyafc", domain=domain, split="test")

    informal_samples = raw_data["informal"][:n_samples]
    formal_samples = raw_data["formal"][:n_samples]

    examples = []
    # format each example to include the formality prompt as a tag and within the text
    for informal, formal in zip(informal_samples, formal_samples):
        prompt_formal = format_control_prompt("unspecified", "Formal")
        prompt_informal = format_control_prompt("unspecified", "Informal")
        examples.append({
            "text": f"{prompt_formal} {formal}",
            "sentiment": "unspecified",
            "formality": "Formal"
        })

        examples.append({
            "text": f"{prompt_informal} {informal}",
            "sentiment": "unspecified",
            "formality": "Informal"
        })

    return Dataset.from_list(examples)


def clean_line(line):
    # regex cleaning for Wikitext dataset: remove html, headings (which are generally too short to be useful as
    # training examples) etc

    if not isinstance(line, str):
        return ""
    line = line.replace("<unk>", " ")
    line = html.unescape(line)
    line = CONTROL_RE.sub(" ", line)
    if HEADING_RE.match(line):
        return ""
    line = TEMPLATE_RE.sub(" ", line)
    line = CATEGORY_RE.sub(" ", line)
    line = URL_RE.sub(" ", line)
    line = FILE_RE.sub(" ", line)
    line = re.sub(r"\s+", " ", line)
    if not line or ONLY_PUNCT_RE.match(line):
        return ""
    return line


def hash_norm(line):
    # normalise whitespace in a text input to allow better deduplication
    return hashlib.sha1(line.encode("utf-8", "ignore")).hexdigest()

def load_wikitext_for_open_ended(n_samples=50000):
    # Load and clean the HuggingFace wikitext 103 dataset for training models in open-ended text generation
    raw_data = load_dataset("wikitext", "wikitext-103-v1",
    split={
        "train": f"train[:{n_samples}]",
        "validation": f"validation[:{n_samples//10}]",
        "test": f"test[:{n_samples//10}]",
    })

    def clean_map(example):
        # map the cleaning function onto each training example
        txt = clean_line(example["text"])
        return {"text": txt}

    output = {}
    for split in ("train", "validation", "test"):
        data = raw_data[split].map(clean_map, desc=f"clean{split}")
        data = data.filter(lambda x: len(x["text"]) >= 20, desc=f"remove short {split}")

        # remove any duplicate data
        seen = set()
        def not_seen(example):
            key = hash_norm(example["text"])
            if key in seen:
                return False
            seen.add(key)
            return True

        data = data.filter(not_seen, desc=f"deduplicate {split}")
        output[split] = data
    return output


# --- Functions for loading and formatting custom datasets for training open-ended generation ---

def build_nar_prefix_examples(dataset, tokenizer, n_samples, seq_len=256, cfg = PrefixSamplerConfig(), seed=1234):
    # sample a random prefix length and mask the rest of the sequence for training the NAR model to fill in the
    # continuation sequence. Includes classifier-free-guidance (i.e. no prefix) if flagged

    rng = random.Random(seed)
    torch.manual_seed(seed)

    # define the mask and pad token ids
    mask_id = tokenizer.mask_token_id
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = tokenizer.eos_token_id

    texts = [dataset[i]["text"] for i in range(min(n_samples, len(dataset)))]
    examples = []

    for t in texts:
        # get token ids for a sequence
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if not ids or len(ids) < 2:
            continue

        prefix_tokens, c = cfg.sample(ids, rng)

        # determine the length of the continuation sequence (max = sequence length - prefix length)
        max_cont = max(1, seq_len - len(prefix_tokens))
        cont = ids[c : c + max_cont]

        # ensure the total prefix + continuation length does not exceed maximum sequence length
        total_len = len(prefix_tokens) + len(cont)
        if total_len > seq_len:
            cont = cont[: seq_len - len(prefix_tokens)]

        # construct the prefix + masked continuation
        input_ids = prefix_tokens + [mask_id] * len(cont)

        labels = ([-100] * len(prefix_tokens)) + cont

        # pad any remaining space at the end of the sequence
        pad_len = seq_len -len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len

        # prepare the attention mask
        attention_mask = [1] * (seq_len - pad_len) + [0] * pad_len

        examples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        })
    return examples


def create_nar_prefix_cont_dataset(examples):
    # wrap the NAR prefix-continuation examples into a HuggingFace Dataset class
    return Dataset.from_list(examples)


# --- Functions for loading and formatting custom datasets for training controllable generation ---

def load_combined_dataset(n_samples=10000):
    # helper function which just concatenates datasets with different control prompts
    yelp = load_yelp_for_controllable(n_samples=n_samples)
    pavlick = load_pavlick_for_controllable(n_samples=n_samples)
    #gyafc = load_gyafc_for_controllable(n_samples=n_samples)
    return concatenate_datasets([yelp, pavlick])


def load_yelp_formality_labelled(n_samples=10000):
    # Load the locally-stored copy of the Yelp polarity dataset with additional formality labels
    data_path = Path(__file__).resolve().parent.parent / "data" / "yelp_formality_tagged_clean.jsonl"
    raw_data = load_dataset("json", data_files=str(data_path), split=f"train[:{n_samples}]")

    def format_example(example):
        # format each example to include the sentiment / formality prompt as a tag and within the text itself
        control_prompt = format_control_prompt(example["sentiment"], example["formality"])
        full_text = control_prompt + " " + example["text"]
        return {
            "text": full_text,
            "sentiment": example["sentiment"],
            "formality": example["formality"]
        }

    return raw_data.map(format_example)


# --- Tokenize functions for different models ---

def tokenize_dataset(dataset, tokenizer, model_type="ar", max_length=512):
    # function to coordinate tokenising the dataset, depending on model type (N.B. diffusion model has a separate,
    # custom tokeniser)
    def tokenize_ar(batch):
        # for AR the standard HF Bert tokeniser can be used
        tokens = tokenizer(batch["text"], max_length=max_length, truncation=True, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def tokenize_nar(batch):
        # for NAR the Bert tokeniser is used, with special tokens (e.g. separators, padding tokens) suppressed
        tokens = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=max_length)

        input_ids = tokens["input_ids"]

        labels = []
        # prepare token labels
        for ids in input_ids:
            new_labels = []
            for j, token_id in enumerate(ids):
                token = tokenizer.convert_ids_to_tokens([token_id])[0]

                if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token]:
                    # suppress special tokens e.g. padding, masks
                    new_labels.append(-100)
                else:
                    new_labels.append(token_id)

            labels.append(new_labels)
        tokens["labels"] = labels
        return tokens

    # use the correct tokenize function for each model
    tokenizer_map = {
        "ar": tokenize_ar,
        "nar": tokenize_nar
    }

    if model_type not in tokenizer_map:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return dataset.map(tokenizer_map[model_type], batched=True, remove_columns=dataset.column_names)


# --- Functions for creating and loading datasets for knowledge distillation ---

def generate_prompts_from_dataset(dataset):
    # extract control prompts from the dataset examples
    prompts = [
        format_control_prompt(
            example.get("sentiment", "unspecified"),
            example.get("formality", "unspecified")
        ) for example in dataset
    ]
    return prompts


def generate_open_ended_prompts_from_dataset(dataset, tokenizer, n_samples, cfg=PrefixSamplerConfig(), seed=1234):
    # sample random prefixes from the dataset example text as prompts for open-ended generation
    # include some examples without prompts if classifier-free guidance is flagged

    rng = random.Random(seed)

    texts = [dataset[i]["text"] for i in range(min(n_samples, len(dataset)))]

    prompts = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if not ids or len(ids) < 2:
            prompts.append("")
            continue
        prefix_tokens, _ = cfg.sample(ids, rng)

        if prefix_tokens:
            prompt = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
        else:
            prompt = ""
        prompts.append(prompt)

    return prompts


def generate_teacher_outputs(teacher_model_path, task, n_samples, batch_size=64):
    # Use an existing fine-tuned model as a teacher model to generate training examples which can be used to train
    # other models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model, teacher_tokenizer = load_ar_model(teacher_model_path)
    teacher_model.to(device)
    teacher_model.eval()

    if task == "controllable":
        # load the locally-saved Yelp dataset with sentiment and formality tags
        labelled_dataset = load_yelp_formality_labelled(n_samples=n_samples)
        prompts = generate_prompts_from_dataset(labelled_dataset)

    elif task == "open_ended":
        # load the wikitext dataset
        wikitext = load_wikitext_for_open_ended(n_samples=n_samples)["train"]
        cfg = PrefixSamplerConfig(max_prefix_k=16, p0=0.6, p_short=0.35, short_lo=3, short_hi=8)
        BASE_SEED=1234
        prompts = generate_open_ended_prompts_from_dataset(wikitext, teacher_tokenizer, n_samples, cfg=cfg,
                                                           seed=BASE_SEED)

    else:
        raise ValueError(f"Unsupported task {task}")

    examples_path = Path(teacher_model_path) / f"teacher_examples_{task}.pkl"

    def valid_pair(p, t):
        # remove duplicates, examples with no continuation after prompt / too short, or if the text doesn't start
        # with the prompt
        if not t:
            return False
        if p and not t.startswith(p):
            return False
        return (len(t) - len(p)) >= 40

    if examples_path.exists():
        # load existing teacher examples if previously saved
        print("Loading teacher examples from cache")
        with open(examples_path, "rb") as f:
            teacher_outputs = pickle.load(f)

        cleaned = []
        # clean the examples and ensure they are saved as a dictionary or prompt / text pairs
        for example in teacher_outputs:
            if isinstance(example, dict):
                p, t = example.get("prompt", "") or "", example.get("text", "") or ""
            else:
                p, t = "", str(example)
            if valid_pair(p, t):
                cleaned.append({"prompt": p, "text": t})

        teacher_outputs = cleaned

        with open(examples_path, "wb") as f:
            pickle.dump(teacher_outputs, f)
        print(f"Cleaned {len(teacher_outputs)} cached examples.")

    else:
        teacher_outputs = []

    # deduplicate examples
    seen_texts = {example["text"] for example in teacher_outputs if isinstance(example, dict) and "text" in example}

    start_index = len(teacher_outputs)
    prompts_to_generate = prompts[start_index:]

    # generate examples and save in batches to avoid memory issues / loss on interruption
    for i in range(0, len(prompts_to_generate), batch_size):
        batch_prompts = prompts_to_generate[i:i + batch_size]
        batch_texts = generate_ar_batch(batch_prompts, teacher_model, teacher_tokenizer, micro_batch_size=2)

        valid = [(p, t) for p, t in zip(batch_prompts, batch_texts) if valid_pair(p, t)]
        teacher_outputs.extend({"prompt": p, "text": t} for p, t in valid)

        with open(examples_path, "wb") as f:
            pickle.dump(teacher_outputs, f)
        print(f"Saved {len(teacher_outputs)} teacher outputs")

    if len(teacher_outputs) < n_samples:
        # keep generating until the required number is achieved (after discarding too-short examples, deduplication etc)
        if task == "open_ended":
            extra_seed = BASE_SEED + 1
        else:
            extra_seed = random.randint(0, 10000)
        rounds = 0
        MAX_ROUNDS = 10   #avoid infinite loop

        while len(teacher_outputs) < n_samples and rounds < MAX_ROUNDS:
            need = n_samples - len(teacher_outputs)
            if task == "open_ended":
                extra_prompts = generate_open_ended_prompts_from_dataset(wikitext, teacher_tokenizer, need, cfg=cfg,
                                                                         seed = extra_seed)
                extra_seed+=1
            else:
                extra_prompts = generate_prompts_from_dataset(random.sample(labelled_dataset,
                                                                            k=min(need, len(labelled_dataset))))

            # generate new examples in batches to help memory management
            extra_texts = generate_ar_batch(extra_prompts, teacher_model, teacher_tokenizer, micro_batch_size=2)
            for p, t in zip(extra_prompts, extra_texts):
                if valid_pair(p, t) and (t not in seen_texts):
                    teacher_outputs.append({"prompt": p, "text": t})
                    seen_texts.add(t)

            with open(examples_path, "wb") as f:
                pickle.dump(teacher_outputs, f)

            rounds +=1
            print(f"{rounds} top-up rounds complete. Teacher outputs total: {len(teacher_outputs)}")

    print(f"Teacher outputs saved to {examples_path}. {len(teacher_outputs)} total examples")

    return teacher_outputs


def create_nar_distilled_dataset(examples, tokenizer, max_length=128):
    # Creates a tokenised dataset for knowledge distillation, using the generated teacher examples

    input_ids = []
    attention_mask_list = []
    labels = []

    # loop through each of the teacher examples and tokenise them
    for prompt, masked_token_ids, target_token_ids in examples:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # build an input sequence, with the prompt and all other tokens masked. Truncate if needed.
        full_input_ids = prompt_ids + masked_token_ids
        full_input_ids = full_input_ids[:max_length]
        attention_mask_example = [1] * len(full_input_ids)

        # Build the label sequence, with prompt tokens set to -100 (ignored) for loss calculation
        # masked tokens are replaced with target token IDs
        # -> the model is only trained to generate the actual review text, with prompt as context
        full_labels = [-100] * len(prompt_ids) + target_token_ids
        full_labels = full_labels[:max_length]
        full_labels += [-100] * (max_length - len(full_labels))

        # pad to max_length
        full_input_ids += [tokenizer.pad_token_id] * (max_length - len(full_input_ids))
        attention_mask_example += [0] * (max_length - len(attention_mask_example))

        input_ids.append(full_input_ids)
        attention_mask_list.append(attention_mask_example)
        labels.append(full_labels)

    # convert everything into a HF dataset object
    distilled_dataset = Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask_list,
        "labels": labels
    })

    return distilled_dataset


def load_teacher_outputs(teacher_egs_path, n_samples):
    # helper function to load existing generated teacher examples
    if not Path(teacher_egs_path).exists():
        raise FileNotFoundError(f"Teacher examples file not found: {teacher_egs_path}")

    with open(teacher_egs_path, "rb") as f:
        teacher_outputs = pickle.load(f)

    if len(teacher_outputs) < n_samples:
        raise ValueError(
            f"Only {len(teacher_outputs)} teacher examples were found in the training dataset, but {n_samples} expected."
        )
    return teacher_outputs[:n_samples]


# --- Dataset cleaning / preprocessing functions ---

def clean_yelp_text(text):
    # clean newline and erroneous \\"\" characters from the Yelp dataset
    text = re.sub(r'(\\n\*|\n\*)', ' ', text)
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = text.replace('\\\"\"', '"')
    text = text.replace('\\"', '"')
    text = text.replace('\\', '')

    try:
        text = bytes(text, 'utf-8').decode('unicode_escape')
    except UnicodeDecodeError:
        pass

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_json_file(input_path, output_path):
    # run the Yelp cleaning process and save the clean file
    with open(input_path, "r", encoding='utf-8') as infile, open(output_path, "w", encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if "text" in data:
                data["text"] = clean_yelp_text(data["text"])
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")




if __name__ == "__main__":
    normalize_json_file(input_path="yelp_formality_tagged.jsonl", output_path=f"yelp_formality_tagged_clean.jsonl")