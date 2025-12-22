import os
from collections import defaultdict
import random
import torch
from datasets import Dataset, load_dataset
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from pathlib import Path
from data.utils import format_control_prompt
import json
from sklearn.metrics import classification_report, accuracy_score
from itertools import islice
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "GYAFC_Corpus" / "GYAFC_Corpus" / "Entertainment_Music"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bert_formality_classifier"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# --- Dataset loading and tokenisation functions ---

def load_gyafc_data(formal_files, informal_files):
    # load the Grammarly Yahoo Answers Formality Corpus into a HuggingFace dataset object, split into train and test
    # subsets and tokenize each example for processing by the classifier
    paths = [(f, 1) for f in formal_files] + [(f, 0) for f in informal_files]
    texts, labels = [], []

    for path, label in paths:
        with open(path, 'r', encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            texts.extend(lines)
            labels.extend([label] * len(lines))
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
    dataset["train"] = dataset["train"].select(range(10000))
    return dataset.map(tokenize_fn, batched=True)

def tokenize_fn(example):
    # tokenise the dataset using the HF Bert tokenizer
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=128)


# --- Model training functions ---

def compute_metrics(eval_pred):
    # Helper function to calculate model accuracy during training
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

def train_formality_classifier(dataset):
    # Fine-tune the HF BertForSequenceClassification model, using the Grammarly Yahoo Answers Formality Corpus

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    args = TrainingArguments(
        output_dir="./bert_formality",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=15,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Use the HF Trainer to manage training loops with early stopping to prevent overtraining
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)


# --- Classification and dataset tagging functions (using trained classifier) ---

def predict_formality_labels(model, texts, device, batch_size, max_length):
    # Use the formality classifier, pretrained on GYAFC dataset, to categorise given texts as formal / informal
    preds = []
    with torch.inference_mode():

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            encodings = {k: v.to(device) for k, v in encodings.items()}
            with torch.inference_mode():
                outputs = model(**encodings)
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                preds.extend(batch_preds)
    return preds


def format_yelp_with_formality(model,
                               start_index=0,   # adjust start point (e.g. if some examples have already been tagged)
                               max_records=None,
                               save_path=Path(__file__).resolve().parent.parent / "data" / "yelp_formality_tagged.jsonl",
                               batch_size=64,
                               max_length=128,
                               write_every=10000):
    # create a version of the standard HuggingFace Yelp dataset with formality labels (in addition to sentiment)
    # ensure dataset includes equal numbers examples with each sentiment / formality combination
    # use batching to manage memory requirements

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset("yelp_polarity", split="train", streaming=True)

    # option to take a specific sub-set of the data
    if start_index:
        ds_iter = islice(raw_data, start_index, None)
    if max_records is not None:
        ds_iter = islice(raw_data, max_records)

    def format_row(example, pred):
        # format the example for input into the classifier
        sentiment = "Positive" if example["label"] == 1 else "Negative"
        formality = "Formal" if pred == 1 else "Informal"
        prompt = format_control_prompt(sentiment, formality)
        return {
            "text": f"{prompt} {example['text']}",
            "sentiment": sentiment,
            "formality": formality
        }

    buffer = []
    total = 0
    batch = []

    # use the classifier to predict formality for batches of Yelp reviews
    with open(save_path, "a", encoding="utf-8") as f:
        for example in tqdm(ds_iter, desc="Predicting yelp formality"):
            batch.append(example)
            if len(batch) >= batch_size:
                texts = [ex["text"] for ex in batch]
                preds = predict_formality_labels(model, texts, device, batch_size=batch_size, max_length=max_length)
                for ex, pred in zip(batch, preds):
                    buffer.append(format_row(ex, pred))

                # each time a batch is completed, balance the batch to have equal numbers of each prompt combination
                # Yelp dataset has approx. 3x more formal than informal examples
                # then save the batch to a json file
                if len(buffer) >= write_every:
                    rows_to_write = balance_labelled_dataset(buffer)
                    for row in rows_to_write:
                        f.write(json.dumps(row) + "\n")
                    buffer.clear()

                total += len(batch)
                batch.clear()

        # handle the last partial batch
        if batch:
            texts = [ex["text"] for ex in batch]
            preds = predict_formality_labels(model, texts, device, batch_size=batch_size, max_length=max_length)
            for ex, pred in zip(batch, preds):
                buffer.append(format_row(ex, pred))
            total +=len(batch)

        if buffer:
            for row in buffer:
                f.write(json.dumps(row) + "\n")

    print(f"Saved {total} labelled Yelp examples to {save_path}")


def balance_labelled_dataset(entries, max_per_class=None):
    # balance the dataset so the output has the same number of examples for each prompt combination
    # allocate the tagged examples to different bins according to prompt, e.g. 'positive formal'
    bins = defaultdict(list)
    counter = defaultdict(int)

    for entry in entries:
        key = (entry["sentiment"], entry["formality"])
        counter[key] +=1
        bins[key].append(entry)

    # find the size of the smallest prompt bin
    min_bin_size=min(len(v) for v in bins.values())
    if max_per_class:
        min_bin_size = min(min_bin_size, max_per_class)

    balanced = []
    # keep only that many examples of each prompt combination, and randomise the order of examples
    for key, examples in bins.items():
        random.shuffle(examples)
        balanced.extend(examples[:min_bin_size])

    random.shuffle(balanced)
    return balanced


# --- Classifier evaluation function ---

def evaluate_formality_classifier(model, dataset, save_path="results/formality_classifier_eval.txt"):
    # compare predicted and true values for formality, using the GYAFC 'test' dataset
    # produce a confusion matrix to assess the accuracy of the formality classifier
    texts = dataset["test"]["text"]
    true_labels = dataset["test"]["label"]
    preds = predict_formality_labels(model, texts)

    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=["Formal", "Informal"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.2}\n\n")
        f.write(report)

    print(f"Saved formality classifier evaluation results to {save_path}")
    return accuracy, report


def main():

    # define GYAFC files for training
    formal_files = [BASE_DIR / "train" / "formal", BASE_DIR / "test" / "formal"]
    informal_files = [BASE_DIR / "train" / "informal", BASE_DIR / "test" / "informal"]

    # load dataset
    dataset = load_gyafc_data(formal_files, informal_files)

    # train classifier
    train_formality_classifier(dataset)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

    # evaluate model
    evaluate_formality_classifier(model, dataset)

    # add formality labels to Yelp dataset
    format_yelp_with_formality(model)


if __name__ == "__main__":
    main()