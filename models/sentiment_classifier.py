import torch
from datasets import load_dataset
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import os
from sklearn.metrics import classification_report, accuracy_score


MODEL_PATH = "bert_sentiment_classifier"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# --- Dataset loading and tokenisation functions ---

def load_yelp_data(n_samples=10000):
    # load the HuggingFace Yelp dataset as a HF dataset object, split into train and test subsets and tokenize each
    # example for processing by the classifier
    raw_dataset = load_dataset("yelp_polarity", split=f"train[:{n_samples}]")
    dataset = raw_dataset.train_test_split(test_size=0.1)
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


def train_sentiment_classifier(dataset):
    # Fine-tune the HF BertForSequenceClassification model, using the Yelp dataset

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    args = TrainingArguments(
        output_dir="./bert_sentiment_classifier",
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


# --- Classification function (using trained classifier) ---

def predict_sentiment_labels(model, texts, batch_size=12):
    # Use the sentiment classifier, pretrained on Yelp dataset, to categorise given texts as positive / negative
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            preds.extend(batch_preds)
    return preds


# --- Classifier evaluation function ---

def evaluate_sentiment_classifier(model, dataset, save_path="results/sentiment_classifier_eval.txt"):
    # compare predicted and true values for sentiment, using the Yelp 'test' dataset
    # produce a confusion matrix to assess the accuracy of the sentiment classifier
    texts = dataset["test"]["text"]
    true_labels = dataset["test"]["label"]
    preds = predict_sentiment_labels(model, texts)

    accuracy = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, target_names=["Positive", "Negative"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.2}\n\n")
        f.write(report)

    print(f"Saved formality classifier evaluation results to {save_path}")
    return accuracy, report


if __name__ == "__main__":
    dataset = load_yelp_data(n_samples=10000)
    train_sentiment_classifier(dataset)