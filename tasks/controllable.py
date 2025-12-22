import csv
import torch
import json
from pathlib import Path
from models.ar import load_ar_model
from models.nar import load_nar_model
from models.diffusion import load_diffusion_model
from transformers import BertForSequenceClassification, BertTokenizer
from collections import Counter
from tasks.task_utils import generate_text_for_task, summarize_perplexity_results
from data.utils import parse_control_prompt


def evaluate_controllable_generation(outputs, summary_path, perplexity_summary_path=None):

    # Load formality and sentiment classifier models
    formality_classifier_path = Path(__file__).resolve().parent.parent / "models" / "bert_formality_classifier"
    sentiment_classifier_path = Path(__file__).resolve().parent.parent / "models" / "bert_sentiment_classifier"

    formality_model = BertForSequenceClassification.from_pretrained(formality_classifier_path)
    formality_tokenizer = BertTokenizer.from_pretrained(formality_classifier_path)

    sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_classifier_path)
    sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_classifier_path)

    # load previously calculated perplexity calculations if file exists
    if perplexity_summary_path:
        with open(perplexity_summary_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            base_rows = list(reader)
    else:
        base_rows = []

    # For each sentence, get the prompt, model details, text and inference time
    updated_rows = []
    for item in outputs:
        prompt = item["prompt"]
        model_type = item["model_type"]
        generated_sentences = item["generated_sentences"]
        avg_inference_time = item["avg_inference_time"]

        try:
            prompt_sentiment, prompt_formality = parse_control_prompt(prompt)
        except ValueError:
            prompt_sentiment, prompt_formality = "unspecified", "unspecified"

        # Analyse prompt accuracy using the fine-tuned Bert sentiment and formality classifiers

        sentiment_preds = batch_classify(sentiment_model, sentiment_tokenizer, generated_sentences,
                                         {0: "Negative", 1: "Positive"})
        formality_preds = batch_classify(formality_model, formality_tokenizer, generated_sentences,
                                         {0: "Informal", 1: "Formal"})

        sentiment_confusion_matrix, formality_confusion_matrix = evaluate_prompt_matches(
            prompt_sentiment, prompt_formality, sentiment_preds, formality_preds
        )

        # Summarise results and save as a csv file

        base_row = next((r for r in base_rows if r["prompt"] == prompt and r["model_type"] == model_type), {})

        sentiment_total = sum(sentiment_confusion_matrix.values())
        formality_total = sum(formality_confusion_matrix.values())

        sentiment_correct = sentiment_confusion_matrix["TP"] + sentiment_confusion_matrix["TN"]
        formality_correct = formality_confusion_matrix["TP"] + formality_confusion_matrix["TN"]

        sentiment_accuracy = (f"{(sentiment_correct / sentiment_total):.2f}" if sentiment_total else "N/A")
        formality_accuracy = (f"{(formality_correct / formality_total):.2f}" if formality_total else "N/A")

        updated_rows.append({
            "prompt": prompt,
            "model_type": model_type,
            "avg_perplexity": base_row.get("avg_perplexity", "N/A"),
            "avg_inference_time": f"{avg_inference_time:.4f}" if isinstance(avg_inference_time, float) else "N/A",
            "sentiment_accuracy": sentiment_accuracy,
            "formality_accuracy": formality_accuracy,
            "example_output": generated_sentences[0] if generated_sentences else "N/A"
        })

        fieldnames = ["prompt", "model_type", "avg_perplexity", "avg_inference_time", "sentiment_accuracy",
                      "formality_accuracy", "example_output"]

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline= "", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

    print(f"Evaluation complete. Saved to {summary_path}")


def batch_classify(model, tokenizer, texts, label_map, batch_size=16):
    # use the fine-tuned classifier model to classify a batch of generated sentences (sentiment or formality_)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # tokenise and encode the sentences as vectors so the classifier can categorise them
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings)
            # choose the most likely category for each sentence
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            preds.extend([label_map[p] for p in batch_preds])
    return preds


def evaluate_prompt_matches(prompt_sentiment, prompt_formality, sentiment_preds, formality_preds):
    # For each sentiment and formality prompt, count the number of generated sentences which match the prompt / don't
    # match the prompt to create a confusion matrix

    sentiment_results = Counter()
    formality_results = Counter()

    for sent_pred, form_pred in zip(sentiment_preds, formality_preds):
        if sent_pred == "Positive":
            if prompt_sentiment == "Positive":
                sentiment_results["TP"] += 1
            else:
                sentiment_results["FP"] += 1
        else:   # sent_pred == "Negative"
            if prompt_sentiment == "Negative":
                sentiment_results["TN"] += 1
            else:
                sentiment_results["FN"] += 1

        if form_pred == "Formal":
            if prompt_formality == "Formal":
                formality_results["TP"] += 1
            else:
                formality_results["FP"] += 1
        else:   # form_pred == "Informal":
            if prompt_formality == "Informal":
                formality_results["TN"] += 1
            else:
                formality_results["FN"] += 1

    return sentiment_results, formality_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define controllable generation prompts

    sentiments = ["Positive", "Negative"]
    formalities = ["Formal", "Informal"]

    prompts = [(s, f) for s in sentiments for f in formalities]

    # Model file paths
    ar_model_path = Path(__file__).resolve().parent.parent / "models" / "gpt2_ar_controllable"
    nar_model_path = Path(__file__).resolve().parent.parent / "models" / "bert-base-uncased_nar_controllable"
    nar_dist_model_path = Path(__file__).resolve().parent.parent / "models" / "bert-base-uncased_nar_controllable_dist"
    diffusion_model_path = Path(__file__).resolve().parent.parent / "models" / "t5-small_diffusion_controllable"

    # Load each model and set to eval mode for generation
    ar_model, ar_tokenizer = load_ar_model(ar_model_path)
    ar_model.eval()

    nar_model, nar_tokenizer = load_nar_model(nar_model_path)
    nar_model.eval()

    nar_dist_model, nar_dist_tokenizer = load_nar_model(nar_dist_model_path)
    nar_dist_model.eval()

    diffusion_model, diffusion_tokenizer, encoder = load_diffusion_model(diffusion_model_path, d_model=768,
                                                                         device=device)
    diffusion_model.eval()
    encoder.eval()


    # Generate text using each model and save a temporary file with the outputs, so perplexity can be calculated
    models_dict = {"ar": ar_model, "nar": nar_model, "nar_dist": nar_dist_model, "diffusion": diffusion_model}
    tokenizer_dict = {"ar": ar_tokenizer, "nar": nar_tokenizer, "nar_dist": nar_dist_tokenizer,
                      "diffusion": diffusion_tokenizer}
    encoder_dict = {"diffusion": encoder}

    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "tasks" / "controllable_results"
    TEMP_PATH = RESULTS_DIR / "controllable_temp_outputs.json"
    SUMMARY_PATH = RESULTS_DIR / "controllable_evaluation_summary.csv"
    PERPLEXITY_SUMMARY_PATH = RESULTS_DIR / "controllable_perplexity_summary.csv"

    generate_text_for_task(prompts=prompts, models_dict=models_dict, tokenizer_dict=tokenizer_dict,
                           encoder_dict=encoder_dict, temp_output_path=TEMP_PATH, remove_prompt=True)

    with TEMP_PATH.open("r", encoding="utf-8") as f:
        generated_outputs = json.load(f)

    # Calculate average perplexity and prompt accuracy and save the results
    summarize_perplexity_results(generated_outputs, PERPLEXITY_SUMMARY_PATH, RESULTS_DIR)
    evaluate_controllable_generation(generated_outputs, SUMMARY_PATH, PERPLEXITY_SUMMARY_PATH)


if __name__ == "__main__":
    main()
