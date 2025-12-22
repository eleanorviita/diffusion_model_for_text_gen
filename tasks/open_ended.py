from tasks.task_utils import generate_text_for_task, summarize_perplexity_results
import json
import csv
from pathlib import Path
from models.ar import load_ar_model
from models.nar import load_nar_model
from models.diffusion import load_diffusion_model
import random
import re
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from transformers import T5Tokenizer


def evaluate_open_ended_generation(outputs, open_ended_summary_path, perplexity_summary_path=None):

    if perplexity_summary_path:
        with open(perplexity_summary_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            base_rows = list(reader)
    else:
        base_rows = []

    updated_rows = []

    for item in outputs:
        prompt = item["prompt"]
        model_type = item["model_type"]
        generated_sentences = item["generated_sentences"]
        avg_inference_time = item["avg_inference_time"]

        # diversity metrics

        d1 = distinct_n(generated_sentences, 1)
        d2 = distinct_n(generated_sentences, 2)
        sbleu = self_bleu(generated_sentences)

        # Summarise results

        base_row = next((r for r in base_rows if r["prompt"] == prompt and r["model_type"] == model_type), {})

        updated_rows.append({
            "prompt": prompt,
            "model_type": model_type,
            "avg_perplexity": base_row.get("avg_perplexity", "N/A"),
            "avg_inference_time": f"{avg_inference_time:.4f}" if isinstance(avg_inference_time, float) else "N/A",
            "d1": f"{d1:.4f}",
            "d2": f"{d2:.4f}",
            "self_bleu": f"{sbleu:.4f}",
            "example_output": generated_sentences[0] if generated_sentences else "N/A"
        })

    fieldnames = ["prompt", "model_type", "avg_perplexity", "avg_inference_time", "d1", "d2", "self_bleu",
                  "example_output"]

    open_ended_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open_ended_summary_path.open("w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Evaluation complete. Saved to {open_ended_summary_path}")


def construct_wikitext_prompts(n_prompts=100, prompt_len=8, seed=42):
    rng = random.Random(seed)
    raw_data = load_dataset("wikitext", "wikitext-103-v1")

    prompt_candidates = list(raw_data["validation"]["text"])+list(raw_data["test"]["text"])

    def is_header_or_markup(s):
        s = s.strip()
        if not s or len(s) < 30:
            return True
        if s.startswith(("=", "*", "#")):
            return True
        if re.match(r"^\s*\{\{.*\}\}\s*$", s):
            return True
        if re.match(r"^\s*\|", s):
            return True

    cleaned = [c.strip() for c in prompt_candidates if not is_header_or_markup(c)]

    prompts, used = [], set ()
    pool = rng.sample(cleaned, k=min(len(cleaned), n_prompts*5))
    for line in pool:
        tokens = line.split()
        if len(tokens) < prompt_len + 3:
            continue
        max_start = max(1, min(10, len(tokens) - prompt_len - 1))
        start = rng.randint(0, max_start)
        frag = " ".join(tokens[start:start + prompt_len])
        frag = re.sub(r"\s+", " ", frag).strip()

        if frag.lower() in used:
            continue
        used.add(frag.lower())
        prompts.append(frag)
        if len(prompts) >= n_prompts:
            break

    return prompts


# --- Output diversity measures: Self-BLEU and Distinct-n ---

def self_bleu(sentences, weights=(0.25, 0.25, 0.25, 0.25)):
    # Calculate the average BLEU score for each generated sentence, using the rest as a comparison
    # uses BLEU calculation from the NLTK library
    scores = []
    for i, hyp in enumerate(sentences):
        refs = [s.split() for j,s in enumerate(sentences) if j!=i and s.strip()]
        hyp_tokens = hyp.split()
        if not refs or not hyp_tokens:
            continue
        scores.append(sentence_bleu(refs, hyp_tokens, weights=weights, smoothing_function=SmoothingFunction().method1))
    return sum(scores)/len(scores) if scores else 0.0


def distinct_n(sentences, n=2):
    # count the total number of tokens generated across all sequences
    tokens = [w for s in sentences for w in s.split()]
    if len(tokens) < n:
        return 0.0
    # count the number of unique n-grams (individual words, pairs of words, etc) across all sequences
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    # return the proportion of unique n-grams out of the total number of tokens generated
    return len(set(ngrams))/len(ngrams)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts = construct_wikitext_prompts(n_prompts=20, prompt_len=9, seed=42)

    # Model file paths
    ar_model_path = Path(__file__).resolve().parent.parent / "models" / "gpt2_ar_open_ended"
    nar_model_path = Path(__file__).resolve().parent.parent / "models" / "bert-base-uncased_nar_open_ended"
    diffusion_model_path = Path(__file__).resolve().parent.parent / "models" / "t5-small_diffusion_open_ended"

    # Load each model and set to eval mode for generation
    ar_model, ar_tokenizer = load_ar_model(ar_model_path)
    ar_model.eval()

    nar_model, nar_tokenizer = load_nar_model(nar_model_path)
    nar_model.eval()

    diffusion_model, diffusion_tokenizer, encoder = load_diffusion_model(
        diffusion_model_path,
        tokenizer=T5Tokenizer.from_pretrained("t5-small"),
        device=device,
        d_model=768,
        max_length=64
    )
    diffusion_model.eval()
    encoder.eval()

    # Generate text using each model and save a temporary file with the outputs, so perplexity can be correctly calculated
    models_dict = {"ar": ar_model, "nar": nar_model, "diffusion": diffusion_model}
    tokenizer_dict = {"ar": ar_tokenizer, "nar": nar_tokenizer, "diffusion": diffusion_tokenizer}
    encoder_dict = {"diffusion": encoder}

    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "tasks" / "open_ended_results"
    TEMP_PATH = RESULTS_DIR / "open_ended_temp_outputs.json"
    SUMMARY_PATH = RESULTS_DIR / "open_ended_summary.csv"
    PERPLEXITY_SUMMARY_PATH = RESULTS_DIR / "open_ended_perplexity_summary.csv"

    generate_text_for_task(prompts=prompts, models_dict=models_dict, tokenizer_dict=tokenizer_dict,
                           encoder_dict=encoder_dict, temp_output_path=TEMP_PATH, remove_prompt=False)

    with TEMP_PATH.open("r", encoding="utf-8") as f:
        generated_outputs = json.load(f)

    # Calculate average perplexity and prompt accuracy and save the results
    summarize_perplexity_results(generated_outputs, PERPLEXITY_SUMMARY_PATH, RESULTS_DIR)
    evaluate_open_ended_generation(generated_outputs, SUMMARY_PATH, PERPLEXITY_SUMMARY_PATH)


if __name__ == "__main__":
    main()