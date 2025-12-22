import torch
import time
import json
import csv
import re
from pathlib import Path
from models.ar import generate_ar_text
from models.nar import generate_nar_text
from models.diffusion import generate_diffusion_text, get_cosine_noise_schedule, get_diffusion_parameters
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
from data.preprocess import format_control_prompt


def generate_from_model(prompt, model_type, model, tokenizer, max_length=128, encoder=None):
    # coordinating function to use the relevant trained model to generate text

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # call the relevant model using the model_type argument
    if model_type == "ar":
        return generate_ar_text(prompt, model, tokenizer, max_length=max_length, knowledge_distillation=False)

    elif model_type == "nar" or model_type == "nar_dist":
        # use mask-predict function to generate text
        return generate_nar_text(prompt, model, tokenizer, max_length=max_length, mask_token="[MASK]", max_iter=10,
                                 sampling=True)
    elif model_type == "diffusion":
        # run the denoising loop to generate text
        return generate_diffusion_text(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            cond_texts=[prompt],
            betas=get_cosine_noise_schedule(1000).to(device),
            alpha_hats=get_diffusion_parameters(get_cosine_noise_schedule(1000).to(device))[1],
            seq_len=32,
            device=device
        )[0]
    else:
        print(f"Model type {model_type} not supported")
        return ""


def timed_generate(prompt, model_type, model, tokenizer, max_length, encoder):
    # wrap generation function with timing to evaluate inference speed

    start_time = time.time()
    output = generate_from_model(prompt, model_type, model, tokenizer, max_length, encoder)
    end_time = time.time()
    return output, end_time - start_time


def generate_batch_outputs(prompt, model_type, model, tokenizer, max_length, encoder, num_iterations):
    # generate multiple samples for the same prompt, collect outputs and average inference time
    # Used to assess average performance metrics across multiple runs, and evaluate diversity between iterations

    model.eval()
    if encoder is not None:
        encoder.eval()

    results = []
    for _ in range(num_iterations):
        s, t = timed_generate(prompt, model_type, model, tokenizer, max_length, encoder)
        results.append((s, t))

    if results:
        generated_sentences, inference_times = zip(*results)
        avg_t = sum(inference_times) / len(inference_times)
    else:
        generated_sentences, avg_t = [], float("nan")
    print(avg_t)

    return {
        "prompt": prompt,
        "generated_sentences": generated_sentences,
        "avg_inference_time": avg_t,
    }


def filter_and_clean_outputs(generated_sentences, remove_prompt=True, prompt=None, sep=":"):
    # remove prompt tokens from generated text output. Includes logic to handle old models with colon separator and
    # newer models without a separator
    # removes artifacts e.g. newline or end of text tags

    cleaned = []
    for s in generated_sentences:
        text = s
        if remove_prompt and prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()
        elif remove_prompt and sep in text:
            text = text.split(sep, 1)[-1].strip()
        else:
            text = text.strip()

        text = text.replace("<|endoftext|>", "").replace("\\n", " ").strip()
        if text:
            cleaned.append(text)
    return cleaned


def generate_text_for_task(prompts, models_dict, tokenizer_dict, encoder_dict=None,
                           temp_output_path="tasks/results/temp_outputs.json", max_length=128,
                           num_iterations=100, remove_prompt=True):

    # Coordinates generation for specific tasks for each model. Iterates over each prompt and model for the task

    encoder_dict = encoder_dict or {}

    outputs_to_evaluate = []

    # Save progress and restart if text has already been generated for a specific prompt / model combination
    temp_output_path = Path(temp_output_path)
    Path(temp_output_path).parent.mkdir(parents=True, exist_ok=True)

    if temp_output_path.exists():
        with temp_output_path.open("r", encoding="utf-8") as f:
            try:
                outputs_to_evaluate = json.load(f)
            except json.JSONDecodeError:
                print("Warning: temp_outputs.json file is corrupted. Restarting generation.")

    existing_keys = {(entry["prompt"], entry["model_type"]) for entry in outputs_to_evaluate if "generated_sentences"
                     in entry and len(entry["generated_sentences"]) >= num_iterations}

    for raw_prompt in prompts:
        if isinstance(raw_prompt, tuple):
            # for controllable generation
            formatted_prompt = format_control_prompt(*raw_prompt)
        else:
            # for open-ended generation
            formatted_prompt = str(raw_prompt)

        for model_type in models_dict:
            key = (formatted_prompt, model_type)
            if key in existing_keys:
                print(f"{key} completed - skipping.")
                continue

            model = models_dict[model_type]
            tokenizer = tokenizer_dict[model_type]
            encoder = encoder_dict.get(model_type, None)
            print(f"Generating text for '{formatted_prompt}' with model {model_type}")

            # run batched text generation
            result = generate_batch_outputs(formatted_prompt, model_type, model, tokenizer, max_length, encoder,
                                            num_iterations)
            result["model_type"] = model_type

            # clean outputs
            result["generated_sentences"] = filter_and_clean_outputs(result["generated_sentences"], remove_prompt,
                                                                     formatted_prompt)

            outputs_to_evaluate.append(result)

            with open(temp_output_path, "w", encoding="utf-8") as f:
                json.dump(outputs_to_evaluate, f, indent=2)

    print(f"All text generated and saved to {temp_output_path}")


def calculate_perplexity(sentences, model_id='gpt2'):
    # Calculate perplexity scores for a list of generated sequences using a reference LLM (default is base model GPT-2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Load the LLM (GPT-2)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        results = []

        for sentence in sentences:
            if not sentence or not sentence.strip():
                print("invalid sentence, skipping")
                continue

            try:
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                labels = inputs["input_ids"].clone()

                # ignore padding tokens - do not want them to contribute to perplexity calc
                labels[inputs["attention_mask"] == 0] = -100

                with torch.no_grad():
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss

                    # use loss as a measure of 'unexpectedness' (i.e. perplexity)
                    perplexity = torch.exp(loss)

                if not torch.isinf(perplexity) and not torch.isnan(perplexity):
                    results.append({
                        "sentence": sentence,
                        "perplexity": perplexity.item()
                    })

            except Exception as e:
                print(f"Error: {e}: sentence: {sentence}")
                continue

        return results

    except Exception as e:
        print(f"Error: {e}")
        return []


def summarize_perplexity_results(outputs_to_evaluate, perplexity_summary_path, results_dir):

    # Calculate and combine perplexity results across all prompts / models

    rows = []

    perplexity_summary_path.parent.mkdir(parents=True, exist_ok=True)

    for item in outputs_to_evaluate:
        prompt = item["prompt"]
        model_type = item["model_type"]
        generated_sentences = item["generated_sentences"]
        avg_inference_time = item["avg_inference_time"]

        # calculate perplexity for each model / prompt
        perplexity_results = calculate_perplexity(generated_sentences)
        print(f"model: {model_type}, Prompt: {prompt}")
        print(f"No of valid perplexity results: {len(perplexity_results) if perplexity_results else 0}")

        if perplexity_results:
            total_perplexity = sum(res["perplexity"] for res in perplexity_results)
            avg_perplexity = total_perplexity / len(perplexity_results)
            print(f"Average perplexity for {model_type} is {avg_perplexity:.2f}")

            prompt_filename_str = re.sub(r"[^a-zA-Z0-9]", "_", prompt.strip())
            details_filename = f"details_{prompt_filename_str}_{model_type}.csv"
            details_path = Path(results_dir) / details_filename

            # save results to csv file
            details_path.parent.mkdir(parents=True, exist_ok=True)
            with open(details_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sentence", "perplexity"])
                writer.writeheader()
                writer.writerows(perplexity_results)
            print(f"Saved generated text with perplexity scores to {details_filename}")

        else:
            avg_perplexity = float("nan")
            print(f"Could not compute perplexity for {model_type}")

        # save a summary table of the results
        rows.append({
            "prompt": prompt,
            "model_type": model_type,
            "avg_perplexity": f"{avg_perplexity:.4f}" if not avg_perplexity != avg_perplexity else "N/A",
            "avg_inference_time": f"{avg_inference_time:.4f}" if isinstance(avg_inference_time, float) else "N/A",
            "example_output": generated_sentences[0] if generated_sentences else "N/A"
        })

        with perplexity_summary_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["prompt", "model_type", "avg_perplexity", "avg_inference_time", "example_output"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Perplexity evaluation complete. Saved to {perplexity_summary_path}")
