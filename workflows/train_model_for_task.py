import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.preprocess import (load_yelp_formality_labelled, tokenize_dataset, load_wikitext_for_open_ended,
                             generate_teacher_outputs, PrefixSamplerConfig, build_nar_prefix_examples,
                             create_nar_prefix_cont_dataset)
from data.utils import split_dataset
from models import ar, nar, diffusion


# --- Function to coordinate model training / fine-tuning based on model type, base model and task ---

def train_model(
        model_type="ar",
        model_name="gpt2",
        task="controllable",
        n_samples=50000,
        distillation=False,
        teacher_model_path=Path(__file__).resolve().parent.parent / "models" / "gpt2_ar_controllable",
        early_stopping=False,
        mask_prob=0.5,
        num_epochs=30,
        ar_model_path=None
):

    # --- Dataset Loading ---

    if task == "controllable":
        # load the customised formality-labelled Yelp dataset
        labelled_dataset = load_yelp_formality_labelled(n_samples=n_samples)
        splits = split_dataset(labelled_dataset)
        train_raw, eval_raw = splits["train"], splits["test"]

    elif task == "open_ended":
        # load and clean the standard HuggingFace wikitext 103 dataset
        wikitext = load_wikitext_for_open_ended(n_samples=n_samples)
        train_raw, eval_raw = wikitext["train"], wikitext["validation"]

    else:
        raise ValueError(f"Unsupported task {task}")

    print("Dataset loaded")

    # --- Definitions ---

    # define where the model will be saved
    output_dir = str(Path(__file__).resolve().parent.parent / "models" / f"{model_name}_{model_type}_{task}")

    # define the relevant model functions, based on the model_type defined in train_model
    model_module = {"ar": ar, "nar": nar, "diffusion": diffusion}.get(model_type)
    if model_module is None:
        raise ValueError(f"Model type {model_type} not supported")

    load_model_fn = getattr(model_module, f"load_{model_type}_model")

    # --- Model Loading ---
    if model_type == "diffusion":
        model, tokenizer, encoder = load_model_fn(model_name, ar_model_path=ar_model_path)
    elif model_type == "nar":
        print(f"Loading {model_name} model and tokenizer")
        add_control_tokens = (task == "controllable")
        model, tokenizer = load_model_fn(model_name, add_control_tokens)
    else:
        print(f"Loading {model_name} model and tokenizer")
        model, tokenizer = load_model_fn(model_name)

    # --- Training and Evaluation Dataset Tokenization and Prompt Handling ---
    if model_type == "diffusion":
        # use the custom diffusion dataclasses for tokenizing the train and eval datasets according to task
        if task == "controllable":
            train_dataset = diffusion.SimpleTextDataset(train_raw, tokenizer, encoder)
            eval_dataset = diffusion.SimpleTextDataset(eval_raw, tokenizer, encoder)
        elif task == "open_ended":
            train_dataset = diffusion.VariablePrefixContinuationDataset(
                train_raw, tokenizer,
                max_prefix_k=16,
                target_len=64,
                cfg_drop_prob=0.15,
                gap_poisson=None
            )
            eval_dataset = diffusion.VariablePrefixContinuationDataset(eval_raw, tokenizer,max_prefix_k=16,
                                                                       target_len=64, cfg_drop_prob=0.0)


    elif model_type == "nar" and distillation:
        # use the raw dataset when training with knowledge distillation (as it already contains prompt / text examples)
        train_dataset = train_raw
        if task == "open_ended":
            # use a custom dataclass to tokenize and construct prompt examples for eval_dataset
            cfg = PrefixSamplerConfig(max_prefix_k=16, p0=0.6, p_short=0.35, short_lo=3, short_hi=8)
            eval_examples = build_nar_prefix_examples(eval_raw, tokenizer, n_samples // 10, seq_len=256, cfg=cfg,
                                                      seed = 1337)

            eval_dataset = create_nar_prefix_cont_dataset(eval_examples)
        else:
            # for NAR distillation for controllable generation, just tokenize the eval dataset
            eval_dataset = tokenize_dataset(eval_raw, tokenizer, model_type=model_type)

    elif model_type == "nar" and task == "open_ended":
        # for non-knowledge-distillation training, use the custom dataclass to tokenize and construct prompt examples
        # for train and eval datasets
        cfg = PrefixSamplerConfig(max_prefix_k=16, p0=0.6, p_short=0.35, short_lo=3, short_hi=8)

        train_examples = build_nar_prefix_examples(train_raw, tokenizer, n_samples, seq_len=256, cfg=cfg,
                                                   seed=1234)
        eval_examples = build_nar_prefix_examples(eval_raw, tokenizer, n_samples // 10, seq_len=256, cfg=cfg,
                                                      seed = 1337)

        train_dataset = create_nar_prefix_cont_dataset(train_examples)
        eval_dataset = create_nar_prefix_cont_dataset(eval_examples)

    else:
        # for AR model, or NAR controllable generation without knowledge distillation, just tokenize
        train_tokenized = tokenize_dataset(train_raw, tokenizer, model_type=model_type)
        train_dataset = train_tokenized

        eval_tokenized = tokenize_dataset(eval_raw, tokenizer, model_type=model_type)
        eval_dataset = eval_tokenized


    # --- Dynamically define model training functions ---

    train_fn = getattr(model_module, f"train_{model_type}_model")

    if model_type == "diffusion":
        # load in small batches to keep within memory limits
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)

        # use GPU if available, or CPU if not
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        encoder.to(device)

        # define a noise schedule for diffusion
        betas = diffusion.get_cosine_noise_schedule(timesteps=100).to(device)

        # calculate alphas and cumulative products of alphas, which are needed for forward and reverse diffusion steps
        alphas, alpha_hats = diffusion.get_diffusion_parameters(betas)

        # optimizer for model parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_fn(
            model=model,
            encoder=encoder,
            tokenizer=tokenizer,
            dataloader=train_loader,
            eval_dataloader=eval_loader,
            output_dir=output_dir,
            num_epochs=num_epochs,
            betas=betas,
            alpha_hats=alpha_hats,
            optimizer=optimizer,
            device=device
        )

    # for AR and NAR training
    else:
        train_fn(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            task=task,
            distillation=distillation,
            teacher_model_path=teacher_model_path,
            n_samples=n_samples,
            mask_prob=mask_prob,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
        )


if __name__ == "__main__":

    # --- Uncomment relevant lines below to train different model types for different tasks ---


    # --- Controllable generation training ---

    #train_model(model_type="nar", model_name="bert-base-uncased", n_samples=362100)
    #train_model(model_type="ar", model_name="gpt2", n_samples=20000)

    # if using knowledge distillation, first generate the teacher model outputs for training
    #teacher_model_path = Path(__file__).resolve().parent.parent / "models" / "gpt2_ar_controllable"
    #generate_teacher_outputs(teacher_model_path=teacher_model_path, n_samples=50000, batch_size=64)

    #train_model(model_type="nar", model_name="bert-base-uncased", distillation=True, early_stopping=True,
    #            teacher_model_path=teacher_model_path, n_samples=50000)

    #train_model(model_type="diffusion", model_name="t5-small", task="controllable", n_samples=362100, num_epochs=30,
    #            ar_model_path=str(Path(__file__).resolve().parent.parent / "models" / "gpt2_ar_controllable"))


    # --- Open-ended generation training ---

    #train_model(model_type="ar", model_name="gpt2", task="open_ended", early_stopping=True)

    #train_model(model_type="nar", model_name="bert-base-uncased", task="open_ended", n_samples=300000,
    #            early_stopping=True)

    #train_model(model_type="diffusion", model_name="t5-tiny", task="open_ended", n_samples=50000, num_epochs=30,
    #            ar_model_path=str(Path(__file__).resolve().parent.parent / "models" / "gpt2_ar_open_ended"))

    # if using knowledge distillation, first generate the teacher model outputs for training
    teacher_model_path = Path(__file__).resolve().parent.parent / "models" / "gpt2_ar_open_ended"
    generate_teacher_outputs(teacher_model_path=teacher_model_path, task="open_ended", n_samples=50000, batch_size=64)
    train_model(model_type="nar", model_name="bert-base-uncased", task="open_ended", distillation=True,
                early_stopping=True, teacher_model_path=teacher_model_path, n_samples=50000)