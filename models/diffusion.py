import torch
import os
from torch.utils.data import Dataset, DataLoader
import math
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForCausalLM, AutoTokenizer
from data.utils import format_control_prompt
import torch.nn.functional
import random
from tqdm import tqdm


# --- Custom Dataset classes for loading and tokenising datasets ---

class SimpleTextDataset(Dataset):
    # Customised PyTorch Dataset class. Wraps raw data into tokenized encoder/decoder inputs for controllable generation
    # Initialised with sentiment, formality and token masks
    def __init__(self, raw_data, tokenizer, encoder, max_length=64):
        self.examples = []
        self.tokenizer = tokenizer

        for item in raw_data:
            # format and encode prompts and associated example text
            cond_text = format_control_prompt(item["sentiment"], item["formality"])
            target_text = item["text"]
            cond_enc = tokenizer(cond_text, padding='max_length', truncation=True, max_length=max_length)
            target_enc = tokenizer(target_text, padding='max_length', truncation=True, max_length=max_length)
            self.examples.append({
                "encoder_input_ids": torch.tensor(cond_enc["input_ids"]),
                "encoder_attention_mask": torch.tensor(cond_enc["attention_mask"]),
                "decoder_input_ids": torch.tensor(target_enc["input_ids"]),
                "cond_text": cond_text,
                "sentiment": item["sentiment"],
                "formality": item["formality"],
                "decoder_attention_mask": (torch.tensor(target_enc["attention_mask"]))
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class VariablePrefixContinuationDataset(Dataset):
    # Customised PyTorch Dataset class. Creates prefix (prompt) / continuation pairs for open-ended generation
    def __init__(
            self,
            docs,
            tokenizer,
            max_prefix_k=64,    # max length of prefix (prompt tokens)
            target_len=64,      # max length of continuation
            cfg_drop_prob=0.1,  # probability of dropping encoder condition, i.e. unconditional training
            gap_poisson=None,   # optional Poisson gap before continuation
            doc_cap=None,       # max tokens ber example
            p0=0.6,             # probability of empty prefix (no prompt, but still conditional)
            p_short=0.35,       # prob of short prefix
            short_lo=3,         # min length of 'short' prefix
            short_hi=8          # max length of 'short' prefix
    ):

        self.tok = tokenizer
        self.K = max_prefix_k
        self.L = target_len
        self.cfgp = cfg_drop_prob
        self.gap_lambda = gap_poisson
        self.doc_cap = doc_cap or (self.K + self.L + 64)    # avoid very long texts
        self.pad_id = self.tok.pad_token_id
        self.p0 = p0
        self.p_short = p_short
        self.short_lo = short_lo
        self.short_hi = short_hi

        self.p0 = max(0.0, min(1.0, self.p0))
        self.p_short = max(0.0, min(1.0 - self.p0, self.p_short))

        self.docs = []
        for d in docs:
            ids = self.tok(d["text"], add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            if len(ids) > self.doc_cap:
                ids = ids[: self.doc_cap]
            self.docs.append(ids)

        self.lengths = [len(x) for x in self.docs]
        self.cum = []
        s = 0
        for n in self.lengths:
            s += max(0, n - 2)
            self.cum.append(s)
        self.total = s

    def _sample_doc(self):
        # enables selection of examples from potential documents, weighted by how many usable prefix / continuation
        # pairs they have (long docs have more valid pairs, so sampled more frequently)
        r = random.randint(0, max(0, self.total - 1))
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum[mid] > r:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __len__(self):
        return max(1, len(self.docs) * 64)

    def _sample_k(self, n_tokens_in_doc):
        # sample a prefix of length k and cut=off point c from a given sequence

        # cap k so that at least one token in the sequence is available for continuation
        k_max = min(self.K, max(1, n_tokens_in_doc - 1))
        r = random.random()

        # if prob < p0, use no prefix ( empty prompt)
        if r < self.p0:
            return 0

        # if prob > p0 and < p_short, sample a short prefix of length 3-8 tokens
        if r < self.p0 + self.p_short:
            lo = min(self.short_lo, k_max)
            hi = min(self.short_hi, k_max)
            if hi >= 1 and hi >= lo:
                return random.randint(max(1, lo), hi)
            return random.randint(1, k_max)

        # if prob > p_short, sample a longer prefix
        lo = max(9, 1)
        if k_max >= lo:
            return random.randint(lo, k_max)

        return random.randint(1, k_max)

    def __getitem__(self, _):
        # choose a document, weighted by its length
        ids = self.docs[self._sample_doc()]
        n = len(ids)

        # ensure doc has at least 1 prompt token and 1 continuation token
        if n < 2:
            return self.__getitem__(0)

        # determine prefix length (biased towards short prompts)
        k = self._sample_k(n)

        # choose a cut-off point between prompt and continuation (making sure at there is least 1 continuation token)
        c = random.randint(k, max(k + 1, n - 1) - 1)

        # optional small gap between prompt and continuation
        gap = 0
        if self.gap_lambda:
            # Poisson-like distribution to determine target start point
            p = 1.0 / (1.0 + float(self.gap_lambda))
            while random.random() > p and c + gap +1 < n:
                gap += 1
                if c + gap >= n - 1:
                    break

        # continuation starts after prefix + gap (leaving at least 1 token for continuation)
        start_tgt = min(c + gap, n - 1)

        pad = self.pad_id

        # build encoder inputs, length K. Initialise as all PAD and mask as 0
        enc_ids = torch.full((self.K,), pad, dtype=torch.long)    # all pad tokens
        enc_mask = torch.zeros((self.K,), dtype=torch.long)    # all zeros


        # copy and craete mask for k tokens before c
        if k > 0:
            prefix_tokens = ids[max(0, c - k): c]
            prefix_tokens = prefix_tokens[-k:]
            prefix_len = min(k, self.K, len(prefix_tokens))
            if prefix_len > 0:
                enc_ids[:prefix_len] = torch.tensor(prefix_tokens[:prefix_len], dtype=torch.long)
                enc_mask[:prefix_len] = 1

        # construct decoder inputs and targets. Use tokens from start_tgt (start of continuation); pad or truncate to L
        # create corresponding mask
        target_tokens = ids[start_tgt: start_tgt + self.L]
        dec_ids = torch.full((self.L,), pad, dtype=torch.long)
        dec_mask = torch.zeros((self.L,), dtype=torch.long)
        target_len = min(len(target_tokens), self.L)
        if target_len > 0:
            dec_ids[:target_len] = torch.tensor(target_tokens[:target_len], dtype=torch.long)
            dec_mask[:target_len] = 1

        # flag for classifier-free guidance dropout
        cfgd_flag = 1
        if self.cfgp > 0 and random.random() < self.cfgp:
            cfgd_flag = 0

        return {
            "encoder_input_ids": enc_ids,           # [K]
            "encoder_attention_mask": enc_mask,     # [K] (1 for prefix tokens, 0 for rest)
            "decoder_input_ids": dec_ids,           # [L] continuation tokens
            "decoder_attention_mask": dec_mask,     # [L] (1 for real tokens, 0 for PAD tokens)
            "cfgd_flag": torch.tensor(cfgd_flag, dtype=torch.long),    # 1=use conditioning, 0= no conditioning
        }





# --- Custom Diffusion Decoder and Tokeniser (based on HuggingFace Transformer) ---

class DiffusionDecoder(torch.nn.Module):
    # customised PyTorch Transformer-based decoder for denoising
    # initialised with token, position and time embeddings
    # optional cross-attention to encoder (conditioning)
    # linear output is tied to token embeddings
    def __init__(self, vocab_size, d_model, max_length):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_length, d_model)
        self.time_embedding = torch.nn.Embedding(1000, d_model)
        self.embed_norm = torch.nn.LayerNorm(d_model, elementwise_affine=True)
        self.cross_attn = torch.nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=0.1,
                                                         norm_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.output = torch.nn.Linear(d_model, vocab_size, bias=True)
        self.output.weight = self.token_embedding.weight
        self.cond_proj = torch.nn.Linear(512, d_model)

    '''
    def forward(self, x, t):
        # old version of forward process
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(positions)
        x = x + self.time_embedding(t).unsqueeze(1)
        x = self.transformer(x)
        return self.output(x)
    '''

    def forward_from_embedding(self, x_emb, t, cond):
        # forward denoising process starting with noisy token embeddings

        # ensure consistent device / dtype use
        dev = x_emb.device
        dtyp = next(self.parameters()).dtype

        # embed timestep and cast across the sequence length
        t = t.to(device=dev, dtype=torch.long)

        t_vec = self.time_embedding(t)
        if t_vec.dim() == 2:
            t_vec = t_vec.unsqueeze(1)

        # add time embeddings to token embeddings and normalise
        x = self.embed_norm(x_emb + t_vec).to(dtyp).contiguous()

        # align encoder conditioning with decoder hidden size
        cond_proj = self.cond_proj(cond).to(dtyp).contiguous()
        if not torch.isfinite(cond_proj).all():
            cond_proj = torch.nan_to_num(cond_proj, nan=0.0, posinf=0.0, neginf=0.0)

        # cross-attention between noisy toekns and conditioning prompt
        x, _ = self.cross_attn(x, cond_proj, cond_proj)

        # Transformer encoder stack for extra context
        x = self.transformer(x.contiguous())

        # shared weight matrix between embeddings and output layer
        W = self.output.weight
        b = self.output.bias

        # cast to float32 for stability and then back to original dtype
        x32 = x.to(torch.float32).contiguous()
        W32 = W.to(torch.float32).contiguous()

        if not torch.isfinite(W32).all():
            W32 = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

        logits32 = torch.matmul(x32, W32.t())
        if b is not None:
            logits32 = logits32 + b.to(torch.float32)

        logits = logits32.to(W.dtype)

        return logits


# --- Helper functions for training diffusion model ---

def get_linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    # use torch linspace method to create a 1D tensor with size = timesteps. Betas increase evenly across timesteps.
    return torch.linspace(beta_start, beta_end, timesteps)


def get_cosine_noise_schedule(timesteps, s=0.008):
    # use torch to create a smooth noising schedule by calculating cumulative alpha hats and get betas from ratios
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_hats = f / f[0]
    betas = 1 - (alpha_hats[1:] / alpha_hats[:-1])
    return betas.clamp(1e-8, 0.999)    # restrict values to within this range - prevents NaN / inf issues

def get_diffusion_parameters(betas):
    # calculate alpha (1 - beta) per step, and cumulative products, alpha hats (for forward / reverse process)
    alphas = 1.0 - betas
    alpha_hats = torch.cumprod(alphas, dim=0)
    return alphas, alpha_hats


def forward_diffusion_sample(x_emb, t, alpha_hats):
    # apply random noise at timestep t. Returns example with added noise and the noise itself
    noise = torch.randn_like(x_emb)
    sqrt_alpha_hat = torch.sqrt(alpha_hats[t])[:, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hats[t])[:, None, None]
    return sqrt_alpha_hat * x_emb + sqrt_one_minus_alpha_hat * noise, noise



# --- Functions for loading and training diffusion model ---

def load_diffusion_model(model_path="t5-small", tokenizer=None, ar_model_path=None, encoder_name="t5-small",
                         max_length=64, d_model=None,
                         device=None, **kwargs):
    # load diffusion decoder, standard T5 encoder and tokenizer
    print("Loading diffusion model")
    device= device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        # in case tokenizer is not saved with model
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_path)
        except OSError:
            tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load encoder weights from previous checkpoint if possible, or from HF if not
    try:
        encoder = T5EncoderModel.from_pretrained(model_path)
    except OSError:
        encoder = T5EncoderModel.from_pretrained(encoder_name)

    # Optionally, use AR model for weights inititalisation and d_model
    if ar_model_path is not None:
        print("Initialising diffusion decoder with token_embedding from AR model")
        ar_model = AutoModelForCausalLM.from_pretrained(ar_model_path)
        d_model = ar_model.config.hidden_size
    else:
        d_model = d_model or 512

    decoder = DiffusionDecoder(vocab_size=tokenizer.vocab_size, d_model=d_model, max_length=max_length)

    weights_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(weights_path):
        decoder.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded fine-tuned diffusion decoder weights")

    elif ar_model_path is not None:
        with torch.no_grad():
            # copy vocab rows for shared token strings
            ar_tok = AutoTokenizer.from_pretrained(ar_model_path)
            ar_embeddings = ar_model.get_input_embeddings().weight.data
            diffusion_embeddings = decoder.token_embedding.weight.data

            torch.nn.init.normal_(diffusion_embeddings, mean=0.0, std=float(ar_embeddings.std().item()))

            t5_vocab = tokenizer.get_vocab()
            ar_vocab = ar_tok.get_vocab()

            n_copied = 0
            for tok_str, t5_id in t5_vocab.items():
                ar_id = ar_vocab.get(tok_str)
                if ar_id is not None and ar_id < ar_embeddings.size(0) and t5_id < diffusion_embeddings.size(0):
                    diffusion_embeddings[t5_id].copy_(ar_embeddings[ar_id])
                    n_copied += 1

            print(f"Initialised token_embedding from AR model for {n_copied}/{len(t5_vocab)} shared tokens")

    decoder.to(device)
    encoder.to(device)

    return (decoder, tokenizer, encoder)


def train_diffusion_model(
        model,
        encoder,
        tokenizer,
        num_epochs,
        dataloader,
        betas,
        alpha_hats,
        optimizer,
        device,
        output_dir,
        eval_dataloader=None,
        use_early_stopping=True,
        patience=4,
        show_progress=True
):
    # manage training loops for diffusion model

    print("Beginning training")
    model.train()
    encoder.eval()
    best_eval_loss = float("inf")
    epochs_no_improvement = 0
    start_epoch = 0

    os.makedirs(output_dir, exist_ok=True)

    # resume from previous checkpoint if available
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        print("Previous checkpoints found")
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        checkpoint_path = os.path.join(output_dir, latest_checkpoint, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_eval_loss = checkpoint.get('best_eval_loss', best_eval_loss)
            epochs_no_improvement = checkpoint.get('epochs_no_improvement', 0)
            print(f"Resuming training from epoch {start_epoch}")

    model.to(device)
    encoder.to(device)

    steps_per_epoch = len(dataloader)
    total_steps = max(0, (num_epochs - start_epoch) * steps_per_epoch)
    step_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, disable=not show_progress)

    for epoch in range(start_epoch, num_epochs):
        epoch_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)

        # use tqdm for progress bar with epoch no. and loss
        total_loss = 0.0
        running = 0.0

        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True,
                         leave=False, disable=not show_progress)

        for step, batch in enumerate(epoch_bar, start=1):
            enc_ids = batch["encoder_input_ids"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            target_ids = batch["decoder_input_ids"].to(device)
            target_mask = batch.get("decoder_attention_mask")

            if target_mask is None:
                target_mask = (target_ids != tokenizer.pad_token_id).long()
            target_mask = target_mask.to(device)

            # classifier-free guidance (1 = conditional, 0 = unconditional)
            cfgd = batch.get("cfgd_flag")
            if cfgd is None:
                cfgd = torch.ones(target_ids.size(0), dtype=torch.long)
                cfgd = cfgd.to(device)
            else:
                cfgd = cfgd.to(device)


            # skip encoder if k=0 (unconditional training)
            if enc_mask.sum().item() == 0:
                B, K = enc_ids.shape
                H = encoder.config.d_model

                cond_embeddings = torch.zeros(B, K, H, device=device, dtype=model.token_embedding.weight.dtype)

            else:
                with torch.no_grad():
                    enc_out = encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state

                if cfgd.ndim == 1:
                    cfgd = cfgd = cfgd.view(-1, 1, 1)
                cfgd = cfgd.to(device)

                cond_embeddings = enc_out * cfgd

            # construct target token embeddings
            positions = torch.arange(target_ids.shape[1], device=device).unsqueeze(0)
            target_emb = model.token_embedding(target_ids) + model.position_embedding(positions)

            # sample timestep and add noise
            t = torch.randint(0, len(betas), (target_emb.size(0),), device=device)
            x_noisy, _ = forward_diffusion_sample(target_emb, t, alpha_hats)

            # denoise and predict token logits
            logits = model.forward_from_embedding(x_noisy, t, cond_embeddings)

            # masked cross-entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            mask_flat = target_mask.view(-1).bool()

            if mask_flat.any():
                loss = torch.nn.functional.cross_entropy(
                    logits_flat[mask_flat],
                    targets_flat[mask_flat],
                    reduction='mean'
                )
            else:
                loss = logits_flat.sum() * 0.0    # just in case a batch with all pad tokens occurs

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            running += loss.item()
            if show_progress:
                epoch_bar.set_postfix(avg=f"{running/step:.4f}")
                step_bar.set_postfix(epoch=f"{epoch+1}/{num_epochs}")
                step_bar.update(1)

        print(f"Epoch {epoch}/{num_epochs}: Loss = {total_loss / len(dataloader):.4f}")

        # Evaluate model on eval dataset for early stopping

        if eval_dataloader is not None:
            eval_loss = evaluate_diffusion_model(model, encoder, tokenizer, eval_dataloader, betas, alpha_hats, device)
            if eval_loss < best_eval_loss:
                epochs_no_improvement = 0
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                print("Loss improved, saving this version as best_model.pt")
            else:
                epochs_no_improvement += 1
                print(f"No improvement after {epochs_no_improvement} epochs")

            if use_early_stopping and epochs_no_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # save checkpoint
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eval_loss": best_eval_loss,
                    "epochs_no_improvement": epochs_no_improvement
                    }, os.path.join(epoch_dir, "checkpoint.pth"))

    step_bar.close()
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)


def evaluate_diffusion_model(model, encoder, tokenizer, dataloader, betas, alpha_hats, device):
    # run evaluation loop using eval_dataset
    # returns mean loss across batches
    model.eval()
    encoder.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", dynamic_ncols=True, leave=False):
            # get batch and masks
            enc_ids = batch["encoder_input_ids"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            target_ids = batch["decoder_input_ids"].to(device)
            target_mask = batch.get("decoder_attention_mask")

            # if no target mask, assume all non-pad positions are valid targets
            if target_mask is None:
                target_mask = (target_ids != tokenizer.pad_token_id).long()
            target_mask = target_mask.to(device)

            # encode prompt (encoder is frozen)
            enc_out = encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state

            # classifier-free guidance (1 = conditional, 0 = unconditional)
            cfgd = batch.get("cfgd_flag")
            if cfgd is None:
                cfgd = torch.ones(enc_out.size(0), 1, 1, device=enc_out.device, dtype=enc_out.dtype)
            else:
                if cfgd.ndim == 1:
                    cfgd = cfgd.view(-1, 1, 1)
                cfgd = cfgd.to(device=enc_out.device, dtype=enc_out.dtype)

            # ignore conditional embeddings with classifier-free guidance
            cond_embeddings = enc_out * cfgd

            # construct target embeddings (clean sequence the model should reproduce)
            positions = torch.arange(target_ids.shape[1], device=device).unsqueeze(0)
            target_emb = model.token_embedding(target_ids) + model.position_embedding(positions)

            # forward diffusion - add noise at timestep t. Random t ensures evaluation across all noise levels
            t = torch.randint(0, len(betas), (target_emb.size(0),), device=device)
            x_noisy, _ = forward_diffusion_sample(target_emb, t, alpha_hats)

            # denoising prediction
            logits = model.forward_from_embedding(x_noisy, t, cond_embeddings)

            # cross-entropy for non-pad tokens only. Flatten to (B*L, V) so only valid positions are indexed
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            mask_flat = target_mask.view(-1).bool()

            if mask_flat.any():
                # ignore index for pad tokens
                loss = torch.nn.functional.cross_entropy(
                    logits_flat[mask_flat],    # token id predictions
                    targets_flat[mask_flat],   # actual ids (for non-pad tokens)
                    reduction="mean"
                )
            else:
                # if all tokens are PAD, loss=0
                loss = logits_flat.sum() * 0.0

            total_loss += loss.item()

    # return mean loss for batch
    return total_loss / len(dataloader)


# --- Functions to generate text using trained model ---

'''

def generate_diffusion_text(model, tokenizer, encoder, cond_texts, betas, alpha_hats, seq_len, device):
    #Inference method 1

    # suppress control prompt tokens during generation
    prompt_strings = [s for s in tokenizer.additional_special_tokens if s]

    banned_ids = set()
    for string in prompt_strings:
        ids = tokenizer.encode(string, add_special_tokens=False)
        banned_ids.update(ids)
    if len(banned_ids) == 0:
        banned_ids_t = torch.empty(0, dtype=torch.long, device=device)
    else:
        banned_ids_t = torch.tensor(sorted(banned_ids), dtype=torch.long, device=device)

    def apply_vocab_mask(logits):
        if banned_ids_t.numel() == 0:
            return logits
        logits[:, :, banned_ids_t] = float('-inf')
        return logits

    model.eval()
    encoder.eval()

    with torch.no_grad():
        # encode the control tokens (conditioning text) as context for text generation
        inputs = tokenizer(cond_texts, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
        encoder_output = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state

        alphas = 1.0 - betas
        x_t = torch.randn(inputs.input_ids.size(0), seq_len, model.token_embedding.embedding_dim).to(device)

        # run the denoising loop
        for t_inv in reversed(range(len(betas))):
            t = torch.full((x_t.size(0),), t_inv, dtype=torch.long, device=device)

            # predict token logits at step t
            logits = model.forward_from_embedding(x_t, t, encoder_output)

            # block control token IDs so they don't appear in generated text
            logits = apply_vocab_mask(logits)

            # choose the most likely token for x_0
            x_0_pred = model.token_embedding(torch.argmax(logits, dim=-1))
            if t_inv > 0:
                noise = torch.randn_like(x_t)
                alpha_t = alphas[t].unsqueeze(-1).unsqueeze(-1)
                beta_t = betas[t].unsqueeze(-1).unsqueeze(-1)
                x_t = torch.sqrt(alpha_t) * x_0_pred + torch.sqrt(beta_t) * noise
            else:
                x_t = x_0_pred

        final_logits = model.output(x_t)
        final_logits = apply_vocab_mask(final_logits)   # apply the control token blocking again just in case

        token_ids = torch.argmax(final_logits, dim=-1)
        #probs = torch.nn.functional.softmax(final_logits, dim=-1)
        #token_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0),probs.size(1))

        # decode from token ids to tokens
        decoded_texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return decoded_texts
'''

def generate_diffusion_text(model, tokenizer, encoder, cond_texts, betas, alpha_hats, seq_len, device, temperature=0.8,
                            top_p=0.9, top_k=0, repetition_penalty=1.5, use_soft_embeddings=True):
    # Inference method 2
    model.eval()
    encoder.eval()

    # encode the control tokens / conditioning text as context for text generation
    inputs = tokenizer(cond_texts, padding=True, truncation=True, return_tensors="pt", max_length=64)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    encoder_output = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
    decoder_param = next(model.parameters())
    decoder_device = decoder_param.device
    decoder_dtype = decoder_param.dtype

    encoder_output = encoder_output.to(device=decoder_device, dtype=decoder_dtype)
    encoder_output = torch.nan_to_num(encoder_output, nan=0.0, posinf=0.0, neginf=0.0)
    encoder_output = encoder_output.clamp(-10.0, 10.0)

    # denoise loop using sample()
    x_t = sample(model, encoder_output, betas, alpha_hats, seq_len, device, tokenizer, num_steps=len(betas),
                 top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                 use_soft_embeddings=use_soft_embeddings)

    W = model.output.weight
    b = model.output.bias
    x32 = x_t.to(torch.float32).contiguous()
    W32 = W.to(torch.float32).contiguous()
    logits32 = torch.matmul(x32, W32.t())
    if b is not None:
        logits32 = logits32 + b.to(torch.float32)
    logits = logits32.to(W.dtype)

    # ensure control tokens do not leak into generated output - assign v low probability
    banned = prompt_tokens_to_suppress(tokenizer, device)
    if banned.numel() > 0:
        banned = banned.to(device=decoder_device, dtype=torch.long)
        V = logits.size(-1)
        safe_banned = banned[(banned >= 0) & (banned < V)]
        if safe_banned.numel() > 0:
                logits[:, :, safe_banned] = -1e9

    token_ids = torch.argmax(logits, dim=-1)
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


# --- Helper functions for sampling and filtering during generation ---

@torch.no_grad()
def sample(model, encoder_output, betas, alpha_hats, seq_len, device, tokenizer=None, num_steps=100, temperature=0.8,
                            top_p=0.9, top_k=0, repetition_penalty=1.2, use_soft_embeddings=True):

    param = next(model.parameters())
    device = param.device
    B = encoder_output.size(0)    # batch size
    L = seq_len    # sequence length
    d_model = model.token_embedding.embedding_dim
    dtype = model.token_embedding.weight.dtype
    E_weight = model.token_embedding.weight    # embedding matrix

    # clamp alpha_hats to avoid masking broken denoising schedule
    alpha_hats = alpha_hats.to(device=device, dtype=torch.float32)
    alpha_hats = alpha_hats.clamp(min=1e-6, max=1 - 1e-6)
    num_steps = min(num_steps, len(alpha_hats))

    # init and cast x_t
    x_t = torch.randn(B, seq_len, d_model, dtype=torch.float32, device=device).mul_(0.02).to(dtype)

    # suppress control prompt tokens to avoid leakage
    banned_ids_t = None
    if tokenizer is not None:
        banned_ids_t = prompt_tokens_to_suppress(tokenizer, device)
        if banned_ids_t is not None and banned_ids_t.numel() > 0:
            banned_ids_t = banned_ids_t.to(device=device, dtype=torch.long)

    # initialise repetition mask state
    seen_mask = None
    if repetition_penalty is not None and repetition_penalty > 1.0:
        seen_mask = torch.zeros(B, E_weight.size(0), dtype=torch.bool, device=device)

    # predict logits at each timestep t_inv in reverse (denoising) process
    for t_inv in reversed(range(num_steps)):
        t = torch.full((B,), t_inv, device=device, dtype=torch.long)

        logits = model.forward_from_embedding(x_t, t, encoder_output)

        B_now, L_now, V_now = logits.shape

        # ensure seen_mask exists and is aligned with sequence tensor shape
        if repetition_penalty is not None and repetition_penalty > 1.0:
            seen_mask = correct_seen_mask(seen_mask, B, V_now, logits.device)
        else:
            seen_mask = None

        # block control prompt tokens if present
        if banned_ids_t is not None and banned_ids_t.numel() > 0:
            safe_ids = banned_ids_t[(banned_ids_t >=0) & (banned_ids_t < V_now)]
            if safe_ids.numel() > 0:
                logits[:, :, safe_ids] = -1e9

        # temperature and flatten to 2D
        flat_logits = (logits / max(1e-6, temperature)).contiguous().view(B * L, V_now)

        #apply repetition penalty
        seen_mask = apply_repetition_penalty_flat(flat_logits, seen_mask, L, repetition_penalty)

        # apply probability filters
        flat_logits = top_k_top_p_filtering(flat_logits, top_k=top_k, top_p=top_p)

        # safety check
        row_all_neg_inf = torch.isneginf(flat_logits).all(dim=-1)
        if row_all_neg_inf.any():
            safe_id = getattr(tokenizer, "unk_token_id", 0) if tokenizer is not None else 0
            safe_id = safe_id if 0 <= safe_id < V_now else 0
            flat_logits[row_all_neg_inf, int(safe_id)] = 0.0

        # predict probabilities from logits
        probs = torch.nn.functional.softmax(flat_logits, dim=-1)
        probs = probs.to(torch.float32).contiguous()

        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs.clamp_(min=1e-4, max=1.0)   # restrict values to avoid NaN / inf problems

        rows_sums = probs.sum(dim=-1, keepdim=True)
        zero_rows = rows_sums.squeeze(-1) <= 0
        if zero_rows.any():
            safe_id = int(getattr(tokenizer, "unk_token_id", 0) if tokenizer is not None else 0)
            safe_id = safe_id if 0 <= safe_id < V_now else 0
            probs[zero_rows] = 0
            probs[zero_rows, safe_id] = 1.0
            rows_sums = probs.sum(dim=-1, keepdim=True)

        probs /= rows_sums


        # predict x_0 in embedding space from probabilities

        if use_soft_embeddings:
            if seen_mask is not None:
                top1 = probs.argmax(dim=-1).view(B, L)   # ensure shape matches (B,L)
                top1 = top1.clamp_(0, V_now - 1)

                seen_mask = correct_seen_mask(seen_mask, B, V_now, logits.device)

                update = torch.nn.functional.one_hot(top1, num_classes=V_now).to(dtype=torch.bool).any(dim=1)
                seen_mask |= update

            E_slice = E_weight[:V_now, :].to(device)
            x_0_flat32 = torch.matmul(probs, E_slice.to(torch.float32).contiguous())
            x_0_pred32 = x_0_flat32.view(B, L, d_model)

            x_0_pred32 = torch.nan_to_num(x_0_pred32, nan=0.0, posinf=0.0, neginf=0.0)
            x_0_pred32.clamp_(-10.0, 10.0)

            x_0_pred = x_0_pred32.to(E_slice.dtype)

        else:
            sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            sampled_ids = sampled_ids.view(B, L)
            sampled_ids.clamp_(0, V_now -1)

            if seen_mask is not None:
                for b in range(B):
                    seen_mask[b, sampled_ids[b]] = True

            x_0_pred = model.token_embedding.weight.to(device)[sampled_ids]
            x_0_pred32 = x_0_pred.to(torch.float32)
            x_0_pred32 = torch.nan_to_num(x_0_pred32, nan=0.0, posinf=0.0, neginf=0.0)

            x_0_pred32.clamp_(-10.0, 10.0)
            x_0_pred = x_0_pred32.to(x_0_pred.dtype)


        #diffusion update
        if t_inv > 0:
            alpha_hat_prev = alpha_hats[t_inv - 1].to(device)
            sqrt_alpha_hat_prev = torch.sqrt(alpha_hat_prev).view(1, 1, 1)
            sqrt_one_minus_prev = torch.sqrt(1 - alpha_hat_prev).view(1, 1, 1)

            # safely handle any NaN / Inf values in x_0_pred
            if not torch.isfinite(x_0_pred).all():
                x_0_pred = torch.where(torch.isfinite(x_0_pred), x_0_pred, x_t)

            noise = torch.randn_like(x_t)
            x_t = sqrt_alpha_hat_prev * x_0_pred + sqrt_one_minus_prev * noise

        else:
            x_t = x_0_pred

    return x_t


def prompt_tokens_to_suppress(tokenizer, device):
    # suppress control prompt tokens during generation
    prompt_strings = [s for s in tokenizer.additional_special_tokens if s]

    banned_ids = set()
    for string in prompt_strings:
        ids = tokenizer.encode(string, add_special_tokens=False)
        banned_ids.update(ids)
    # keep EOS and PAD tokens
    if tokenizer.eos_token_id is not None:
        banned_ids.discard(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        banned_ids.discard(tokenizer.pad_token_id)

    if len(banned_ids) == 0:
        banned_ids_t = torch.empty(0, dtype=torch.long, device=device)
    else:
        banned_ids_t = torch.tensor(sorted(banned_ids), dtype=torch.long, device=device)

    return banned_ids_t

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):

    N, V = logits.shape
    out = logits

    # top-k: filter top k candidate tokens
    if top_k > 0:
        k = min(top_k, V)
        kth_vals = torch.topk(out, k, dim=-1).values[..., -1].unsqueeze(-1)
        out = out.masked_fill(out < kth_vals, -1e9)

    # top-p: sort candidate tokens by probability
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        mask = cumprobs > top_p
        # keep at least one token
        mask[..., 0] = False
        sorted_filtered = sorted_logits.masked_fill(mask, -1e9)
        out = torch.full_like(out, -1e9).scatter(-1, sorted_indices, sorted_filtered)

    return out


def apply_repetition_penalty_flat(flat_logits, seen_mask, seq_len, penalty):
    # reduce probability of choosing a token which already appears in the sequence

    if penalty == 1.0 or penalty is None or seen_mask is None:
        return

    # check shape
    BxL, V = flat_logits.shape
    B = BxL // seq_len

    # ensure seen_mask exists and is correctly aligned
    seen_mask = correct_seen_mask(seen_mask, B, V, flat_logits.device)

    # expand mask to (B*L, V)
    mask_expanded = seen_mask.unsqueeze(1).expand(B, seq_len, V).reshape(B * seq_len, V)

    # broadcasted scaling to avoid cuda indexing issues
    scale = 1.0 + (penalty - 1.0) * mask_expanded.to(dtype=flat_logits.dtype)
    flat_logits /= scale

    return seen_mask


def correct_seen_mask(seen_mask, B, V, device):
    # ensure seen_mask tensor matches the shape of batch x vocab
    if seen_mask is None:
        return torch.zeros(B, V, dtype=torch.bool, device=device)

    if seen_mask.device != device:
        seen_mask = seen_mask.to(device)

    if seen_mask.shape[0] != B:
        raise RuntimeError(f"seen_mask batch {seen_mask.shape[0]} != B {B}")

    if seen_mask.shape[1] != V:
        if seen_mask.shape[1] < V:
            pad = V - seen_mask.shape[1]
            seen_mask = torch.nn.functional.pad(seen_mask, (0, pad), value=False)
        else:
            seen_mask = seen_mask[:, :V]

    return seen_mask