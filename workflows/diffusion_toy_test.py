from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.diffusion import *
from data.utils import format_control_prompt, add_control_tokens_and_resize


# --- tiny sample dataset ---

toy_data = [
    {"sentiment": "Positive", "formality": "unspecified", "text": "This is amazing!"},
    {"sentiment": "Negative", "formality": "unspecified", "text": "I hated it."},
    {"sentiment": "unspecified", "formality": "Formal", "text": "We regret to inform you."},
    {"sentiment": "unspecified", "formality": "Informal", "text": "yeah I'll be there"}
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# use T5 tokenizer and encoder

tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
t5_encoder = T5EncoderModel.from_pretrained('t5-small').to(device)

# Add control tokens and resize tokenizer

add_control_tokens_and_resize(tokenizer, t5_encoder)


# define number of denoising steps, output sequence length and tokenizer config
T = 1000
vocab_size = len(tokenizer)
seq_len = 20
hidden_size = t5_encoder.config.d_model

# load diffusion model
model = DiffusionDecoder(vocab_size, hidden_size, max_length=seq_len).to(device)


# define noise schedule and compute alphas and alpha hats
betas = get_cosine_noise_schedule(timesteps=T).to(device).float()
alphas, alpha_hats = get_diffusion_parameters(betas)

alpha_hats = alpha_hats.to(device).float()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# load toy dataset as a HuggingFace Dataset
dataset = SimpleTextDataset(toy_data, tokenizer, t5_encoder, max_length=seq_len)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# overtrain with large no. of epochs so the model replicates toy data
num_epochs = 1000

# run the training loop
train_diffusion_model(
    model=model,
    encoder=t5_encoder,
    tokenizer=tokenizer,
    num_epochs=num_epochs,
    dataloader=dataloader,
    betas=betas.to(device),
    alpha_hats=alpha_hats.to(device),
    optimizer=optimizer,
    device=device
)


# generate text and print so it can be compared to toy data

generated = generate_diffusion_text(
    model=model,
    tokenizer=tokenizer,
    encoder=t5_encoder,
    cond_texts=["positive: ", "formal: "],
    betas=betas.to(device),
    alpha_hats= alpha_hats.to(device),
    seq_len=10,
    device=device
)

for g in generated:
    print("Generated text: ", g)
