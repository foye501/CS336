import torch
import numpy as np
import time
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from adapters import (
    run_transformer_lm, get_adamw_cls, run_get_batch,
    run_get_lr_cosine_schedule, run_cross_entropy,
    run_gradient_clipping, run_save_checkpoint
)

# === Configuration ===
block_size = 1024
context_length = block_size
batch_size = 16
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
rope_theta = 10000.0
learning_rate = 1e-3
warmup_iters = 200
cosine_cycle_iters = 10000
num_iters = 10000
log_interval = 100
eval_interval = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Tokenizer Setup ===
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
vocab_size = tokenizer.vocab_size

# === Tokenization ===
def stream_tokenize_file(file_path, tokenizer, block_size=2048):
    token_chunks = []
    buffer = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading lines"):
            buffer += line.strip() + "\n"
            if len(buffer) > block_size * 10:
                ids = tokenizer(buffer, return_tensors="pt", truncation=False).input_ids[0]
                token_chunks.append(ids)
                buffer = ""
        if buffer:
            ids = tokenizer(buffer, return_tensors="pt", truncation=False).input_ids[0]
            token_chunks.append(ids)
    return torch.cat(token_chunks, dim=0)

print("Start tokenization...")
tokens = stream_tokenize_file("TinyStoriesV2-GPT4-train.txt", tokenizer)
print(f"Tokenization complete. Total tokens: {len(tokens)}")

# === Split into blocks ===
num_blocks = len(tokens) // block_size
train_tensor = tokens[:num_blocks * block_size].reshape(num_blocks, block_size)

# === Convert to numpy format ===
train_data_np = train_tensor.cpu().numpy().astype(np.uint16)
val_data_np = train_data_np[:10000]  # Use first 10K tokens for validation


# === Model Definition ===
class TransformerLM(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v.clone().detach()) for k, v in weights.items()
        })

    def forward(self, x):
        return run_transformer_lm(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights={k: v for k, v in self.weights.items()},
            in_indices=x,
        )

def init_weights(vocab_size, d_model, num_layers, d_ff):
    weights = {
        "token_embeddings.weight": torch.randn(vocab_size, d_model) * 0.01,
        "ln_final.weight": torch.ones(d_model),
        "lm_head.weight": torch.randn(vocab_size, d_model) * 0.01,
    }
    for i in range(num_layers):
        weights.update({
            f"layers.{i}.attn.q_proj.weight": torch.randn(d_model, d_model) * 0.01,
            f"layers.{i}.attn.k_proj.weight": torch.randn(d_model, d_model) * 0.01,
            f"layers.{i}.attn.v_proj.weight": torch.randn(d_model, d_model) * 0.01,
            f"layers.{i}.attn.output_proj.weight": torch.randn(d_model, d_model) * 0.01,
            f"layers.{i}.ln1.weight": torch.ones(d_model),
            f"layers.{i}.ln2.weight": torch.ones(d_model),
            f"layers.{i}.ffn.w1.weight": torch.randn(d_ff, d_model) * 0.01,
            f"layers.{i}.ffn.w2.weight": torch.randn(d_model, d_ff) * 0.01,
            f"layers.{i}.ffn.w3.weight": torch.randn(d_ff, d_model) * 0.01,
        })
    return weights


# === Evaluation ===
def run_eval(model, val_data, tokenizer, context_length, device, batch_size=8):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(10):
            inputs, targets = run_get_batch(val_data, batch_size, context_length, device)
            logits = model(inputs)
            loss = run_cross_entropy(logits[:, :-1], targets[:, 1:])
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


# === Training ===
def train(train_data, val_data):
    weights = init_weights(vocab_size, d_model, num_layers, d_ff)
    model = TransformerLM(weights).to(device)
    optimizer = get_adamw_cls()(model.parameters(), lr=learning_rate)

    for iter in range(num_iters):
        start_time = time.time()

        inputs, targets = run_get_batch(train_data, batch_size, context_length, device)
        logits = model(inputs)

        loss = run_cross_entropy(logits[:, :-1], targets[:, 1:])  # fixed loss

        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), 1.0)
        optimizer.step()

        if iter % log_interval == 0:
            print(f"[{iter:>5}] loss = {loss.item():.4f} ({time.time() - start_time:.2f}s)")

        if iter % eval_interval == 0 and iter > 0:
            val_loss = run_eval(model, val_data, tokenizer, context_length, device)
            print(f"Eval @iter {iter:>5}: val_loss = {val_loss:.4f}")
            run_save_checkpoint(model, optimizer, iter, f"checkpoint_iter{iter}.pt")

train(train_data_np, val_data_np)
