from networkx.drawing.nx_pydot import from_pydot
from transformers import GPT2TokenizerFast
import torch

from adapters import  run_transformer_lm,get_adamw_cls,run_get_batch,run_get_lr_cosine_schedule,run_cross_entropy,run_gradient_clipping,run_save_checkpoint
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})  # Optional for padding

# Load and tokenize the dataset
with open("owt_train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text, return_tensors="pt")[0]  # 1D tensor
dataset = tokens.numpy()  # Convert to numpy for compatibility


vocab_size = tokenizer.vocab_size

batch_size = 16
context_length = 128
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
rope_theta = 10000.0
learning_rate = 1e-3
warmup_iters = 200
cosine_cycle_iters = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Random init (or load pre-trained weights)
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

model = TransformerLM(weights).to(device)
optimizer = get_adamw_cls()(model.parameters(), lr=learning_rate)
import time
import numpy as np

def run_eval(model, val_data, tokenizer, context_length, device, batch_size=8):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(10):  # run 10 batches for quick eval
            inputs, targets = run_get_batch(val_data, batch_size, context_length, device)
            logits = run_transformer_lm(
                vocab_size=len(tokenizer.vocab),
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                weights=model.state_dict(),
                in_indices=inputs,
            )
            loss = run_cross_entropy(logits[:, -1], targets[:, -1])
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    with open("owt_train.txt", "rb") as f:
        train_data = np.frombuffer(f.read(), dtype=np.uint8)
    with open("owt_valid.txt", "rb") as f:
        val_data = np.frombuffer(f.read(), dtype=np.uint8)



    # Model

    model.to(device)

    optimizer = get_adamw_cls()(model.parameters(), lr=learning_rate)

    num_iters = 10000
    context_length = 128
    log_interval = 100
    eval_interval = 1000

    for iter in range(num_iters):
        start_time = time.time()

        inputs, targets = run_get_batch(train_data, batch_size=8, context_length=context_length, device=device)
        logits = run_transformer_lm(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights=model.state_dict(),
            in_indices=inputs,
        )

        loss = run_cross_entropy(logits[:, -1], targets[:, -1])

        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), 1.0)
        optimizer.step()

        if iter % log_interval == 0:
            print(f"[{iter:>5}] loss = {loss.item():.4f} ({time.time() - start_time:.2f}s)")

        if iter % eval_interval == 0 and iter > 0:
            val_loss = run_eval(model, val_data, tokenizer, context_length, device)
            print(f"Eval @iter {iter:>5}: val_loss = {val_loss:.4f}")

            # Optional: save checkpoint
            run_save_checkpoint(model, optimizer, iter, f"checkpoint_iter{iter}.pt")

train()