from networkx.drawing.nx_pydot import from_pydot
from transformers import GPT2TokenizerFast
import torch

from .adapters import  run_transformer_lm,get_adamw_cls,run_get_batch,run_get_lr_cosine_schedule,run_cross_entropy,run_gradient_clipping,run_save_checkpoint
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

for iteration in range(1, 10001):
    model.train()
    x, y = run_get_batch(dataset, batch_size, context_length, device)
    logits = model(x)
    logits = logits.view(-1, vocab_size)
    y = y.view(-1)

    loss = run_cross_entropy(logits, y)
    loss.backward()

    run_gradient_clipping(model.parameters(), max_l2_norm=1.0)
    lr = run_get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=learning_rate,
        min_learning_rate=learning_rate * 0.01,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    optimizer.zero_grad()

    if iteration % 100 == 0:
        print(f"Step {iteration}: loss={loss.item():.4f}, lr={lr:.6f}")

    if iteration % 1000 == 0:
        run_save_checkpoint(model, optimizer, iteration, f"checkpoint_{iteration}.pt")