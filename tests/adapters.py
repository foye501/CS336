from __future__ import annotations

import os
from concurrent.futures.process import ProcessPoolExecutor
from typing import IO, Any, BinaryIO
from collections.abc import Iterable

import numpy as np
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from mpmath import cos_sin
from scipy.signal import max_len_seq
from sympy.geometry.entity import scale
from torch import Tensor
from torch.ao.nn.quantized.functional import linear
from torch.distributions.utils import logits_to_probs
from torch.utils.checkpoint import checkpoint


# from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import attention


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    # raise NotImplementedError
    assert weights.shape == (d_out, d_in)
    assert in_features.shape[-1] == d_in
    return in_features@weights.T

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight


    proj1 = run_linear(d_model,d_ff,w1_weight,in_features)
    proj2 =run_linear(d_model,d_ff,w3_weight,in_features)
    gated =run_silu(proj1)
    return run_linear(d_ff,d_model,w2_weight,gated*proj2)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]

    attention_weight = Q@K.transpose(-2,-1)/torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    if mask is not None:
        attention_weight =attention_weight.masked_fill(mask==0,float("-inf"))

    attention_weight = run_softmax(attention_weight,dim=-1)
    return attention_weight@V




def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_in = in_features.shape[-1]
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    batch = in_features.shape[:-2]
    sequence_len = in_features.shape[-2]

    d_att_k = d_k // num_heads
    d_att_v = d_v // num_heads

    Q = run_linear(d_in, d_k, q_proj_weight, in_features).view(*batch, sequence_len, num_heads, d_att_k).transpose(-3,-2)
    K = run_linear(d_in, d_k, k_proj_weight, in_features).view(*batch, sequence_len, num_heads, d_att_k).transpose(-3,-2)
    V = run_linear(d_in, d_v, v_proj_weight, in_features).view(*batch, sequence_len, num_heads, d_att_v).transpose(-3,-2)
    causal_mask = torch.tril(torch.ones(sequence_len, sequence_len, device=in_features.device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    out = run_scaled_dot_product_attention(Q, K, V,mask=causal_mask)  # [*, num_heads, seq, d_att_v]
    out = out.transpose(-3, -2).contiguous().view(*batch, sequence_len, d_v)

    return run_linear(d_v, d_model, o_proj_weight, out)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_in = in_features.shape[-1]
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    batch = in_features.shape[:-2]
    sequence_len = in_features.shape[-2]

    d_att_k = d_k // num_heads
    d_att_v = d_v // num_heads

    Q = run_linear(d_in, d_k, q_proj_weight, in_features).view(*batch, sequence_len, num_heads, d_att_k).transpose(-3,-2)

    K = run_linear(d_in, d_k, k_proj_weight, in_features).view(*batch, sequence_len, num_heads, d_att_k).transpose(-3, -2)

    V = run_linear(d_in, d_v, v_proj_weight, in_features).view(*batch, sequence_len, num_heads, d_att_v).transpose(-3, -2)

    Q =run_rope(d_att_k, theta,max_seq_len,Q,token_positions)

    K =run_rope(d_att_k,theta, max_seq_len,K,token_positions)

    causal_mask = torch.tril(torch.ones(sequence_len, sequence_len, device=in_features.device)).bool()

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    out = run_scaled_dot_product_attention(Q, K, V, mask=causal_mask)  # [*, num_heads, seq, d_att_v]
    out = out.transpose(-3, -2).contiguous().view(*batch, sequence_len, d_v)

    return run_linear(d_v, d_model, o_proj_weight, out)



def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # inv_freq = 1.0/(theta**(torch.arange(0,d_k,2, device=in_query_or_key.device).float()/d_k))
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=in_query_or_key.device).to(torch.float64)/ d_k))
    angles = token_positions.float().unsqueeze(-1)* inv_freq

    x_1 =in_query_or_key[...,::2]
    x_2 =in_query_or_key[...,1::2]

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    x_rot_even = x_1 * cos - x_2 * sin
    x_rot_odd = x_1 * sin + x_2 * cos

    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)
    x_rot = x_rot.view_as(in_query_or_key)
    x_rot = x_rot.to(in_query_or_key.dtype)

    return x_rot




def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    eps = 1e-6
    x = in_features
    norm_1 = run_rmsnorm(d_model,eps,weights["ln1.weight"],in_features)
    input_length = x.shape[1]
    att = run_multihead_self_attention_with_rope(d_model,num_heads,max_seq_len,theta,
                                                 weights["attn.q_proj.weight"],
                                                 weights["attn.k_proj.weight"],
                                                 weights["attn.v_proj.weight"],
                                                 weights["attn.output_proj.weight"],
                                                 norm_1,torch.arange(0,input_length),)
    x =x+att
    norm_2 = run_rmsnorm(d_model,eps,weights["ln2.weight"],x)
    ff_o = run_swiglu(d_model,d_ff,weights["ffn.w1.weight"],weights["ffn.w2.weight"],weights["ffn.w3.weight"],norm_2)
    x = x + ff_o
    return x



def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # max_len_seq = torch.max(in_indices,dim=-1)
    in_features = run_embedding(vocab_size,d_model,weights["token_embeddings.weight"],in_indices)
    seq_l =in_indices.shape[1]
    for i in range(num_layers):
        block_weights = get_transformer_block_weights(weights, i)
        in_features = run_transformer_block(d_model,num_heads,d_ff,seq_l,rope_theta,
                                        block_weights,in_features
                                        )

    if "ln_final.weight" in weights:
        in_features = run_rmsnorm(d_model, 1e-9, weights["ln_final.weight"], in_features)

    logits = in_features @ weights["lm_head.weight"].T
    return logits


def get_transformer_block_weights(state_dict, layer_num):
    # print([k for k in state_dict.keys() if k.startswith("layers.")])
    prefix = f"layers.{layer_num}."
    keys = [
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.output_proj.weight",
        "ln1.weight",
        "ffn.w1.weight",
        "ffn.w2.weight",
        "ffn.w3.weight",
        "ln2.weight",
    ]
    return {key: state_dict[prefix + key] for key in keys}

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    import math
    norms = in_features.norm(dim=-1, keepdim=True) / math.sqrt(d_model) + eps
    return in_features/norms * weights



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return torch.sigmoid(in_features)*in_features


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    inputs = []
    outputs = []
    max_start = len(dataset) - context_length -1
    for i in range(batch_size):
        start = np.random.randint(0,max_start+1)

        data_in =dataset[start:context_length+start]
        data_out = dataset[start+1:context_length+start+1]
        inputs.append(torch.tensor(data_in,dtype=torch.long))
        outputs.append(torch.tensor(data_out,dtype=torch.long))
    outputs =torch.stack(outputs).to(device)
    inputs =torch.stack(inputs).to(device)
    return inputs, outputs






def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_x,_= torch.max(in_features,dim=dim,keepdim=True)
    scaled_feature = in_features - max_x
    sum_dim = torch.sum(torch.exp(scaled_feature) ,dim=dim,keepdim=True)
    prob = torch.exp(scaled_feature)/sum_dim
    return prob




def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    batch_size = targets.shape[0]
    max_i,_ = torch.max(inputs,dim=-1,keepdim=True)
    shift = inputs -max_i
    log_sum = torch.log(torch.sum(torch.exp(shift),dim=-1,keepdim=True))

    log_sum =shift -log_sum
    losses = -log_sum[torch.arange(batch_size),targets]

    return losses.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_norm =0.0
    for p in parameters:
        if p.grad is not None:
            param_norm =p.grad.data.norm(2)
            total_norm +=param_norm.item()**2
    total_norm = torch.sqrt(torch.tensor(total_norm))
    if total_norm> max_l2_norm:
        scale =max_l2_norm/total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)





def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return torch.optim.AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    import numpy as np
    # Linear warmup phase
    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    # Cosine decay phase
    elif it <= cosine_cycle_iters:
        t = it -warmup_iters
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
                1 + np.cos(np.pi * t / (cosine_cycle_iters-warmup_iters))
        )
    # Constant minimum learning rate phase
    else:
        lr = min_learning_rate

    return lr


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint ={
        "model_state_dict":model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        "iteration":iteration,
    }
    torch.save(checkpoint,out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint =torch.load(src,map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]

    return iteration



def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    class ManualBPETokenizer:
        def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
            self.vocab = vocab
            self.merges = merges
            self.special_tokens = set(special_tokens or [])
            self.byte_merges = {pair: idx for idx, pair in enumerate(merges, start=256)}
            self.inv_vocab = {v: k for k, v in vocab.items()}

        def encode(self, text: str) -> list[int]:
            # Handle special tokens as whole units
            # Find all special tokens and their positions in the text
            import re
            if self.special_tokens:
                # Sort by length descending to handle overlapping tokens correctly
                sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
                # Build regex pattern to match any special token
                pattern = "(" + "|".join(re.escape(s) for s in sorted_specials) + ")"
                # Split text into segments: [nonspecial, special, nonspecial, ...]
                segments = []
                last_end = 0
                for m in re.finditer(pattern, text):
                    if m.start() > last_end:
                        segments.append(("text", text[last_end:m.start()]))
                    segments.append(("special", m.group(0)))
                    last_end = m.end()
                # Add the remaining part as a single segment (preserves "\n\n" merging)
                if last_end < len(text):
                    segments.append(("text", text[last_end:]))
            else:
                segments = [("text", text)]

            output_ids = []
            for typ, seg in segments:
                if typ == "special":
                    # Encode special token as a whole
                    token_bytes = seg.encode("utf-8")
                    # Find the vocab id for this special token
                    found = False
                    for k, v in self.vocab.items():
                        if v == token_bytes:
                            output_ids.append(k)
                            found = True
                            break
                    if not found:
                        # If not found, treat as normal text (fallback) -- use the same logic as for normal text segments
                        tokens = [bytes([b]) for b in token_bytes]
                        while True:
                            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                            pair_ranks = [(pair, self.byte_merges.get(pair, float("inf"))) for pair in pairs]
                            best_pair, best_rank = min(pair_ranks, key=lambda x: x[1], default=(None, None))
                            if best_pair is None or best_rank == float("inf"):
                                break
                            new_tokens = []
                            i = 0
                            while i < len(tokens):
                                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                                    new_tokens.append(tokens[i] + tokens[i + 1])
                                    i += 2
                                else:
                                    new_tokens.append(tokens[i])
                                    i += 1
                            tokens = new_tokens
                        output_ids.extend([self.inv_vocab[token] for token in tokens if token in self.inv_vocab])
                else:
                    # Normal text: byte-level tokenization and merges
                    if not seg:
                        continue
                    tokens = [bytes([b]) for b in seg.encode("utf-8")]
                    while True:
                        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                        pair_ranks = [(pair, self.byte_merges.get(pair, float("inf"))) for pair in pairs]
                        best_pair, best_rank = min(pair_ranks, key=lambda x: x[1], default=(None, None))
                        if best_pair is None or best_rank == float("inf"):
                            break
                        new_tokens = []
                        i = 0
                        while i < len(tokens):
                            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                                new_tokens.append(tokens[i] + tokens[i + 1])
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1
                        tokens = new_tokens
                    output_ids.extend([self.inv_vocab[token] for token in tokens if token in self.inv_vocab])
            return output_ids

        def decode(self, ids: list[int]) -> str:
            return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

        def encode_iterable(self, file_obj):
            # Match tiktoken's behavior: treat trailing newlines as their own token
            for line in file_obj:
                if line.endswith("\n"):
                    line = line[:-1]
                    yield from self.encode(line)
                    yield self.encode("\n")[0]
                else:
                    yield from self.encode(line)

    return ManualBPETokenizer(vocab, merges, special_tokens)
from collections import Counter, defaultdict
import os
import regex as re
from concurrent.futures import ProcessPoolExecutor


def pretokenize(text):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)
    return [m.group(0) for m in pattern.finditer(text)]

def process_chunk(chunk, special_tokens):
    # Split on special tokens, then pretokenize each part
    pre_token_counts = Counter()
    pattern = re.compile("|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)))
    chunks = pattern.split(chunk)
    # specials = special_pattern.findall(chunk)
    for i, chunk_text in enumerate(chunks):
        for tok in pretokenize(chunk_text):
            pre_token_counts[tok.encode('utf-8')] += 1

    return pre_token_counts
def process_chunk_args(args):
    chunk, special_tokens = args
    return process_chunk(chunk, special_tokens)
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer matching GPT-2's behavior.
    Returns:
        vocab: Mapping from token ID to bytes.
        merges: List of merge pairs in order of creation.
    """
    # Initialize vocabulary with single bytes
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    token_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
    special_token_bytes = [s.encode("utf-8") for s in special_tokens]
    merges: list[tuple[bytes, bytes]] = []
    current_id = 256

    num_processes = 1 if os.path.getsize(input_path) < 10_000_000 else 4

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    # Prepare the argument list
    args_list = [(chunk, special_tokens) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        counters = list(executor.map(process_chunk_args, args_list))

    from functools import reduce
    from operator import add
    pre_token_counts = reduce(add, counters, Counter())

    # Build corpus: list of (tuple of token ids, freq)
    # **Ensures special tokens are always atomic**
    for token in special_token_bytes:
        if token not in vocab.values():
            vocab[current_id] = token
            token_to_id[token] = current_id
            current_id += 1
    token_to_id = {v: k for k, v in vocab.items()}  # Rebuild for safety

    corpus = []
    for pretoken_bytes, freq in pre_token_counts.items():

        tok_ids = tuple(token_to_id[bytes([b])] for b in pretoken_bytes)
        corpus.append((tok_ids, freq))

    # special_token_ids = set(token_to_id[tok] for tok in special_token_bytes)

    # BPE merge loop
    while current_id < vocab_size:
        # 1. Count all pairs
        pair_freq = Counter()
        for token_seq, freq in corpus:
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair_freq[pair] += freq

        if not pair_freq:
            break

        # 2. Pick the most frequent pair (min() for tie-break)
        max_count = max(pair_freq.values())
        candidates = [pair for pair, count in pair_freq.items() if count == max_count]
        # print("cadidate",vocab[candidates[0][0]])
        best_pair = max(
            candidates,
            key=lambda pair: (vocab[pair[0]], vocab[pair[1]]) # I took nearly  2weeks for here,# IMPORTANT: When there are multiple pairs tied for the highest frequency,
                # GPT-2's BPE merges the lexicographically *greatest* pair.
                # This means we compare the actual bytes for the two tokens in each pair,
                # first by the left token, then by the right token if needed.
                # I spent nearly two weeks tracking down subtle errors here—
                # it’s *crucial* to use (vocab[pair[0]], vocab[pair[1]]) for lexicographic order,
                # not their sum or just the IDs.
                # Changing this fixed all my reference mismatches!
            # (It’s a pain point: this tiny detail tripped me up for almost two weeks!)
        )

            # print(f"Chosen pair: {ord(vocab[best_pair[0]])} {ord(vocab[best_pair[1]])}")
        # 3. Add new token to vocab
        merged_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[current_id] = merged_bytes
        token_to_id[merged_bytes] = current_id
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 4. Rebuild the corpus: replace every occurrence of best_pair with new token
        new_corpus = []
        for token_seq, freq in corpus:
            new_seq = []
            i = 0
            while i < len(token_seq):
                # Check for the pair at i, i+1
                if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == best_pair:
                    new_seq.append(current_id)
                    i += 2  # skip next, because it's merged
                else:
                    new_seq.append(token_seq[i])
                    i += 1
            new_corpus.append((tuple(new_seq), freq))
        corpus = new_corpus

        current_id += 1



    return vocab, merges


import os
from typing import BinaryIO


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


