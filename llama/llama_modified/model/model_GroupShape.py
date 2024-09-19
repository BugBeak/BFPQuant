# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import os

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
# torch.set_printoptions(threshold=5000) 

def print_to_file(*args, file_name='default_log.txt'):
    with open(file_name, 'a') as f:
        print(*args, file=f)
def print_tensor_to_file(tensor, file_name='default_log.txt'):
    with open(file_name, 'a') as f:
        f.write(f"{tensor.cpu().numpy().tolist()}\n")
def append_and_save(new_data, file_path):
    # Load existing data
    try:
        existing_data = torch.load(file_path)
        if existing_data.ndimension() == 1:
            existing_data = existing_data.unsqueeze(0)
        updated_data = torch.cat((existing_data, new_data.unsqueeze(0)), dim=0)
    except FileNotFoundError:
        updated_data = new_data.unsqueeze(0)
    torch.save(updated_data, file_path)

#Block Floating Point Functions
def value_range(e, m, bias):
  #these are min and max abs values
  max_exp = 2**(e) - 1 - bias
  maxMantissa = 1 + (1 - 2**(-m-1))
  maxValue = 2**max_exp * maxMantissa
  return maxValue, -maxValue

def shiftTruncateInBinaryForm(shifted_mantissa, m, shift):
  factor = 2**(m-1)
  shifted_mantissa_multiple = (shifted_mantissa * factor).to(torch.int32)
  shift = shift.to(torch.int32)
  shifted = shifted_mantissa_multiple >> shift #both numbers need to be integers to use bitwise operation
  bitlength = (shifted.abs().log2().floor() + 1).to(torch.int32)
  mask = torch.where(bitlength > 8, 0xFF << (bitlength - 8), torch.tensor(0xFF, dtype=torch.int32)) #for truncating
  return (shifted & mask)/factor

def BlockQuantize(input, m, e, bias):
  sign = input.sign()

  #scale and clip: clamping values that cannot be represented in the given BFP config
  max_value, min_value = value_range(e, m, bias)
  # print('max_value = ', max_value, ', min_value = ', min_value)
  input = torch.clamp(input.abs(), min_value, max_value) #taking absolute, sign to be added later

  #calculating integer exponent by using floor function; dividing by found exponent to get the mantissa
  exponent = torch.floor(torch.log2(input+1e-40))
  # print('exponent: ', exponent)
  mantissa = input/(2**exponent)
  # print('mantissa: ', mantissa)

  #computing group exponent
  groupExp = torch.max(exponent)
  if (groupExp+bias<1):
    groupExp = -bias
  # print('groupExp: ', groupExp)

  #shifting and truncating mantissa
  shifted_mantissa = torch.tensor(shiftTruncateInBinaryForm(mantissa, m, groupExp - exponent))
  return sign * shifted_mantissa * 2**groupExp

def BlockQuantize2D(data, g, calculate_bias, bias = 0, m = 8, e = 8):
  tensor = data.clone()
  [a, b] = tensor.shape
  [c, d] = g
  tensor = tensor.reshape(-1, d)
  new_width = int(b/d)
  for start in range(new_width):
    selected_rows = tensor[start::new_width] #selecting subblock
    flattened_rows = selected_rows.flatten() #subblock flattened into vector
    if(calculate_bias):#compute bias
      maxvalue = torch.max(torch.abs(flattened_rows))
      print('max value: '+str(maxvalue))
      bias = (2**e)-math.log2(maxvalue)+math.log2(2-2**(-m+1))-1
    print('bias: '+str(bias))
    quantized_flattened_rows = BlockQuantize(flattened_rows, m, e, bias)
    altered_rows = quantized_flattened_rows.reshape(selected_rows.shape)
    tensor[start::new_width] = altered_rows
  return tensor.reshape(a, b)


group_shapes = [[1, 4096], [64, 64], [4096, 1]]


#
def update_line_by_adding(file_path, line_number, new_list):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    if line_number > len(lines) or line_number < 1:
        print(f"Line number {line_number} is out of range.")
        # If the line number is out of range, add the line at the end
        lines.append(','.join(map(str, new_list)) + '\n')
        with open(file_path, 'w') as file:
            file.writelines(lines)
        print(f"Line {line_number} does not exist. Added new line with '{new_list}'.")
        return
    line = lines[line_number - 1].strip()
    if line=='':
        updated_line = ','.join(map(str, new_list))
    else:
        number_list = list(map(int, line.split(',')))
        updated_list = [num + add for num, add in zip(number_list, new_list)]
        updated_line = ','.join(map(str, updated_list))
    lines[line_number - 1] = updated_line + '\n'
    with open(file_path, 'w') as file:
        file.writelines(lines)

    print(f"Line {line_number} has been updated to '{updated_line}'.")





@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.layer_id = args.layer_id
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # New
        print('ATTENTION_WQ_WEIGHT: ',self.wq.get_master_weight().shape)
        
        wq_shape = ColumnParallelLinear(
            in_features=self.wq.in_features,
            out_features=self.wq.out_features,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        wkv_shape = ColumnParallelLinear(
            in_features=self.wk.in_features,
            out_features=self.wk.out_features,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        wo_shape = RowParallelLinear(
            in_features=self.wo.in_features,
            out_features=self.wo.out_features,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        layer_filename = 'groupshape/group_shape.txt'
        errors = []
        for shape in group_shapes:
            quantized_weights = BlockQuantize2D(self.wq.get_master_weight(), shape, 1)
            wq_shape.weight.copy_(quantized_weights)
            wq_shapeout = wq_shape(x)
            errors.append(torch.sum((wq_shapeout - xq) ** 2).item())
        update_line_by_adding(layer_filename, 1+(self.layer_id)*8+1, errors)
        errors = []
        for shape in group_shapes:
            quantized_weights = BlockQuantize2D(self.wk.get_master_weight(), shape, 1)
            wkv_shape.weight.copy_(quantized_weights)
            wk_shapeout = wkv_shape(x)
            errors.append(torch.sum((wk_shapeout - xk) ** 2).item())
        update_line_by_adding(layer_filename, 1+(self.layer_id)*8+2, errors)
        errors = []
        for shape in group_shapes:
            quantized_weights = BlockQuantize2D(self.wv.get_master_weight(), shape, 1)
            wkv_shape.weight.copy_(quantized_weights)
            wv_shapeout = wkv_shape(x)
            errors.append(torch.sum((wv_shapeout - xv) ** 2).item())
        update_line_by_adding(layer_filename, 1+(self.layer_id)*8+3, errors)

        
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        #New
        # if(self.layer_id==10):
        #     print('ATTENTION_Q: ',xq.shape)
        #     print('ATTENTION_Q: ',xq.squeeze()[0].shape)
        # if(self.layer_id%10==0):
        #     layer_filename = 'activations/'+str(self.layer_id)+'_attention_'
        #     append_and_save(xq.squeeze()[0],layer_filename+'Q.pt')
        #     append_and_save(xk.squeeze()[0],layer_filename+'K.pt')
        #     append_and_save(xv.squeeze()[0],layer_filename+'V.pt')

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        errors = []
        for shape in group_shapes:
            quantized_weights = BlockQuantize2D(self.wo.get_master_weight(), shape, 1)
            wo_shape.weight.copy_(quantized_weights)
            wo_shapeout = wo_shape(x)
            errors.append(torch.sum((wo_shapeout - output) ** 2).item())
        update_line_by_adding(layer_filename, 1+(self.layer_id)*8+4, errors)

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        layer_id: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.layer_id = layer_id

    def forward(self, x):        
        w1out = self.w1(x)
        w3out = self.w3(x)
        tempout = F.silu(w1out) * w3out
        output = self.w2(tempout)

        w13_shape = ColumnParallelLinear(
            self.w1.in_features, self.w1.out_features, bias=False, gather_output=False, init_method=lambda x: x
        )
        w2_shape = RowParallelLinear(
            self.w2.in_features, self.w2.out_features, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        layer_filename = 'groupshape/group_shape.txt'
        errors = []
        for shape in group_shapes:
            quantized_weights = BlockQuantize2D(self.w1.get_master_weight(), shape, 1)
            w13_shape.weight.copy_(quantized_weights)
            w1_shapeout = w13_shape(x)
            errors.append(torch.sum((w1_shapeout - w1out) ** 2).item())
        update_line_by_adding(layer_filename, 1+(self.layer_id)*8+5, errors)
        errors = []
        for shape in group_shapes:
            quantized_weights = BlockQuantize2D(self.w3.get_master_weight(), shape, 1)
            w13_shape.weight.copy_(quantized_weights)
            w3_shapeout = w13_shape(x)
            errors.append(torch.sum((w3_shapeout - w3out) ** 2).item())
        update_line_by_adding(layer_filename, 1+(self.layer_id)*8+7, errors)
        errors = []
        for shape in group_shapes:
            quantized_weights = BlockQuantize2D(self.w2.get_master_weight(), shape, 1)
            w2_shape.weight.copy_(quantized_weights)
            w2_shapeout = w2_shape(tempout)
            errors.append(torch.sum((w2_shapeout - output) ** 2).item())
        update_line_by_adding(layer_filename, 1+(self.layer_id)*8+6, errors)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        args.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            layer_id=layer_id,
            ffn_dim_multiplier=args.ffn_dim_multiplier
            
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        countlayer = 0;
        for layer in self.layers:
            print("Layer number ",countlayer,"\n")
            # layer_file_name = 'weights/generated_texts'+str(countlayer)+'.txt'
            # print_to_file('Layer Number ',countlayer,'/n',file_name=layer_file_name)
            h = layer(h, start_pos, freqs_cis, mask)
            # print_to_file('Layer ',countlayer,' activation: \n',file_name='activations_text.txt')
            # print_tensor_to_file(h,file_name='activations_text.txt')
            # print_to_file('\n',file_name='activations_text.txt')
            countlayer+= 1;
            # print(layer)
            # exit()
        h = self.norm(h)
        output = self.output(h).float()
        return output
