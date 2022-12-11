import torch
import torch.nn.functional as F
import tensorflow as tf
from diagonaled_mm_tvm import mask_invalid_locations

def _skew(x, padding_value):
    '''Convert diagonals into columns'''
    pads = tf.convert_to_tensor([0,0],[0,0],[0,1],[0,0])
    x_padded = tf.pad(x, pads, constant_values=padding_value)
    
    x_padded = tf.reshape(x_padded, (tf.shape_list(x_padded)[0], tf.shape_list(x_padded)[1], tf.shape_list(x_padded)[-1], tf.shape_list(x_padded)[-2]))
    return x_padded


def _skew2(x, padding_value):
    '''shift every row 1 step to right converting columns into diagonals'''
    # X = B x C x M x L
    B, C, M, L = tf.shape_list(x)
    pads = tf.convert_to_tensor([0,0], [0,0],[0,0],[0, M+1])
    x = tf.pad(x, pads, constant_values=padding_value)
    x = tf.reshape(x, (B, C, -1))
    x = x[:,:,:-M]
    x = tf.reshape(x, (B, C, M, M + L))
    x = x[:, :, :, :-1]
    return x


def _chunk(x, w):
    '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

    # non-overlapping chunks of size = 2w
    '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

    # non-overlapping chunks of size = 2w
    x = torch.from_numpy(x.numpy())
    x = torch.from_numpy(w.numpy())
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    output = x.as_strided(size=chunk_size, stride=chunk_stride)
    return tf.convert_to_tensor(output.numpy())


def sliding_chunks_matmul_qk(q, k, w, padding_value):
    '''Matrix multiplication of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    
    bsz, seqlen, num_heads, head_dim = tf.shape_list(q)
    assert seqlen % (w * 2) == 0
    assert tf.shape_list(q) == tf.shape_list(k)

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = tf.transpose(q, (0,2,1,3))
    q = tf.reshape(q, (bsz * num_heads, seqlen, head_dim))
    k = tf.transpose(k, (0,2,1,3))
    k = tf.reshape(k, (bsz * num_heads, seqlen, head_dim))

    chunk_q = _chunk(q, w)
    chunk_k = _chunk(k, w)

    chunk_attn = tf.einsum('bcxd,bcyd->bcxy', chunk_q, chunk_k)

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(chunk_attn, padding_value=padding_value)

    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)
    diagonal_attn = torch.from_numpy(diagonal_attn.numpy())
    w = torch.from_numpy(w.numpy())
    mask_invalid_locations(diagonal_attn, w, 1, False)
    return diagonal_attn


def sliding_chunks_matmul_pv(probs, v, w):
    '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
    format from sliding_chunks_matmul_qk'''
    bsz, seqlen, num_heads, head_dim = tf.shape_list(v)
    assert seqlen % (w * 2) == 0
    assert tf.shape_list(probs)[:3] == tf.shape_list(v)[:3]
    assert tf.shape_list(probs)[3] == 2 * w + 1
    chunks_count = seqlen // w - 1
    chunk_prob = tf.transpose(probs, (0,2,1,3))
    chunk_prob = tf.reshape(chunk_prob, (bsz * num_heads, seqlen // w, w, 2 * w + 1))

    v = tf.transpose(v, (0,2,1,3))
    v = tf.reshape(v, (bsz * num_heads, seqlen, head_dim))

    pads = tf.convert_to_tensor([0,0], [w,w], [0,0])
    padded_v = tf.pad(v, pads, constant_values=-1)

    # chunk padded_v into chunks of size 3w and an overlap of size w
    # chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
    # chunk_v_stride = padded_v.stride()
    # chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
    # chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

    # skewed_prob = _skew2(chunk_prob, padding_value=0)

    context = tf.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
    context = tf.reshape(context, (bsz, num_heads, seqlen, head_dim))
    return tf.transpose(context, (0,2,1,3))


def pad_to_window_size(input_ids, attention_mask,
                       one_sided_window_size, pad_token_id):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = int(2 * one_sided_window_size)
    seqlen = tf.shape_list(input_ids)[:2]
    padding_len = (w - seqlen % w) % w
    pads = tf.convert_to_tensor([[0, 0], [0, padding_len]])
    input_ids = tf.pad(input_ids, pads, constant_values=pad_token_id)
    attention_mask = tf.pad(attention_mask, pads, constant_values=False)  # no attention on the padding tokens
    return input_ids, attention_mask

