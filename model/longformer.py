from typing import List
import math
import tensorflow as tf
#from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
# from longformer.sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv
from modeling_roberta_tf import RobertaConfig, RobertaModel, RobertaForMaskedLM, LongformerSelfAttention
import numpy as np
class Longformer(RobertaModel):
    def __init__(self, config, **kwargs):
        super(Longformer, self).__init__(config, **kwargs)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config, layer_id, **kwargs):
        super(LongformerForMaskedLM, self).__init__(config, **kwargs)
        for i, layer in enumerate(self.roberta.encoder.layer):
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerConfig(RobertaConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks', **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = 1
        self.autoregressive = False
        self.attention_mode = 'sliding_chunks'
        

# class LongformerSelfAttention(tf.keras.layers.Layer): #used to be nn.module
#     def __init__(self, config, layer_id, **kwargs):
#         super(LongformerSelfAttention, self).__init__(**kwargs)
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (config.hidden_size, config.num_attention_heads))
#         self.num_heads = config.num_attention_heads
#         self.head_dim = int(config.hidden_size / config.num_attention_heads)
#         self.embed_dim = config.hidden_size
#         self.query = tf.keras.layers.Layer.Dense(self.embed_dim, activation='relu')
#         self.key = tf.keras.Layers.Dense(self.embed_dim, activation='relu')
#         self.value = tf.keras.Layers.Dense(self.embed_dim, activation='relu')
#         self.query_global = tf.keras.Layers.Dense(self.embed_dim, activation='relu')
#         self.key_global = tf.keras.Layers.Dense(self.embed_dim, activation='relu')
#         self.value_global = tf.keras.Layers.Dense(self.embed_dim, activation='relu')
#         self.dropout = config.attention_probs_dropout_prob #this is a rate
#         self.layer_id = layer_id
#         self.attention_window = config.attention_window[self.layer_id] #this is a value
#         self.attention_dilation = config.attention_dilation[self.layer_id] #this is a value
#         self.attention_mode = config.attention_mode #this is a string
#         self.autoregressive = config.autoregressive #this is a boolean
#         #self.one_sided_attn_window = config.attention_window[self.layer_id] // 2
#         assert self.attention_window > 0
#         assert self.attention_dilation > 0
#         assert self.attention_mode in ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
#         if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
#             assert not self.autoregressive  # not supported
#             assert self.attention_dilation == 1  # dilation is not supported

#     def call(self, inputs, training=False,):
#         (
#             hidden_states,
#             attention_mask,
#             head_mask,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             output_attentions,
#         ) = inputs
#         '''
#         The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
#             -ve: no attention
#               0: local attention
#             +ve: global attention
#         '''
#         assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
#         assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"
#         assert head_mask is None, "`head_mask` is not supported and should be None"

#         if attention_mask is not None:
#             #attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
#             attention_mask = tf.squeeze(tf.squeeze(attention_mask, axis=2), axis=1)
#             key_padding_mask = attention_mask < 0
#             extra_attention_mask = attention_mask > 0
#             remove_from_windowed_attention_mask = attention_mask != 0

#             #num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
#             num_extra_indices_per_batch = tf.math.reduce_sum(tf.convert_to_tensor(extra_attention_mask, dtype=tf.int64)
#     , axis=1, name='num_extra_indices_per_batch')
#             #max_num_extra_indices_per_batch = num_extra_indices_per_batch.max() #what is this doing???
#             extra_attention_mask = None
#             # if max_num_extra_indices_per_batch <= 0:
#             #     extra_attention_mask = None #this is the global attention
#             # else:
#             #     # To support the case of variable number of global attention in the rows of a batch,
#             #     # we use the following three selection masks to select global attention embeddings
#             #     # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
#             #     # 1) selecting embeddings that correspond to global attention
#             #     #extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
#             #     extra_attention_mask_nonzeros=tf.experimental.numpy.nonzero(extra_attention_mask)
#             #     # zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
#             #     #                                  device=num_extra_indices_per_batch.device)
#             #     zero_to_max_range = tf.range(0, max_num_extra_indices_per_batch ,dtype=tf.int32, name='zero_to_max_range')
#             #     # mask indicating which values are actually going to be padding
#             #     # num_extra_indices_per_batch.unsqueeze(dim=-1)
#             #     selection_padding_mask = zero_to_max_range < tf.expand_dims(num_extra_indices_per_batch, axis = -1)
#             #     # 2) location of the non-padding values in the selected global attention
#             #     #selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
#             #     selection_padding_mask_nonzeros = tf.experimental.numpy.nonzero(selection_padding_mask)
#             #     # 3) location of the padding values in the selected global attention
#             #     # selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
#             #     selection_padding_mask_zeros = tf.experimental.numpy.nonzero((selection_padding_mask == 0))

#         else:
#             remove_from_windowed_attention_mask = None
#             extra_attention_mask = None
#             key_padding_mask = None

#         #hidden_states = hidden_states.transpose(0, 1)
#         hidden_states = tf.transpose(hidden_states)
#         seq_len, bsz, embed_dim = tf.TensorShape(hidden_states).as_list()
#         assert embed_dim == self.embed_dim
#         q = self.query(hidden_states)
#         k = self.key(hidden_states)
#         v = self.value(hidden_states)
#         q /= math.sqrt(tf.cast(self.head_dim, dtype=q.dtype))

#         q = tf.reshape(q, (bsz, seq_len, self.num_heads, self.head_dim))
#         k = tf.reshape(k, (bsz, seq_len, self.num_heads, self.head_dim))
#         # attn_weights = (bsz, seq_len, num_heads, window*2+1)
#         attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
#         #mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
#         if remove_from_windowed_attention_mask is not None:
#             # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
#             # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
#             # remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
#             remove_from_windowed_attention_mask = tf.expand_dims(tf.expand_dims(remove_from_windowed_attention_mask, axis=-1), axis=-1)
#             #remove_from_windowed_attention_mask = tf.cast(remove_from_windowed_attention_mask, dtype=q.dtype) * LARGE_NEGATIVE
#             # cast to float/half then replace 1's with -inf
#             #float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
#             float_mask = tf.where(remove_from_windowed_attention_mask,-10000.0, tf.cast(remove_from_windowed_attention_mask, dtype=q.dtype))
#             repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
#             # float_mask = float_mask.repeat(1, 1, repeat_size, 1)
#             float_mask = tf.repeat(float_mask, 1, repeat_size, 1)
#             ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
#             # diagonal mask with zeros everywhere and -inf inplace of padding
#             d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

#             attn_weights += d_mask
#             attn_probs = tf.keras.activations.softmax(attn_weights, axis=-1)
#         assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
#         assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]
#         attn_probs = self.dropout(attn_probs, training=training)


#         # # the extra attention --> FOR GLOBALL ATTENTION, NOT IMPLEMENTING
#         # if extra_attention_mask is not None:
#         #     selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
#         #     selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
#         #     # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
#         #     # selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
#         #     selected_attn_weights = tf.einsum('blhd,bshd->blhs', (q, selected_k))
#         #     selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
#         #     # concat to attn_weights
#         #     # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
#         #     # attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)
#         #     attn_weights = tf.concat((selected_attn_weights, attn_weights), axis=-1)
#         # # attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
#         # attn_weights_float = tf.nn.softmax(attn_weights, dim=-1, name='attn_weights_float')  # use fp32 for numerical stability
#         # if key_padding_mask is not None:
#         #     # softmax sometimes inserts NaN if all positions are masked, replace them with 0
#         #     #attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
#         #     attn_weights_float = tf.where(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
#         # attn_weights = attn_weights_float.type_as(attn_weights)
#         # attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
#         # v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
#         # attn = 0
#         # if extra_attention_mask is not None:
#         #     selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
#         #     selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
#         #     selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
#         #     # use `matmul` because `einsum` crashes sometimes with fp16
#         #     # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
#         #     #attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
#         #     attn = tf.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
#         #     attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

#         # if self.attention_mode == 'tvm':
#         #     v = v.float().contiguous()
#         #     attn = diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
#         if self.attention_mode == "sliding_chunks":#this is the only case we will ever enter i think
#             attn = sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
#         # elif self.attention_mode == "sliding_chunks_no_overlap": 
#         #     attn = sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
#         else:
#             raise False

#         attn = attn.type_as(hidden_states)
#         assert list(tf.TensorShape(attn)) == [bsz, seq_len, self.num_heads, self.head_dim]
#         attn = tf.reshape(attn, (bsz, seq_len, embed_dim))
        

#         # # For this case, we'll just recompute the attention for these indices
#         # # and overwrite the attn tensor. TODO: remove the redundant computation
#         # if extra_attention_mask is not None:
#         #     selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
#         #     selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

#         #     q = self.query_global(selected_hidden_states)
#         #     k = self.key_global(hidden_states)
#         #     v = self.value_global(hidden_states)
#         #     q /= math.sqrt(self.head_dim)

#         #     q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
#         #     k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
#         #     v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
#         #     attn_weights = torch.bmm(q, k.transpose(1, 2))
            
#         #     assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

#         #     attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
#         #     attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
#         #     if key_padding_mask is not None:
#         #         attn_weights = attn_weights.masked_fill(
#         #             key_padding_mask.unsqueeze(1).unsqueeze(2),
#         #             -10000.0,
#         #         )
#         #     attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
#         #     attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
#         #     attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
#         #     selected_attn = torch.bmm(attn_probs, v)
#         #     assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

#         #     selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
#         #     nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
#         #     attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

#         #context_layer = attn.transpose(0, 1)
#         # if output_attentions:
#         #     if extra_attention_mask is not None:
#         #         # With global attention, return global attention probabilities only
#         #         # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
#         #         # which is the attention weights from tokens with global attention to all tokens
#         #         # It doesn't not return local attention
#         #         # In case of variable number of global attantion in the rows of a batch,
#         #         # attn_weights are padded with -10000.0 attention scores
#         #         attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
#         #     else:
#         #         # without global attention, return local attention probabilities
#         #         # batch_size x num_heads x sequence_length x window_size
#         #         # which is the attention weights of every token attending to its neighbours
#         #         attn_weights = attn_weights.permute(0, 2, 1, 3)
#         outputs = (attn, attn_probs) 
#         return outputs
