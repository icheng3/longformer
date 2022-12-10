
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model."""

import math
from typing import List, Optional, Tuple, Union
from longformer.longformer import LongformerSelfAttention
import tensorflow as tf

import torch.utils.checkpoint
from transformers.activations_tf import get_tf_activation

from transformers.modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFMaskedLMOutput,
    get_initializer,
    keras_serializable,
    unpack_inputs,
    hf_compute_loss
)

from transformers.tf_utils import shape_list, stable_softmax
from transformers.utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from longformer.roberta_config import RobertaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


class RobertaEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(config.initializer_range),
            )
        self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[config.max_position_embeddings, config.hidden_size],
                initializer=get_initializer(config.initializer_range),
            )
        self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[config.type_vocab_size, config.hidden_size],
                initializer=get_initializer(config.initializer_range),
            )
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="Layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # End copy
        self.padding_idx = config.pad_token_id
    
    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask
        return incremental_indices + self.padding_idx
    def create_position_ids_from_input_embeds(self, input_ids, past_key_values_length=0):
        input_shape = input_ids.get_shape().as_list()
        position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1, dtype=tf.int64),
                    axis=0,
                )
        return position_ids

    def call(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, training=False
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = tf.gather(params=self.position_embeddings, indices=position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings



# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class RobertaSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, training:bool=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class RobertaAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.self = LongformerSelfAttention(config, layer_id, name='roberta_self_att')
        self.output = RobertaSelfOutput(config, name='roberta_output')

    def call(self, inputs, training=False):
        (
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions
        ) = inputs

        self_outputs = self.self(
            [hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions], training=training
        )
        attention_output = self.output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    

# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class RobertaIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range,
        name='dense_layer'))
        self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range,
        name='dense_layer'))
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training: bool=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class RobertaLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config, layer_id, name='attention')
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = RobertaIntermediate(config, name='intermed')
        self.output = RobertaOutput(config, name='output')

    def call(self, inputs, training=False):
        
        (
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        ) = inputs
       
        self_attention_outputs = self.attention(
            [hidden_states,
            attention_mask,
            head_mask,
            output_attentions], training=True
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(
            intermediate_output, attention_output,training=training 
        )
        outputs = (layer_output,) + attention_output[1:]

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [RobertaLayer(config, i, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        self.gradient_checkpointing = False
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_values,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
        padding_len=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                hidden_states_to_add = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
                all_hidden_states = all_hidden_states + (hidden_states_to_add,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                #all_global_attentions = all_global_attentions + (tf.transpose(layer_outputs[2], (0, 1, 3, 2)),)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        
        if output_hidden_states:
            hidden_states_to_add = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
            all_hidden_states = all_hidden_states + (hidden_states_to_add,)
        hidden_states = hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states
        if output_attentions:
            all_attentions = (
                tuple([state[:, :, :-padding_len, :] for state in all_attentions])
                if padding_len > 0
                else all_attentions
            )
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class RobertaPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="pooler_dense",
        )
        self.activation = tf.keras.activations.tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output                                                                                                         


class RobertaPreTrainedModel(TFPreTrainedModel): #COME BACK TO THIS
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, tf.keras.layers.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, std=self.config.initializer_range)
            module.weight.data = initializer(tf.shape(module.weight.data))
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # module.bias.data.zero_()
                tf.zeros(tf.shape(module.bias.data), dtype=tf.dtypes.float32)
        elif isinstance(module, tf.keras.layers.Embedding):
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, std=self.config.initializer_range)
            module.weight.data = initializer(tf.shape(module.weight.data))
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0
        elif isinstance(module, tf.keras.layers.LayerNorm):
            tf.ones(tf.shape(module.bias.data), dtype=tf.dtypes.float32)
            #module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RobertaEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


ROBERTA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class RobertaModel(TFPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """
    config_class = RobertaConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        config.attention_window = [config.attention_window] * config.num_hidden_layers 
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.pad_token_id = config.pad_token_id
        self.attention_window = config.attention_window

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def call(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_values,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=False,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # batch_size, seq_length = input_shape
        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = tf.cast(tf.fill(input_shape, 1), tf.int64)

        if token_type_ids is None:
            tf.cast(tf.fill(input_shape, 0), tf.int64)

        # # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # # If a 2D or 3D attention mask is provided for the cross-attention
        # # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        attention_mask_shape = shape_list(attention_mask)
        extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], attention_mask_shape[1], 1, 1))
        encoder_extended_attention_mask = tf.cast(tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            training=training
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
         
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions, #may need to set to none
        )


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(TFPreTrainedModel, TFMaskedLanguageModelingLoss): # TFLongformerPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False, name='roberta_model')
        self.lm_head = RobertaLMHead(config, self.roberta.embeddings, name='lm_head')

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    def call(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        labels,
        output_attentions,
        output_hidden_states,
        return_dict,
        training: Optional[bool] = False,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training= training,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, training=training)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.hf_compute_loss(labels, prediction_scores)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return TFMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            #global_attention=outputs.global_attentions,
        )

class RobertaLMHead(tf.keras.layers.Layer):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")

        self.decoder = input_embeddings
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")

    def call(self, hidden_states):
        hidden = self.dense(hidden_states)
        hidden = tf.keras.activations.gelu(hidden)
        hidden = self.layer_norm(hidden)

        seq_length = shape_list(tensor=hidden)[1]
        hidden = tf.reshape(tensor=hidden, shape=[-1, self.hidden_size])
        hidden = tf.matmul(a=hidden, b=self.decoder.weight, transpose_b=True)
        hidden = tf.reshape(tensor=hidden, shape=[-1, seq_length, self.vocab_size])
        hidden = tf.nn.bias_add(value=hidden, bias=self.bias)

        return hidden

