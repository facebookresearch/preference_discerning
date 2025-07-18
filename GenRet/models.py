"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import cast, List, Optional, Tuple, Union

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.init import kaiming_uniform_
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModel,
    T5Config,
    T5EncoderModel,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import ModelOutput
from utils import set_module_params_not_trainable

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@dataclass
class Seq2SeqModelOutputWithPosNegLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    pos_loss: Optional[torch.FloatTensor] = None
    neg_loss: Optional[torch.FloatTensor] = None


class PartiallyFixedEmbedding(torch.nn.Module):
    def __init__(self, fixed_weights, num_to_learn):
        super().__init__()
        self.num_fixed = fixed_weights.size(0)
        self.num_to_learn = num_to_learn
        weight = torch.empty(self.num_fixed + num_to_learn, fixed_weights.size(1))
        weight[: self.num_fixed] = fixed_weights
        self.trainable_weight = nn.Parameter(
            torch.empty(num_to_learn, fixed_weights.size(1))
        )
        kaiming_uniform_(self.trainable_weight)
        weight[self.num_fixed :] = self.trainable_weight
        self.register_buffer("weight", weight)

    def forward(self, inp):
        self.weight.detach_()
        self.weight[self.num_fixed :] = self.trainable_weight
        return nn.functional.embedding(inp, self.weight, None, None, 2.0, False, False)


class LinearEncDecModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def register_new_params(
        self,
        tokenizer,
        path_to_embs,
        enc_hidden_sizes,
        dec_hidden_sizes=None,
        tie_weights=False,
    ):
        if not tie_weights:
            # We do not tie weights, i.e. go from RQ-VAE embeddings to LM emb space
            self.codebook_embs = torch.load(path_to_embs, map_location=self.device)
            # append fourth semantic id dimension
            # initialize remaining semantic id with positional embedding
            fourth_codes = sinusoidal_positional_embedding(
                self.codebook_embs[0].shape[0] + 2, self.codebook_embs[0].shape[1]
            )
            self.codebook_embs.append(fourth_codes)
            self.codebook_embs = nn.Embedding.from_pretrained(
                torch.cat([c for c in self.codebook_embs])
            )
            # define linear encoding layer
            input_shape = self.codebook_embs.embedding_dim
            enc_hidden_sizes = [input_shape] + enc_hidden_sizes
            self.semid_encoder = self._get_mlp(enc_hidden_sizes)
            self.emb_dim = enc_hidden_sizes[-1]
            self.use_decoder = dec_hidden_sizes is not None
            if self.use_decoder:
                self.semid_decoder = self._get_mlp(dec_hidden_sizes)
        else:
            # randomly initialize embeddings and share them
            self.codebook_embs = nn.Embedding(
                num_embeddings=1026, embedding_dim=enc_hidden_sizes[-1]
            )
        self.semid_bos = tokenizer.vocab_size + 1024
        self.semid_eos = tokenizer.vocab_size + 1025
        self.vocab_size = tokenizer.vocab_size
        self.tie_weights = tie_weights

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # check input_ids and extract the positions of the semantic ids
        new_ids = []
        new_labels = []
        for in_ids, label in zip(input_ids, labels):
            # can be the case that labels do not contain semid_bos => check for that as well
            new_ids.append(self._get_inputs_embeds(in_ids))
            new_labels.append(self._get_inputs_embeds(label))
        inputs_embeds = torch.stack(new_ids)
        decoder_inputs_embeds = torch.stack(new_labels)
        input_ids = None

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if self.use_decoder:
            lm_logits = self.semid_decoder(sequence_output)
            # shift labels back to their original range, since we do not use the lm_head
            label_mask = labels >= self.vocab_size
            labels[label_mask] = labels[label_mask] - self.vocab_size
        elif self.tie_weights:
            # if we share embedding weights, we use the embeddings as lm_head weights
            lm_logits = sequence_output @ self.codebook_embs.weight.T
            label_mask = labels >= self.vocab_size
            labels[label_mask] = labels[label_mask] - self.vocab_size
        else:
            lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _get_mask(self, in_vec):
        return (in_vec >= self.vocab_size).bool()

    def _get_inputs_embeds(self, in_vec):
        mask = self._get_mask(in_vec)
        if not self.tie_weights:
            new_embs = self.semid_encoder(
                self.codebook_embs(in_vec[mask] - self.vocab_size - 1)
            )
        else:
            new_embs = self.codebook_embs(in_vec[mask] - self.vocab_size - 1)
        pretr_embs = self.encoder.embed_tokens(in_vec[~mask]).type(new_embs.dtype)
        new_id_vec = torch.zeros(
            (in_vec.shape[0], self.emb_dim), dtype=new_embs.dtype
        ).to(in_vec.device)
        new_id_vec[~mask] = pretr_embs
        new_id_vec[mask] = new_embs
        return new_id_vec

    def _get_mlp(self, hidden_sizes, dropout=0.0):
        if len(hidden_sizes) == 2:
            return nn.Linear(hidden_sizes[0], hidden_sizes[1])
        mlp = nn.ModuleList()
        for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            mlp.append(
                nn.Sequential(
                    nn.Linear(in_shape, out_shape),
                    nn.LayerNorm(out_shape),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
        return mlp


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_input_name = "input_ids"


class LangEncSemIDDecModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def register_lang_encoder(
        self,
        hf_name,
        cache_dir,
        set_trainable=False,
        lora_config=None,
        semantic_ids_in_encoder=False,
        path_to_embs=None,
        center_and_scale=False,
        embedded_history=False,
        embedding_model=None,
        cache_hidden_states=False,
        neg_weight=None,
    ):
        if (not embedded_history and not cache_hidden_states) or set_trainable:
            if any([x in hf_name for x in ["gtr", "t5", "instructor"]]):
                self.encoder = T5EncoderModel.from_pretrained(
                    hf_name, cache_dir=cache_dir, torch_dtype=torch.float16
                )
                src_dim = self.encoder.shared.embedding_dim
            else:
                self.encoder = AutoModel.from_pretrained(hf_name, cache_dir=cache_dir)
                src_dim = self.encoder.config.d_model
        else:
            self.encoder = DummyEncoder()
            embedding_model = embedding_model if embedded_history else hf_name
            if any([x in hf_name for x in ["gtr", "t5", "instructor"]]):
                src_dim = T5Config.from_pretrained(embedding_model).d_model
            else:
                src_dim = AutoConfig.from_pretrained(embedding_model).d_model
        self.center_and_scale = center_and_scale
        self.neg_weight = neg_weight
        if src_dim != self.config.d_model:
            print(
                f"Dimensions of encoder and decoder mismatch, adding projection to decoder hidden dim"
            )
            self.enc_dec_mapping = nn.Linear(src_dim, self.config.d_model)
        if semantic_ids_in_encoder:
            assert (
                path_to_embs is not None
            ), "A path to pretrained embeddings needs to be provided for using semantic ids in encoder"
            self.codebook_embs = torch.load(path_to_embs, map_location=self.device)
            # initialize fourth semantic id with positional embedding
            fourth_codes = sinusoidal_positional_embedding(
                self.codebook_embs[0].shape[0] + 3, self.codebook_embs[0].shape[1]
            )
            self.codebook_embs.append(fourth_codes)
            self.codebook_embs = nn.Embedding.from_pretrained(
                torch.cat([c for c in self.codebook_embs])
            )
            # define linear encoding layer
            input_shape = self.codebook_embs.embedding_dim
            print("Using linear mapping from codebook embeddings to language encoder")
            enc_hidden_sizes = [input_shape] + [self.encoder.config.d_model]
            self.semid_encoder = self._get_mlp(enc_hidden_sizes)
            self.vocab_size = self.encoder.config.vocab_size
            self.emb_dim = self.encoder.shared.embedding_dim
        if not isinstance(self.encoder, DummyEncoder):
            set_module_params_not_trainable(self.encoder)
        if set_trainable:
            if semantic_ids_in_encoder:
                raise NotImplementedError(
                    "Semantic ids in language encoder not supported yet!"
                )
            else:
                # train encoder model with lora
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_config["lora_r"],
                    lora_alpha=lora_config["lora_alpha"],
                    target_modules=lora_config["target"],
                    lora_dropout=lora_config["lora_dropout"],
                )
                self.encoder = get_peft_model(self.encoder, peft_config).base_model
                for param in filter(
                    lambda p: p.requires_grad, self.encoder.parameters()
                ):
                    param.data = param.data.to(torch.float32)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sentiment: Optional[list] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # check input_ids and extract the positions of the semantic ids
        if decoder_input_ids is None:
            assert (
                labels is not None
            ), "Either labels or decoder_input_ids must be provided"
        if labels is None:
            assert (
                decoder_input_ids is not None
            ), "Either labels or decoder_input_ids must be provided"

        if hasattr(self, "codebook_embs"):
            # check input_ids and extract the positions of the semantic ids
            new_ids = []
            for in_ids in input_ids:
                # can be the case that labels do not contain semid_bos => check for that as well
                new_ids.append(self._get_inputs_embeds(in_ids))
            inputs_embeds = torch.stack(new_ids)
            input_ids = None

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs,
                hidden_states=None,
                attentions=None,
            )

        hidden_states = encoder_outputs[0]
        if hasattr(self, "enc_dec_mapping"):
            if self.center_and_scale:
                hidden_states = hidden_states - hidden_states.mean(dim=0)
                hidden_states = hidden_states / (hidden_states.std(dim=0) + 1e-8)
            hidden_states = self.enc_dec_mapping(hidden_states)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            if sentiment is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            if sentiment is not None:
                # we are in train mode, sentiment is given, decompose loss
                loss = loss.view(-1, labels.shape[-1])
                sentiment = np.array(sentiment)
                neg_inds = np.nonzero(sentiment == "negative")[0]
                pos_inds = np.setdiff1d(np.arange(len(sentiment)), neg_inds)
                if not len(neg_inds):
                    neg_loss = torch.tensor(0.0)
                else:
                    neg_weight = (
                        len(neg_inds) / len(input_ids)
                        if self.neg_weight is None
                        else self.neg_weight
                    )
                    neg_loss = -(neg_weight * loss[neg_inds].mean())
                pos_weight = len(pos_inds) / len(input_ids)
                pos_loss = pos_weight * loss[pos_inds].mean()
                return_neg_loss = -neg_loss.item()
                return_pos_loss = pos_loss.item()
                loss = pos_loss + neg_loss
            else:
                return_pos_loss = 0.0
                return_neg_loss = 0.0
        else:
            return_pos_loss = 0.0
            return_neg_loss = 0.0

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqModelOutputWithPosNegLoss(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=hidden_states,
            encoder_hidden_states=None,
            encoder_attentions=None,
            pos_loss=return_pos_loss,
            neg_loss=return_neg_loss,
        )

    def _get_mask(self, in_vec):
        return (in_vec >= self.vocab_size).bool()

    def _get_inputs_embeds(self, in_vec):
        mask = self._get_mask(in_vec)
        new_embs = self.semid_encoder(
            self.codebook_embs(in_vec[mask] - self.vocab_size)
        )
        pretr_embs = self.encoder.shared(in_vec[~mask]).type(new_embs.dtype)
        new_id_vec = torch.zeros(
            (in_vec.shape[0], self.emb_dim), dtype=new_embs.dtype
        ).to(in_vec.device)
        new_id_vec[~mask] = pretr_embs
        new_id_vec[mask] = new_embs
        return new_id_vec

    def _get_mlp(self, hidden_sizes, dropout=0.0):
        if len(hidden_sizes) == 2:
            return nn.Linear(hidden_sizes[0], hidden_sizes[1])
        mlp = nn.ModuleList()
        for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            mlp.append(
                nn.Sequential(
                    nn.Linear(in_shape, out_shape),
                    nn.LayerNorm(out_shape),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
        return mlp


class PretrainedCodebookEmbedding(nn.Module):

    def __init__(self, num_embeddings, emb_shape, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, emb_shape)
        self.up_proj = nn.Linear(emb_shape, d_model, bias=False)

    def init(self, codebooks):
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        self.embedding.weight.data[1:1025] = codebooks

    def forward(self, input):
        embs = self.embedding(input)
        embs = self.up_proj(embs)
        return embs


def sinusoidal_positional_embedding(
    token_sequence_size, token_embedding_dim, n=10000.0
):

    if token_embedding_dim % 2 != 0:
        raise ValueError(
            "Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(
                token_embedding_dim
            )
        )

    T = token_sequence_size
    d = token_embedding_dim  # d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(
        n, 2 * torch.arange(0, d // 2) / d
    )  # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(
        positions / denominators
    )  # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(
        positions / denominators
    )  # cos(pos/10000^(2i/d_model))

    return embeddings


def expand_emb_layer(model, n_new_embs, hf_name):
    if "t5" in hf_name:
        # we have encoder decoder -> encoder architecture, embeddings get extended, decoder embs get newly initialized
        new_emb_layer = PartiallyFixedEmbedding(model.shared.weight.data, n_new_embs)
        model.shared = new_emb_layer
        new_emb_layer = PartiallyFixedEmbedding(
            model.encoder.embed_tokens.weight.data, n_new_embs
        )
        model.encoder.embed_tokens = new_emb_layer
        model.decoder.embed_tokens = nn.Embedding(
            n_new_embs, embedding_dim=new_emb_layer.weight.shape[-1]
        )
    else:
        raise NotImplementedError(
            f"Embedding extension for {hf_name} not supported yet!"
        )


def expand_lm_head(model, n_new_embs, hf_name, rand_lm_head=False):
    if "t5" in hf_name:
        # encoder-decoder architecture -> replace lm head with newly initialized weights
        _, hidden_dim = model.lm_head.weight.data.shape
        model.lm_head = nn.Linear(hidden_dim, n_new_embs, bias=False)
    else:
        raise NotImplementedError(f"LM head extension for {hf_name} not supported yet!")


def initialize_embeddings(model, save_location, instruct_tune=False, device="cuda"):
    # initialize embeddings of the transformer with embeddings coming from RQ-VAE
    # start-index = 1
    # codebook1: 1-256, codebook 2: 257-512 codebook 3: 513-768
    if instruct_tune:
        # for instruction tuning we only need the decoder embedding layer
        model_embs = model.decoder.embed_tokens.weight.data
    else:
        # model.shared is the shared embedding layer of T5
        # TODO: potentially extend to other backbones -> ablation of langauge encoder = train from scratch
        model_embs = model.shared.weight.data

    codebook_embs = torch.load(
        save_location.replace(".pkl", ".pth"), map_location=device
    )
    n_codes = codebook_embs[0].shape[0]
    codebook_embs = torch.cat([p for p in codebook_embs])
    fourth_codes = sinusoidal_positional_embedding(n_codes, codebook_embs.shape[1]).to(
        codebook_embs.device
    )
    codebook_embs = torch.cat([codebook_embs, fourth_codes])
    if model_embs.shape[-1] == codebook_embs.shape[-1]:
        model_embs[1:1025] = codebook_embs
        if instruct_tune:
            model.decoder.embed_tokens.weight.data = model_embs
        else:
            model.shared.weight.data = model_embs
            model.encoder.embed_tokens = model.shared
            model.decoder.embed_tokens = model.shared
    else:
        num_embs, emb_dim = model_embs.shape
        new_embs = PretrainedCodebookEmbedding(
            num_embs, codebook_embs.shape[-1], emb_dim
        )
        new_embs.init(codebook_embs)
        if instruct_tune:
            # if we do instruction tuning, we only initialize the decoder tokens
            model.decoder.embed_tokens = new_embs
        else:
            model.shared = new_embs
            model.encoder.embed_tokens = model.shared
            model.decoder.embed_tokens = model.shared
