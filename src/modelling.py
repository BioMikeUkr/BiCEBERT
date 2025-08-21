from typing import Optional, Union, Literal
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from .config import BiCEBertConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging



logger = logging.get_logger(__name__)


class BiCEBertEmbeddings(nn.Module):
    def __init__(self, config: BiCEBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = inputs_embeds if inputs_embeds is not None else self.tok_embeddings(input_ids)
        return self.drop(self.norm(x))


class BiCEBertMLP(nn.Module):
    def __init__(self, config: BiCEBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(a) * g))


class BiCEBertAttention(nn.Module):
    def __init__(self, config: BiCEBertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.n_left = int(round(self.num_heads * config.left_heads_ratio))
        self.n_left = max(1, min(self.num_heads - 1, self.n_left))
        self.n_right = self.num_heads - self.n_left
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.attention_dropout = config.attention_dropout

    def _directional_masks(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        pad = _prepare_4d_attention_mask(attention_mask, dtype)
        T = attention_mask.shape[1]
        z = torch.zeros((T, T), device=attention_mask.device, dtype=dtype)
        neg = torch.finfo(dtype).min
        left = torch.triu(torch.full_like(z, neg), 1)[None, None, :, :] + pad
        right = torch.tril(torch.full_like(z, neg), -1)[None, None, :, :] + pad
        return left, right

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        B, T, C = x.shape
        if attention_mask is None:
            attention_mask = torch.ones((B, T), device=x.device, dtype=torch.bool)
        left_mask, right_mask = self._directional_masks(attention_mask, x.dtype)
        qkv = self.Wqkv(x).view(B, T, 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)
        s = self.head_dim**-0.5

        lq, lk, lv = q[:, :self.n_left], k[:, :self.n_left], v[:, :self.n_left]
        ls = torch.matmul(lq, lk.transpose(-2, -1)) * s
        ls = ls + left_mask
        la = F.softmax(ls, dim=-1, dtype=torch.float32).to(lq.dtype)
        la = F.dropout(la, p=self.attention_dropout, training=self.training)
        lc = torch.matmul(la, lv)

        if self.n_right > 0:
            rq, rk, rv = q[:, self.n_left:], k[:, self.n_left:], v[:, self.n_left:]
            rs = torch.matmul(rq, rk.transpose(-2, -1)) * s
            rs = rs + right_mask
            ra = F.softmax(rs, dim=-1, dtype=torch.float32).to(rq.dtype)
            ra = F.dropout(ra, p=self.attention_dropout, training=self.training)
            rc = torch.matmul(ra, rv)
            ctx = torch.cat([lc, rc], dim=1)
            if output_attentions:
                attn = torch.cat([la, ra], dim=1)
        else:
            ctx = lc
            if output_attentions:
                attn = la

        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_drop(self.Wo(ctx))
        if output_attentions:
            return out, attn
        return (out,)


class BiCEBertEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BiCEBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.attn_norm = nn.Identity() if (layer_id == 0) else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = BiCEBertAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = BiCEBertMLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        a = self.attn(self.attn_norm(hidden_states), attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = hidden_states + a[0]
        m = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + m
        if output_attentions and len(a) > 1:
            return hidden_states, a[1]
        return (hidden_states,)


class BiCEBertPreTrainedModel(PreTrainedModel):
    config_class = BiCEBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BiCEBertEmbeddings", "BiCEBertEncoderLayer"]

    def _init_weights(self, module: nn.Module):
        cutoff = self.config.initializer_cutoff_factor or 3.0

        def init_w(m: nn.Module, std: float):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-cutoff * std, b=cutoff * std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, BiCEBertEmbeddings):
            init_w(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, BiCEBertMLP):
            init_w(module.Wi, stds["in"])
            init_w(module.Wo, stds["out"])
        elif isinstance(module, BiCEBertAttention):
            init_w(module.Wqkv, stds["in"])
            init_w(module.Wo, stds["out"])
        elif isinstance(module, BiCEBertPredictionHead):
            init_w(module.dense, stds["out"])
        elif isinstance(module, BiCEBertForMaskedLM):
            init_w(module.decoder, stds["out"])
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


class BiCEBertModel(BiCEBertPreTrainedModel):
    def __init__(self, config: BiCEBertConfig):
        super().__init__(config)
        self.embeddings = BiCEBertEmbeddings(config)
        self.layers = nn.ModuleList([BiCEBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, ...], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            B, T = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            B, T = input_ids.shape[:2]
            device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((B, T), device=device, dtype=torch.bool)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_out = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden_states = layer_out[0]
            if output_attentions and len(layer_out) > 1:
                all_self_attentions = all_self_attentions + (layer_out[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)


class BiCEBertPredictionHead(nn.Module):
    def __init__(self, config: BiCEBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(x)))


class BiCEBertForMaskedLM(BiCEBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: BiCEBertConfig):
        super().__init__(config)
        self.model = BiCEBertModel(config)
        self.head = BiCEBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.sparse_prediction = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        x = outputs.last_hidden_state

        if self.sparse_prediction and labels is not None:
            labels = labels.view(-1)
            x = x.view(labels.shape[0], -1)
            mask_tokens = labels != self.sparse_pred_ignore_index
            x = x[mask_tokens]
            labels = labels[mask_tokens]

        logits = self.decoder(self.head(x))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            out = (logits,)
            return ((loss,) + out) if loss is not None else out

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


__all__ = [
    "BiCEBertConfig",
    "BiCEBertModel",
    "BiCEBertPreTrainedModel",
    "BiCEBertEmbeddings",
    "BiCEBertAttention",
    "BiCEBertMLP",
    "BiCEBertEncoderLayer",
    "BiCEBertPredictionHead",
    "BiCEBertForMaskedLM",
]
