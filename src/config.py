from typing import Optional, Union, Literal
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BiCEBertConfig(PretrainedConfig):
    model_type = "BiCEbert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50368,
        hidden_size: int = 768,
        intermediate_size: int = 1152,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        left_heads_ratio: float = 0.5,
        hidden_activation: str = "gelu",
        initializer_range: float = 0.02,
        initializer_cutoff_factor: float = 2.0,
        norm_eps: float = 1e-5,
        norm_bias: bool = False,
        pad_token_id: int = 50283,
        eos_token_id: int = 50282,
        bos_token_id: int = 50281,
        cls_token_id: int = 50281,
        sep_token_id: int = 50282,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        mlp_bias: bool = False,
        mlp_dropout: float = 0.0,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: float = 0.0,
        classifier_bias: bool = False,
        classifier_activation: str = "gelu",
        sparse_prediction: bool = False,
        sparse_pred_ignore_index: int = -100,
        reference_compile: Optional[bool] = None,
        repad_logits_with_grad: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.left_heads_ratio = left_heads_ratio
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad

        if not (0.0 < self.left_heads_ratio < 1.0):
            raise ValueError("left_heads_ratio must be in (0, 1)")
        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError('classifier_pooling must be "cls" or "mean"')

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output