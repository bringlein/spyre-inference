# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre OOT replacement for GemmaRMSNorm.

Gemma models (1/2/3) use GemmaRMSNorm for every normalization (input/post-attn/
pre-post-feedforward layernorms and gemma-3's per-head q_norm/k_norm). The stock
forward routes through `ir.ops.rms_norm`, whose native implementation promotes to
float32 (`x.to(torch.float32)`) — unsupported on Spyre, producing NaN/Inf garbage.
This mirrors SpyreRMSNorm (fp16, no promotion) with Gemma's two differences:
`x * (1 + w)` instead of `x * w`.

References:
    - Upstream GemmaRMSNorm: vllm/model_executor/layers/layernorm.py
    - SpyreRMSNorm: spyre_inference/custom_ops/rms_norm.py
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

logger = init_logger(__name__)


@GemmaRMSNorm.register_oot(name="GemmaRMSNorm")
class SpyreGemmaRMSNorm(GemmaRMSNorm):
    """Out-of-tree (OOT) GemmaRMSNorm implementation for IBM's Spyre."""

    _dynamic_arg_dims = {"x": [], "residual": []}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled_forward_spyre = self.maybe_compile(self._forward_spyre_impl)

        logger.warning_once(
            "SpyreGemmaRMSNorm: no dtype promotion is performed, "
            "expect numerical differences to upstream vLLM."
        )

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._compiled_forward_spyre(
            x,
            self.variance_epsilon,
            self.weight.data,
            residual,
        )

    @staticmethod
    def _forward_spyre_impl(
        x: torch.Tensor,
        variance_epsilon: float,
        weight: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """GemmaRMSNorm kernel for Spyre. Compiled separately via maybe_compile."""
        if residual is not None:
            x = x + residual
            residual = x

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight)

        if residual is None:
            return x
        return x, residual
