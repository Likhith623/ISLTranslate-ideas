"""
Loss Functions for the Articulatory-Bound Spatiotemporal Transformer.

Combines:
1. Label-Smoothed Translation Cross-Entropy
2. Dominant-Hand CTC Loss
3. Deformable Keypoint Offset Regularization
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from .config import LossConfig


if _HAS_TORCH:
    class CompositeABSTLoss(nn.Module):
        """
        Calculates multi-task training loss with numerical stability guarantees.
        """
        def __init__(self, config: Optional[LossConfig] = None):
            super().__init__()
            self.cfg = config or LossConfig()
            self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        def forward(
            self,
            model_outputs: Dict[str, Any],
            target_ids: Optional[torch.Tensor] = None,
            ctc_targets: Optional[torch.Tensor] = None,
            ctc_target_lengths: Optional[torch.Tensor] = None,
            ctc_input_lengths: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Args:
                model_outputs: Dictionary returned by ArticulatoryTransformer.forward().
                target_ids: (B, L) ground truth translation token IDs.
                ctc_targets: Optional (B, S) or 1D concatenated CTC targets.
                ctc_target_lengths: Optional (B,) length per target.
                ctc_input_lengths: Optional (B,) input frame counts.
            Returns:
                total_loss: Scalar differentiable tensor.
                metrics: Dict of float loss values for logging.
            """
            metrics: Dict[str, float] = {}
            total_loss = torch.tensor(0.0, device=target_ids.device if target_ids is not None else "cpu")

            # 1. Translation Cross-Entropy Loss
            translation_loss = model_outputs.get("translation_loss")
            if translation_loss is not None:
                total_loss = total_loss + self.cfg.translation_weight * translation_loss
                metrics["loss/translation"] = translation_loss.item()

            # 2. Multi-Task CTC Loss
            ctc_logits = model_outputs.get("ctc_logits")
            if (
                ctc_logits is not None
                and ctc_targets is not None
                and ctc_target_lengths is not None
                and ctc_input_lengths is not None
            ):
                ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
                ctc_loss = self.ctc_loss_fn(
                    ctc_log_probs,
                    ctc_targets,
                    ctc_input_lengths,
                    ctc_target_lengths,
                )
                if not torch.isnan(ctc_loss) and not torch.isinf(ctc_loss):
                    total_loss = total_loss + self.cfg.ctc_weight * ctc_loss
                    metrics["loss/ctc"] = ctc_loss.item()

            # 3. Deformable Offset L2 Regularization (keeps offsets local)
            offsets = model_outputs.get("offsets")
            if offsets is not None:
                offset_reg = torch.mean(offsets ** 2)
                total_loss = total_loss + 0.05 * offset_reg
                metrics["loss/offset_reg"] = offset_reg.item()

            metrics["loss/total"] = total_loss.item()
            return total_loss, metrics
else:
    class CompositeABSTLoss:  # type: ignore
        def __init__(self, *args, **kwargs): pass

