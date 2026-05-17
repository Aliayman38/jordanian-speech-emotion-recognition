# models/wav2vec_clf.py
"""
Arabic Wav2Vec2 fine-tuning model for Speech Emotion Recognition.

Architecture
------------
* Backbone : ``jonatasgrosman/wav2vec2-large-xlsr-53-arabic``
  (or any HuggingFace Wav2Vec2 checkpoint)
* Pooling  : mean + std over the time dimension → 2H-dim vector
* Head     : LayerNorm → Dropout → Linear(2H, 256) → ReLU
                       → Dropout → Linear(256, num_classes)

Only the last ``UNFREEZE_LAST_N`` encoder layers and the head are trained;
the CNN feature extractor and remaining encoder layers are frozen.

Usage
-----
    from models.wav2vec_clf import Wav2VecEmotionModel
    model = Wav2VecEmotionModel(BASE_MODEL, NUM_CLASSES).to(device)

    # Standard forward
    logits = model(input_values, attention_mask)

    # Forward with intermediate features (needed for fusion)
    logits, feat256, pooled = model(input_values, attention_mask,
                                    return_features=True)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from config import NUM_CLASSES, UNFREEZE_LAST_N


class Wav2VecEmotionModel(nn.Module):
    """
    Wav2Vec2-based emotion classifier with optional feature extraction.

    Parameters
    ----------
    base_model : str
        HuggingFace model identifier.
    num_classes : int
        Number of emotion categories.
    dropout : float
        Dropout probability applied inside the classification head.
    """

    def __init__(
        self,
        base_model: str,
        num_classes: int = NUM_CLASSES,
        dropout: float   = 0.35,
    ) -> None:
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(base_model)
        hidden = self.wav2vec.config.hidden_size

        # Classifier head — keep layer order fixed so saved checkpoints can
        # be reloaded with strict=True.
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),       # [0]
            nn.Dropout(dropout),            # [1]
            nn.Linear(hidden * 2, 256),     # [2]
            nn.ReLU(),                      # [3]
            nn.Dropout(dropout),            # [4]
            nn.Linear(256, num_classes),    # [5]
        )

        # 1. Freeze entire backbone
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        # 2. Selectively unfreeze last N encoder layers
        try:
            for layer in self.wav2vec.encoder.layers[-UNFREEZE_LAST_N:]:
                for p in layer.parameters():
                    p.requires_grad = True
        except Exception:
            print("[WARN] Could not selectively unfreeze encoder layers.")

        # 3. Always keep the CNN feature extractor frozen
        try:
            self.wav2vec.feature_extractor._freeze_parameters()
        except Exception:
            pass

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        """
        Parameters
        ----------
        input_values : Tensor  (B, T)
        attention_mask : Tensor or None  (B, T)
        return_features : bool
            If *True*, return ``(logits, feat256, pooled)`` instead of
            just ``logits``.  Required for fusion feature extraction.

        Returns
        -------
        logits : Tensor  (B, num_classes)
        feat256 : Tensor (B, 256)  — only when return_features=True
        pooled  : Tensor (B, 2H)   — only when return_features=True
        """
        hidden_states = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask,
        ).last_hidden_state                          # (B, T, H)

        mean_pool = hidden_states.mean(dim=1)        # (B, H)
        std_pool  = hidden_states.std(dim=1)         # (B, H)
        pooled    = torch.cat([mean_pool, std_pool], dim=1)  # (B, 2H)

        # Step through classifier manually to expose feat256
        x      = self.classifier[0](pooled)   # LayerNorm
        x      = self.classifier[1](x)        # Dropout
        x      = self.classifier[2](x)        # Linear(2H → 256)
        feat   = self.classifier[3](x)        # ReLU  → 256-dim features
        x      = self.classifier[4](feat)     # Dropout
        logits = self.classifier[5](x)        # Linear(256 → num_classes)

        if return_features:
            return logits, feat, pooled
        return logits
