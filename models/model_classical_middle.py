"""Model 3b: CNN-only baseline, MIDDLE-placement variant (param-matched to model_middle_vqc.py)."""

import torch
import torch.nn as nn

from cnn_backbone import CNNBackbone
from model_classical import ParamMatchedClassicalLayer


class CNNClassicalBaselineMiddle(nn.Module):
    def __init__(self, n_qubits: int = 4, hidden_dim: int = 16, num_classes: int = 10):
        super().__init__()
        self.backbone = CNNBackbone(reduced_dim=n_qubits, angle_scale=False)
        self.classical_layer = ParamMatchedClassicalLayer(n_qubits=n_qubits)
        self.post_layer = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.extract_features(x)
        reduced = self.backbone.reduce(feats)
        c_out = self.classical_layer(reduced)
        hidden = self.post_layer(c_out)
        logits = self.classifier(hidden)
        return logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict:
        return {
            "backbone": self.backbone.n_params(),
            "classical_layer": self.classical_layer.n_params(),
            "post_layer": sum(p.numel() for p in self.post_layer.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "total": self.n_params(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CNNClassicalBaselineMiddle()
    print("Param breakdown:", model.param_breakdown())
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print(f"Output shape: {out.shape}")
    loss = out.sum()
    loss.backward()
    print("Sanity check passed.")