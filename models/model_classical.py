"""Model 3: CNN-ONLY baseline (classical layer, parameter-matched to the VQC)."""

import torch
import torch.nn as nn

from cnn_backbone import CNNBackbone


class ParamMatchedClassicalLayer(nn.Module):
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.linear = nn.Linear(n_qubits, n_qubits, bias=True)
        self.scale = nn.Parameter(torch.ones(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x)) * self.scale

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CNNClassicalBaseline(nn.Module):
    def __init__(self, n_qubits: int = 4, num_classes: int = 10):
        super().__init__()
        self.backbone = CNNBackbone(reduced_dim=n_qubits, angle_scale=False)
        self.classical_layer = ParamMatchedClassicalLayer(n_qubits=n_qubits)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        c_out = self.classical_layer(feats)
        logits = self.classifier(c_out)
        return logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict:
        return {
            "backbone": self.backbone.n_params(),
            "classical_layer": self.classical_layer.n_params(),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "total": self.n_params(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CNNClassicalBaseline()
    print("Param breakdown:", model.param_breakdown())
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print(f"Output shape: {out.shape}")
    loss = out.sum()
    loss.backward()
    print("Sanity check passed.")