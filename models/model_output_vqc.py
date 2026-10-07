"""
Model 1: CNN + VQC at the OUTPUT. (See earlier version for full design notes.)

Now accepts init_method/init_std, passed through to QuantumLayer, so we can compare
"default" vs "small_angle" VQC weight initialization's effect on the barren-plateau-style
failure rate observed in multi-seed testing.
"""

import torch
import torch.nn as nn

from cnn_backbone import CNNBackbone
from quantum_layer import QuantumLayer


class CNNVQCOutput(nn.Module):
    def __init__(self, n_qubits: int = 4, depth: int = 2, num_classes: int = 10,
                 init_method: str = "default", init_std: float = 0.1):
        super().__init__()
        self.backbone = CNNBackbone(reduced_dim=n_qubits, angle_scale=True)
        self.vqc = QuantumLayer(n_qubits=n_qubits, depth=depth, init_method=init_method, init_std=init_std)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        q_out = self.vqc(feats)
        logits = self.classifier(q_out)
        return logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict:
        return {
            "backbone": self.backbone.n_params(),
            "vqc": self.vqc.n_params(),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "total": self.n_params(),
        }

    def initial_grad_probe(self, batch_size: int = 32) -> float:
        """Forward+backward one random batch through the FULL model (not just the VQC)
        and return the mean gradient norm across all trainable params, before training.
        This captures failures caused by the VQC+classifier init combination, which the
        VQC's standalone probe (QuantumLayer.initial_grad_probe) may miss."""
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        logits = self.forward(x)
        loss = criterion(logits, y)
        loss.backward()
        norms = [p.grad.norm().item() for p in self.parameters() if p.grad is not None]
        for p in self.parameters():
            p.grad = None
        return sum(norms) / len(norms) if norms else 0.0


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CNNVQCOutput()
    print("Param breakdown:", model.param_breakdown())

    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print(f"Output shape: {out.shape}")

    loss = out.sum()
    loss.backward()
    print("Sanity check passed: forward + backward both work.")