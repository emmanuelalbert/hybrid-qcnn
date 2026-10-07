"""
Model 2: CNN + VQC in the MIDDLE. (See earlier version for full design notes.)

Now accepts init_method/init_std, passed through to QuantumLayer.
"""

import torch
import torch.nn as nn

from cnn_backbone import CNNBackbone
from quantum_layer import QuantumLayer


class CNNVQCMiddle(nn.Module):
    def __init__(self, n_qubits: int = 4, depth: int = 2, hidden_dim: int = 16, num_classes: int = 10,
                 init_method: str = "default", init_std: float = 0.1):
        super().__init__()
        self.backbone = CNNBackbone(reduced_dim=n_qubits, angle_scale=False)
        self.vqc = QuantumLayer(n_qubits=n_qubits, depth=depth, init_method=init_method, init_std=init_std)
        self.post_vqc = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.extract_features(x)
        reduced = self.backbone.reduce(feats)
        angle_in = torch.tanh(reduced) * torch.pi
        q_out = self.vqc(angle_in)
        hidden = self.post_vqc(q_out)
        logits = self.classifier(hidden)
        return logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict:
        return {
            "backbone": self.backbone.n_params(),
            "vqc": self.vqc.n_params(),
            "post_vqc": sum(p.numel() for p in self.post_vqc.parameters()),
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "total": self.n_params(),
        }

    def initial_grad_probe(self, batch_size: int = 32) -> float:
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
    model = CNNVQCMiddle()
    print("Param breakdown:", model.param_breakdown())

    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print(f"Output shape: {out.shape}")

    loss = out.sum()
    loss.backward()
    print("Sanity check passed: forward + backward both work.")