"""
Quantum layer: angle-encoded, hardware-efficient VQC wrapped as a PyTorch module.

Design (see project plan):
- n_qubits = 4 (default), configurable up to 6 for later sweeps
- Encoding: qml.AngleEmbedding (one RY rotation per feature/qubit)
- Ansatz: qml.StronglyEntanglingLayers (RX/RY/RZ per qubit + ring CNOTs), depth-configurable
- Measurement: PauliZ expectation on every qubit -> n_qubits-dim output
- Device: default.qubit (swap to lightning.qubit later for speed)

Param count check (depth=2, n_qubits=4): 2 * 4 * 3 = 24 trainable quantum weights.

WEIGHT INITIALIZATION (added after multi-seed testing found ~10-30% of random seeds land
in a barren-plateau-style failure -- near-zero gradients from epoch 1, loss stuck at
ln(n_classes)):
- "default": PennyLane's default TorchLayer init, uniform over [0, 2*pi) for every weight.
  This is the widest possible search space and is what all earlier results used.
- "small_angle": weights drawn from a narrow band around 0 (Normal(0, small_std)). Small-angle
  initialization is a documented mitigation for barren plateaus in hardware-efficient
  ansaetze (e.g. Grant et al. 2019's identity-block strategy is a related idea) -- starting
  near the "flat but not stuck" identity-like point of the circuit rather than a uniformly
  random point across the full 2*pi range gives the optimizer a better chance of finding
  a useful gradient direction early on.
"""

import pennylane as qml
import torch
import torch.nn as nn


def _small_angle_init(tensor: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Fill an existing tensor with Normal(0, std) values in-place, matching the
    signature PennyLane's TorchLayer expects for a custom init_method callable
    (it passes a pre-shaped tensor in, and expects the filled tensor back)."""
    return tensor.normal_(mean=0.0, std=std)


def build_vqc(n_qubits: int = 4, depth: int = 2, device_name: str = "default.qubit",
              init_method: str = "default", init_std: float = 0.1):
    """Returns a qml.qnn.TorchLayer wrapping the angle-encoding + strongly-entangling circuit.

    init_method: "default" (PennyLane's built-in Uniform(0, 2*pi)) or "small_angle"
        (Normal(0, init_std), a barren-plateau mitigation strategy).
    """

    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
    weight_shapes = {"weights": weight_shape}

    if init_method == "small_angle":
        init_fn = lambda tensor: _small_angle_init(tensor, std=init_std)
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_fn)
    elif init_method == "default":
        return qml.qnn.TorchLayer(circuit, weight_shapes)
    else:
        raise ValueError(f"Unknown init_method: {init_method!r}. Use 'default' or 'small_angle'.")


class QuantumLayer(nn.Module):
    """Thin nn.Module wrapper so it composes cleanly inside larger models.

    Input:  (batch, n_qubits) -- pre-scaled features (apply tanh * pi/2 upstream, or similar)
    Output: (batch, n_qubits) -- PauliZ expectation values, range [-1, 1]
    """

    def __init__(self, n_qubits: int = 4, depth: int = 2, device_name: str = "default.qubit",
                 init_method: str = "default", init_std: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.init_method = init_method
        self.vqc = build_vqc(n_qubits, depth, device_name, init_method=init_method, init_std=init_std)

    def forward(self, x):
        return self.vqc(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def initial_grad_probe(self) -> float:
        """Quick diagnostic: run one forward+backward on random input and return the mean
        gradient norm at initialization, BEFORE any training. A near-zero value here is a
        strong early warning sign of a barren-plateau-style start for this particular init.
        """
        x = torch.rand(4, self.n_qubits) * torch.pi
        out = self.forward(x)
        loss = out.sum()
        loss.backward()
        norms = [p.grad.norm().item() for p in self.parameters() if p.grad is not None]
        for p in self.parameters():
            if p.grad is not None:
                p.grad = None
        return sum(norms) / len(norms) if norms else 0.0


if __name__ == "__main__":
    # Compare initial gradient magnitude across many seeds, for both init strategies,
    # to see whether small_angle reduces the rate of near-zero-gradient starts.
    print("Probing initial gradient norms across seeds (before any training)...\n")

    for init_method in ["default", "small_angle"]:
        near_zero_count = 0
        grad_norms = []
        for seed in range(20):
            torch.manual_seed(seed)
            layer = QuantumLayer(n_qubits=4, depth=2, init_method=init_method)
            g = layer.initial_grad_probe()
            grad_norms.append(g)
            if g < 0.05:
                near_zero_count += 1

        mean_g = sum(grad_norms) / len(grad_norms)
        min_g = min(grad_norms)
        print(f"[{init_method}] mean initial grad norm: {mean_g:.4f}, min: {min_g:.4f}, "
              f"near-zero (<0.05) starts: {near_zero_count}/20")