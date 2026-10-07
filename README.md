# Hybrid Quantum-Classical Convolutional Architectures for Image Classification

A systematic empirical study of where (and whether) a **variational quantum circuit (VQC)** helps when inserted into a classical CNN pipeline for image classification.

The study uses parameter-matched quantum/classical model pairs, multi-seed runs, and paired non-parametric statistics, so that the effect of *quantumness* is separated from the effect of *model capacity*.

---

## Research Questions

1. Does inserting a VQC into a classical CNN pipeline improve classification accuracy?
2. Can a VQC achieve comparable performance with **fewer trainable parameters**?
3. Does the **placement** of the VQC (middle of the network vs. output stage) change accuracy or training stability?

---

## Key Findings (MNIST)

| Finding | Result |
|---|---|
| Classical vs. parameter-matched VQC | Classical models significantly outperform VQC variants in both placements (Wilcoxon, p < 0.01) |
| Accuracy gap when VQC trains successfully | ~3 percentage points below classical |
| Training stability, middle placement | ~30% seed failure rate, with a barren-plateau-style signature |
| Training stability, output placement | ~10% seed failure rate |
| Small-angle initialization | Did not reduce failure rates in either placement |

**Takeaway:** the instability of middle-placement VQCs appears to originate upstream of the VQC's own parameters. The same seeds (3 and 9) failed under both initialization strategies.

> Replace or extend this table with exact accuracies, mean ± std, and p-values from your final result files.

---

## Study Design

- **Placements compared:** VQC in the middle of the network vs. VQC at the output stage
- **Baselines:** parameter-matched classical counterparts, so differences are not explained by capacity
- **Seeds:** 10 independent seeds per configuration
- **Statistics:** paired Wilcoxon signed-rank test (preferred when failure/outlier seeds are present), paired t-test, Fisher's exact test (seed failure rates)
- **Ablations:** initialization strategy (standard vs. small-angle) and placement
- **Hardware:** CPU-only training, to keep timing comparisons between simulated quantum and classical models architecturally fair

### Quantum layer

| Setting | Value |
|---|---|
| Framework | PennyLane (`qml.qnn.TorchLayer`) |
| Device | `default.qubit` |
| Differentiation | `backprop` |
| Ansatz | `StronglyEntanglingLayers` |
| Depth | 2 |
| Qubits | 4 |

---

## Datasets

| Dataset | Status | Notes |
|---|---|---|
| MNIST | Complete | Custom CNN backbone |
| Tamil numerals | In progress | ~400 images/class, 10 classes, folder-per-class layout. Pretrained ResNet-18 backbone with only `layer4` fine-tuned and earlier layers frozen |

The Tamil numerals extension repeats the identical experimental design to test whether pretrained ResNet-18 features interact differently with VQC placement than the custom CNN backbone did.

> Datasets are **not** included in this repository. See [Data](#data) for how to obtain and arrange them.

---

## Repository Structure

> Adjust this to match your actual layout.

```
hybrid-qcnn/
├── data/                  # dataset loaders (e.g. tamil_dataset.py)
├── models/                # backbone, quantum layer, placement variants
├── training/              # training loop, multi-seed runner
├── analysis/              # statistics, tables, figures
├── results/               # small result files (CSV/JSON), figures
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/hybrid-qcnn.git
cd hybrid-qcnn

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Core dependencies:** Python 3.9+, PyTorch, torchvision, PennyLane, NumPy, SciPy, Matplotlib, pandas.

---

## Data

**MNIST** downloads automatically through torchvision on first run.

**Tamil numerals** expects a folder-per-class layout:

```
data/tamil_numerals/
├── 0/
├── 1/
├── ...
└── 9/
```

The loader handles a known `ImageFolder` quirk: class folders with numeric names are sorted alphabetically as strings (`0, 1, 10, 2, ...`), which would scramble labels. The custom dataset class enforces numeric ordering.

---

## Usage

> Replace the script names below with your actual entry points.

```bash
# Train a single configuration
python train.py --model vqc_middle --seed 0

# Run all seeds (resumable: skips runs whose results already exist)
python run_seeds.py --model vqc_output --seeds 10

# Run statistical analysis and generate tables/figures
python analyze.py
```

The multi-seed runner uses skip-if-exists logic, so an interrupted experiment can be restarted without repeating completed seeds.

---

## Reproducibility

- Fixed per-run seeds (0 to 9) for model initialization and data ordering
- Parameter counts are reported for every model pair
- CPU-only execution for all reported timing comparisons
- Failed seeds are **kept and reported**, not silently dropped; this is why paired Wilcoxon tests are used

---

## Roadmap

- [x] MNIST study with parameter-matched placement comparison
- [x] Statistical analysis and manuscript draft
- [ ] Tamil numerals study with ResNet-18 backbone
- [ ] Replicate multi-seed pipeline and ablations on Tamil numerals
- [ ] Investigate the upstream source of middle-placement instability

---

## Citation

If you use this code, please cite:

```bibtex
@misc{hybrid_qcnn,
  author = {Emmanuel},
  title  = {Hybrid Quantum-Classical Convolutional Architectures for Image Classification},
  year   = {2026},
  url    = {https://github.com/YOUR-USERNAME/hybrid-qcnn}
}
```

## Author

**Emmanuel**, MSc Data Science, VIT Chennai
