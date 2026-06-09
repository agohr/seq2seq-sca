# Supplementary Code and Data for the Paper "Robust Single-Trace Full-Key Extraction from Million-Point Traces With Cross-Implementation Transfer"

This repository contains code and data for reproducing the experiments in the paper.

We provide some data from the CM0 and CM3 implementations (Bursztein et al., TCHES 2024, see [here](https://github.com/google/scaaml/tree/main/papers/2024/GPAM) for their full datasets) at the low resolution required for our experiments to run. The data given should be sufficient to run training on CM0 and to test zero-shot transfer of a U-Net-CTC model to CM3 after large-scale segmentation of the traces.

For experiments involving CM1 and CM2, or for experiments that involve the full CM0 or CM3 datasets, please download the datasets from the SCAAML/GPAM repository (see [here](https://github.com/google/scaaml/tree/main/papers/2024/GPAM)). After downloading the datasets, use the `resize_data` function in `processing.py` to downsample the traces to the resolution your experiments require.

Pre-trained models are also provided.

The rationale for the choice of experiments is to provide a minimal set of experiments that demonstrates all the main ideas and results of the paper, and to provide the tools to reproduce all results reported.

## Brief Overview of Main Notebooks

**SIFT synthetic task** (`alignment_synthetic.ipynb`, `alignment_synthetic_ctc.ipynb`): These correspond to the SIFT experiments in Sections 3.2 and 5.2 of the paper. `alignment_synthetic.ipynb` reproduces the failure of VGG-like and subsampling-based networks on SIFT-512-32 (cf. Figure 4), illustrating the indexing bottleneck that also appears on real SCAAML data. `alignment_synthetic_ctc.ipynb` shows that U-Net-CTC solves the same task, which confirms the theoretical expectation that it should.

**CM0 full-key extraction** (`cm0_subsampler_test_feb_2026.ipynb`): Covers part of the CM0 results from Section 5.1: full-key extraction of all 256 key bits in a single forward pass with the subsampling-based network (with a Random Forest baseline for comparison), after hard low-pass downsampling to 2000 points. The single-byte VGG extraction and the CM1 segmentation pipeline (99%/98.3% per-share accuracy) discussed in Section 5.1 are not included here, as they are superseded by the U-Net-CTC pipeline. 

However, training the subsampling-based network on CM1 is not hard with the tools provided here: one only needs to segment the CM1 traces using the fourier interpolation-based segmentation algorithm in `seglib.py`, identify the two largest segments, upsample them as in the paper, and train on both key shares simultaneously. The training itself works exactly as CM0 training after these dataset preparation steps have been performed.

**CM0 U-Net transfer to CM3** (`cm0_unet_test_feb_2026.ipynb`): An additional experiment not in the main paper. Tests a U-Net trained on CM0 directly on CM3. Because the CM0 model has never seen the inter-operation regions that appear in CM3 traces, the CM3 traces must first be resampled and then the six scalar-multiplication windows manually cut out at fixed offsets before each segment is decoded independently.

**U-Net-CTC on CM1 and CM3, without segmentation** (`ctc_scaaml_data_train.ipynb`, `ctc_scaaml_data_test_feb_2026.ipynb`): These implement the unsegmented sequence-to-sequence pipeline of Section 5.4. `ctc_scaaml_data_train.ipynb` trains a U-Net-CTC model directly on unsegmented CM1 holdout traces downsampled to 5000 points, without any segmentation into key shares. `ctc_scaaml_data_test_feb_2026.ipynb` runs the evaluation: robustness of the trained model on CM1 under trace corruptions and rotations, the self-explanatory blank-channel analysis, and the zero-shot full-trace CM3 attack, where blank-channel analysis with short-burst suppression is used to locate and decode the six scalar-multiplication regions. The segmented Section 5.3 experiments (the 21-/201-epoch models and the Table 3 cross-share/cross-dataset transfer results) are not reproduced by these notebooks.

## Project Structure

```
paper_artifact/
├── data/                  # Pre-trained models and datasets
│   ├── cm0/               # CM0 trace data
│   ├── cm3_test/          # CM3 cross-implementation test data
│   └── pretrained/        # Pre-trained model weights and architectures
├── lib/                   # Python library modules
│   ├── model_library.py           # Neural network architectures (MLP, VGG_1D, ResNet_1D, U-Net, Transformer)
│   ├── learning_rates.py          # Cyclic learning rate schedulers
│   ├── experiments.py             # Experiment utilities (occlusion, gradient saliency, synthetic data, CTC training)
│   ├── processing.py              # Signal processing (resampling, resizing)
│   ├── complex_system_simulation.py  # Trace augmentation (noise, interrupts, jitter, frequency changes)
│   └── seglib.py                  # Trace segmentation utilities
├── notebooks/             # Jupyter notebooks for experiments
│   ├── alignment_synthetic.ipynb           # Synthetic alignment experiments
│   ├── alignment_synthetic_ctc.ipynb       # Synthetic alignment with CTC decoding
│   ├── cm0_subsampler_test_feb_2026.ipynb  # CM0 subsampler model tests
│   ├── cm0_unet_test_feb_2026.ipynb        # CM0 U-Net with CTC tests
│   ├── ctc_scaaml_data_train.ipynb         # U-Net CTC training on CM1 data
│   └── ctc_scaaml_data_test_feb_2026.ipynb # U-Net CTC evaluation and cross-implementation transfer
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

## Installation

### 1. Create a Python virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Purpose |
|---------|---------|
| `tensorflow` | Deep learning framework (includes Keras) |
| `numpy` | Numerical computing |
| `matplotlib` | Plotting and visualization |
| `scikit-learn` | Machine learning utilities (Random Forest, Decision Tree, SGD) |
| `scipy` | Signal processing (resampling, convolution, interpolation) |
| `tqdm` | Progress bars |
| `python-Levenshtein` | String distance metrics for CTC evaluation |
| `fuzzywuzzy` | Fuzzy string matching for CTC evaluation |
| `optuna` | Hyperparameter optimization (used by segmentation library) |

### 3. GPU support (optional)

If you have an NVIDIA GPU and want GPU-accelerated training, ensure you have the appropriate CUDA toolkit and cuDNN installed. See the [TensorFlow GPU guide](https://www.tensorflow.org/install/pip) for details.

### 4. Set up the Python path

The notebooks expect the `lib/` directory to be on the Python path. You can do this by either:

**Option A**: Set the `PYTHONPATH` environment variable before launching Jupyter:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/lib"
jupyter notebook notebooks/
```

**Option B**: Add the path inside each notebook (a `sys.path.append` cell is typically included).

## Running the Notebooks

After installation, launch Jupyter and open any notebook from the `notebooks/` directory:

```bash
jupyter notebook notebooks/
```

The notebooks are self-contained and are, we hope, easy to follow.

## Citing the Paper

If you use this code or data in your own research, please cite the paper as follows:

```bibtex
@inproceedings{GLL26,
  author    = {Aron Gohr and Friederike Laus and Gregor Leander},
  title     = {Robust Single-Trace Full-Key Extraction from Million-Point Traces With Cross-Implementation Transfer},
  booktitle = {Advances in Cryptology -- {CRYPTO} 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
}
```

