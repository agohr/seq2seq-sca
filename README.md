# Paper Artifact

This repository contains the code and data for reproducing the experiments in the paper.

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

The notebooks are self-contained and include inline comments explaining each experiment step.
