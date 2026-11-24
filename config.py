"""Central hyperparameters for the two-layer model.

Values follow Cadieu & Olshausen (2012).
Layer-specific config dataclasses import these constants; 
the shared tensor operations in ``models.ops`` do not depend
on this module.
"""

import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = _PROJECT_ROOT / "natural-movies"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(DEFAULT_DATA_ROOT)))
OUTPUT_DIR = _PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PATCH_SIZE = 32
CLIP_LENGTH = 64
CROP_MARGIN_TOP = 15
CROP_MARGIN_LR = 4
CROP_MARGIN_BOTTOM = 4

# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------
WHITENING_PIXEL_NOISE_FRACTIONAL_VARIANCE = 0.01
WHITENING_PIXEL_NOISE_VARIANCE_CUTOFF_RATIO = 1.25
WHITENING_X_NOISE_FRACTION = 8.0
WHITENING_X_NOISE_VAR = 0.01
WHITENING_CROPS_PER_CHUNK = 20
WHITENING_PATCHES_PER_CROP = 20

# ---------------------------------------------------------------------------
# Layer 1: complex basis A
# ---------------------------------------------------------------------------
LAYER1_N_FEATURES = 1024
LAYER1_LAMBDA_A = 10.0          # Cauchy sparsity weight
LAYER1_BETA_A = 0.5             # slowness weight (energy uses an extra 0.5)
LAYER1_SIGMA_SP_A = 0.4         # Cauchy sparsity scale
LAYER1_A_THRESH = math.exp(-4)  # amplitude threshold for log-amplitude
LAYER1_SWAP_REAL_IMAG = True  # Swaps Re/Im of the Gram-Schmidt-projected basis; phases must be transformed as phi' = pi/2 - phi to leave the reconstruction unchanged.
LAYER1_EPS = 1e-12

# ---------------------------------------------------------------------------
# Layer 1 inference
# ---------------------------------------------------------------------------
LAYER1_INFER_ITERS = 120
LAYER1_LBFGS_ITERS = 100  # Max L-BFGS iterations per train_layer1 inference call.
LAYER1_STEP_A = 5e-5
LAYER1_STEP_PHI = 5e-4
# The paper's coefficient gradients (not the natural-gradient variant).
LAYER1_NATURAL_GRADIENT = False
LAYER1_INFER_INIT = "random"   # Random initialisation of coefficients.
# Resets the phase to the residual correlation angle whenever an amplitude is clamped at zero.
LAYER1_RESET_PHASE_ON_NEGATIVE_A = True
# Inference stops early when the relative energy change falls below this.
LAYER1_INFER_CONV_TOL = 1e-4

# ---------------------------------------------------------------------------
# Layer 1 training
# ---------------------------------------------------------------------------
LAYER1_EPOCHS = 80
LAYER1_ANNEAL_EPOCHS = 20
LAYER1_TRIALS_PER_EPOCH = 1000  # clips sampled and basis updates per epoch
LAYER1_BATCH_SIZE = 64          # clips per batched inference call; one basis update per batch
LAYER1_ETA = 1e-4
LAYER1_ETA_TARGET = 0.05
LAYER1_ETA_ANNEAL_FACTOR = 0.25
LAYER1_LOG_EVERY = 100
LAYER1_SAVE_EVERY = 1
SEED = 42

# ---------------------------------------------------------------------------
# Layer 2: form (amplitude, B) and motion (phase-transition, D)
#
# Both sub-models are trained after layer 1 has converged, on MAP estimates
# (a, phi) inferred by a frozen first-layer basis A. They share the layer-1
# feature count N (= n_input_dims). Paper values: 625 basis functions for both
# B and D. Clips are 64 frames because raw chunks contain 64 frames each.
# ---------------------------------------------------------------------------
LAYER2_K = 625           # number of form (amplitude) basis functions in B
LAYER2_L = 625          # number of motion (phase-transition) basis functions in D

# Form 
LAYER2_LAMBDA_V = 2        # sparsity weight on v
LAYER2_BETA_V = 10           # slowness weight on v (energy uses extra 0.5)
LAYER2_FORM_SIGMA_A2 = 0.2   # sigma_a^2 scaling for the form data term (notebook sigma_a**2)
LAYER2_A_THRESH = math.exp(-4)  # Amplitude threshold for log-amplitude (matches layer 1).

# Motion
LAYER2_LAMBDA_W = 0.5       # sparsity weight on w
LAYER2_BETA_W = 5            # slowness weight on w (energy uses extra 0.5)
LAYER2_W_SIGMA = .2236           # (sqrt 0.5) Cauchy sparsity scale for w
LAYER2_KAPPA = 4             # phase noise factor for the circular data term
LAYER2_EPS = 1e-12             # epsilon for normalisation / numerical stability
LAYER2_MOTION_A_THRESH = 0.2    # Motion activation threshold on amplitude (notebook motion_a_thresh)
LAYER2_MOTION_A_ON_FRACTION = 0.25   # Target fraction of active entries; if not None, the threshold is computed adaptively as the percentile of layer-1 amplitudes (matches MATLAB init_phasetrans.m).

# Layer 2 inference
LAYER2_INFER_ITERS = 100       # Max L-BFGS iterations per inference call
LAYER2_INFER_CONV_TOL = 1e-4
LAYER2_NATURAL_GRADIENT = False


# Layer 2 training
LAYER2_EPOCHS = 40
LAYER2_ANNEAL_EPOCHS = 10
LAYER2_TRIALS_PER_EPOCH = 1000
LAYER2_BATCH_SIZE = 64          # clips per batched inference call; one basis update per batch
LAYER2_ETA = 1e-4
LAYER2_ETA_TARGET = 0.05
LAYER2_ETA_ANNEAL_FACTOR = 0.25
LAYER2_LOG_EVERY = 100
LAYER2_SAVE_EVERY = 1

# ---------------------------------------------------------------------------
# BSD500 still-image dataset (rotation applied at sampling time)
# ---------------------------------------------------------------------------
BSD500_DATA_ROOT = Path(os.environ.get("BSD500_DATA_ROOT", str(_PROJECT_ROOT / "bsd500")))
BSD500_NPY_DIR = _PROJECT_ROOT / "bsd500-npy"
BSD500_IMAGE_SIZE = 128          # canonical grayscale square size (matches natural movies)
