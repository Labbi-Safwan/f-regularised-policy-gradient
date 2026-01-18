# f-regularised-policy-gradient

This repository provides the code for the paper **Beyond Softmax and Entropy: Improving  Convergence Guarantees of Policy Gradients by f-SoftArgmax Parameterization with Coupled Regularization**. Our Tsallis-PPO implementation is built on top of the `PureJaxRL` repository (https://github.com/luchris429/purejaxrl), developed by the FLAIR team (University of Oxford). We also deeply thank Vincent Roulet for sharing an implementation of the alpha-Tsallis SoftArgmax.

## Repository layout

- `Tabular/`
  - `main.py`: run tabular experiments (DeepSea, NChain) with hyperparameter grids and multiple seeds.
  - `algos_jax.py`: policy-gradient update code and policy parameterisations.
  - `fdiv.py`: f-divergences and `softargmax` operators (JAX).
  - `envs/`: small JAX environments (`deepsea_env/`, `nchain_env/`).
  - `run_all.py`: convenience launcher to run a predefined grid across algorithms/environments/sizes.
  - `plot_xp.py`: plotting script for tabular experiments (writes PDFs).

- `DeepRL/`
  - `ppo_deepsea.py`: Tsallis-PPO style experiment on Gymnax `DeepSea-bsuite`.
  - `ppo_cartpool_noise.py`: Tsallis-PPO style experiment on noisy CartPole.
  - `fdiv.py`: same divergence/operator utilities as above (duplicated for convenience).
  - `plot_deepsea.py`, `plot_noisy_cartpool.py`: plotting scripts (write PDFs).

- `Landscape_plots/`
  - `Landscape_plot.ipynb`: notebook for objective / landscape visualisations.

## Installation

The simplest workflow is to create a virtual environment and install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Core deps
pip install numpy matplotlib tqdm

# JAX + ecosystem
pip install jax jaxlib flax optax distrax chex gymnax
```

## Tabular experiments (`Tabular/`)

### What is implemented

`Tabular/algos_jax.py` contains several policy parameterisations / update rules:
- `fpg`: an f-regularised policy-gradient update (Tsallis-style f-divergence utilities).
- `logbarrier`: softmax parameterisation with a log-barrier style regulariser.
- `escort`: escort-transform parameterisation.
- `hadamard`: Hadamard / squared-parameter parameterisation.

### Running a single experiment grid

From the repository root:

```bash
cd Tabular

# Example: fPG on DeepSea of size 15
python main.py \
  --algorithm fpg \
  --environment deepsea \
  --size 15 \
  --n_iteration 20000 \
  --len_truncation 25 \
  --batch_size 2 \
  --step 0.01 0.1 1.0 \
  --alpha 0.1 0.3 0.5 0.7 0.9 1.0 \
  --temperature 0.0001 0.001 0.01 0.1 1.0 \
  --seeds 0 1 2 3
```

Outputs are saved under:

```
Tabular/experiments/<algorithm>/<environment>/size_<N>/<config_folder>/size_<N>_seed_<seed>_true_objective.pkl
```

Each pickle contains a 1D array of length `n_iteration` (the Monte-Carlo estimate of the objective along training).

### Running the predefined grid

```bash
cd Tabular
python run_all.py
```

By default, `run_all.py` runs all combinations in:
- `ALGORITHMS = ["hadamard", "escort", "logbarrier", "fpg"]`
- `ENVIRONMENTS = ["nchain", "deepsea"]`
- `SIZES = [10, 15, 20]`

### Plotting tabular results

```bash
cd Tabular
python plot_xp.py
```

The plotting script is intentionally simple and uses hard-coded settings near the bottom of the file:
- `read_root` (default `./experiments`)
- environment sizes to plot
- list of seeds
- subsampling factor

It writes PDFs to `Tabular/plots/`.

## DeepRL experiments (`DeepRL/`)

These scripts implement a PPO loop with:
- a Tsallis / f-softargmax policy parameterisation (`param_alpha`), and
- a divergence regulariser against the uniform policy (`reg_alpha`) applied to the reward: `r_reg = r - beta * D_alpha(pi || uniform)`.

### DeepSea (Gymnax bsuite)

```bash
cd DeepRL
python ppo_deepsea.py
```

This performs a grid over:
- `reg_alpha`, `param_alpha` in `{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`
- learning rates in `{1e-4, 3e-4, 1e-3}`
- regularisation strengths (named `entropy_coeff` in code)
- multiple RNG seeds

It saves a flat list of records to a pickle file in the current directory:

```
deepsea_results_<ENV_SIZE>_reg_rewards.pkl
```

To change the environment size, edit `general_config["ENV_SIZE"]` in `ppo_deepsea.py`.

### Noisy CartPole

```bash
cd DeepRL
python ppo_cartpool_noise.py
```

This wraps Gymnax CartPole with a small JAX reward-noise wrapper (noise is not applied on the first step after reset, matching bsuite-style noise injection). It saves:

```
cartpole_noisy_results_<NOISE_SCALE>.pkl
```

Change the noise level by editing `general_config["NOISE_SCALE"]`.

### Plotting DeepRL results

```bash
cd DeepRL
python plot_deepsea.py
python plot_noisy_cartpool.py
```

Both scripts expect the corresponding pickle file to exist in the current directory. They write PDFs to `DeepRL/plots/`.

## Reproducibility tips

- You can force JAX to CPU or GPU with environment variables, for example:

```bash
# CPU
export JAX_PLATFORM_NAME=cpu

# GPU (if your JAX install supports it)
export JAX_PLATFORM_NAME=gpu
```

- The scripts store the RNG keys / seeds used in the recorded outputs.

## Notes

- The repo is intentionally minimal and does not include a `requirements.txt`. If you want one, a good starting point is:
  `jax`, `jaxlib`, `flax`, `optax`, `distrax`, `chex`, `gymnax`, `numpy`, `matplotlib`, `tqdm`.
- `Tabular/fdiv.py` and `DeepRL/fdiv.py` currently duplicate the same divergence/operator utilities.
