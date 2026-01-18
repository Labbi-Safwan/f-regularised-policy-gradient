import os
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from envs.deepsea_env.deepsea_deterministic import DeepSeaDet
from envs.nchain_env.nchain import NChain  # <- JAX NChain

from algos_jax import build_pg_algos


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def create_folder_if_not_exists(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def save_results_to_pickle(file_path: str, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)



ENV_FACTORIES = {
    "deepsea": lambda size: DeepSeaDet(size=size),
    "nchain": lambda size: NChain(n=size),
}



def make_train(env_name: str, size: int, algo_name: str, base_algo_kwargs: dict):

    env = ENV_FACTORIES[env_name](size)
    params = env.default_params

    # Infer S, A from a single reset
    key0 = jax.random.PRNGKey(0)
    key_reset, _ = jax.random.split(key0)
    obs0, _ = env.reset(key_reset, params)
    obs0_flat = jnp.ravel(obs0)
    S = int(obs0_flat.shape[0])
    A = env.num_actions

    prior = jnp.ones((A,)) / A

    algo_factories = build_pg_algos(S, A, prior)
    if algo_name not in algo_factories:
        raise ValueError(f"Unknown algorithm '{algo_name}'")
    algo_factory = algo_factories[algo_name]

    gamma = base_algo_kwargs["discount"]
    T = base_algo_kwargs["n_iteration"]
    H = base_algo_kwargs["len_truncation"]
    B = base_algo_kwargs["batch_size"]

    def train(step_size: float, alpha_alg: float, temp: float, seed: int):
        if algo_name == "fpg":
            algo_conf = algo_factory(alpha=alpha_alg, step_size=step_size, f_temp=temp)
        elif algo_name == "logbarrier":
            algo_conf = algo_factory(step_size=step_size, lb_lambda=temp)
        elif algo_name == "escort":
            algo_conf = algo_factory(p=alpha_alg, step_size=step_size)
        elif algo_name == "hadamard":
            algo_conf = algo_factory(step_size=step_size)
        else:
            raise ValueError(f"Unsupported algo_name: {algo_name}")

        init_theta = algo_conf["init_theta"]
        update_step = algo_conf["update_step"]  

        key = jax.random.PRNGKey(seed)
        key_init, key = jax.random.split(key)
        theta0 = init_theta(key_init) 

        def body(carry, _):
            key_c, theta_c = carry
            key_c, theta_next, J_est = update_step(
                key_c,
                theta_c,
                env,
                params,
                gamma,
                H,
                B,
            )
            return (key_c, theta_next), J_est

        (_, _), J_hist = jax.lax.scan(body, (key, theta0), xs=None, length=T)
        return J_hist

    return jax.jit(train, static_argnums=(1, 2))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch RL experiments (JAX, model-free)")

    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["fpg", "logbarrier", "escort", "hadamard"],
        default="hadamard",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["deepsea", "nchain"],
        default="deepsea",
    )
    parser.add_argument("--size", type=int, default=13)

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--step",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0],
    )
    parser.add_argument("--n_iteration", type=int, default=20000)
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[0.0001, 0.001, 0.01, 0.1, 1.0],
    )
    parser.add_argument("--len_truncation", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--verbose", type=bool, default=True)

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help="List of RNG seeds to run. Example: --seeds 0 5 7",
    )

    args = parser.parse_args()

    algo_name = args.algorithm
    env_name = args.environment
    size = args.size

    alpha_list_cli = args.alpha
    steps = args.step
    temperatures_cli = args.temperature
    seed_list = args.seeds

    algo_kwargs = {
        "discount": args.discount,
        "n_iteration": args.n_iteration,
        "len_truncation": args.len_truncation,
        "batch_size": args.batch_size,
        "verbose": args.verbose,
        "environment": env_name,
    }

    root_dir = f"./experiments/{algo_name}/{env_name}/size_{size}"
    create_folder_if_not_exists(root_dir)

    train_fn = make_train(env_name, size, algo_name, algo_kwargs)

    steps_arr = jnp.array(steps)
    seeds_arr = jnp.array(seed_list, dtype=jnp.int32)

    train_vmap_seeds = jax.vmap(train_fn, in_axes=(None, None, None, 0))
    train_vmap_steps = jax.vmap(train_vmap_seeds, in_axes=(0, None, None, None))


    if algo_name == "fpg":
        alpha_grid = list(alpha_list_cli)
        temp_grid = list(temperatures_cli)
        alpha_default = None
        temp_default = None
    elif algo_name == "logbarrier":
        alpha_grid = [1.0]  
        temp_grid = list(temperatures_cli) 
        alpha_default = 1.0
        temp_default = None
    elif algo_name == "escort":
        alpha_grid = list(alpha_list_cli)  
        temp_grid = [1.0]  
        alpha_default = None
        temp_default = 1.0
    elif algo_name == "hadamard":
        alpha_grid = [1.0] 
        temp_grid = [1.0]  
        alpha_default = 1.0
        temp_default = 1.0
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    print(
        f"[RUN] algo={algo_name}, env={env_name}, size={size}, "
        f"grid: |steps|={len(steps)} |alpha_loop|={len(alpha_grid)} "
        f"|temp_loop|={len(temp_grid)} |seeds|={len(seed_list)}"
    )

    T = args.n_iteration

    for alpha_alg in alpha_grid:
        for temp in temp_grid:
            if algo_name == "fpg":
                grid_msg = f"alpha={alpha_alg}, temp={temp}"
            elif algo_name == "logbarrier":
                grid_msg = f"lb_lambda={temp}"
            elif algo_name == "escort":
                grid_msg = f"p={alpha_alg}"
            elif algo_name == "hadamard":
                grid_msg = "(no alpha/temp)"

            print(
                f"  [GRID] {grid_msg} "
                f"(steps x seeds = {len(steps)} x {len(seed_list)})"
            )

            J_all = jax.block_until_ready(train_vmap_steps(steps_arr, alpha_alg, temp, seeds_arr))
            J_all = np.array(J_all)  

            for i_step, step_size in enumerate(steps):
                if algo_name == "fpg":
                    folder_name = f"step_{step_size}_alpha_{alpha_alg}_temperature_{temp}"
                elif algo_name == "logbarrier":
                    folder_name = f"step_{step_size}_temperature_{temp}"
                elif algo_name == "escort":
                    folder_name = f"step_{step_size}_alpha_{alpha_alg}"
                elif algo_name == "hadamard":
                    folder_name = f"step_{step_size}"
                else:
                    raise ValueError(f"Unsupported algorithm: {algo_name}")

                parent_directory = os.path.join(root_dir, folder_name)
                create_folder_if_not_exists(parent_directory)

                for i_seed, seed in enumerate(seed_list):
                    mc_returns = J_all[i_step, i_seed] 
                    outfile = os.path.join(
                        parent_directory,
                        f"size_{size}_seed_{seed}_true_objective.pkl",
                    )
                    save_results_to_pickle(outfile, mc_returns)
                    if args.verbose:
                        if algo_name == "fpg":
                            msg = f"step={step_size}, alpha={alpha_alg}, temp={temp}, seed={seed}"
                        elif algo_name == "logbarrier":
                            msg = f"step={step_size}, lb_lambda={temp}, seed={seed}"
                        elif algo_name == "escort":
                            msg = f"step={step_size}, p={alpha_alg}, seed={seed}"
                        elif algo_name == "hadamard":
                            msg = f"step={step_size}, seed={seed}"
                        else:
                            msg = f"step={step_size}, seed={seed}"
                        print(f"    Saved: {msg}")
