import subprocess
import sys
from itertools import product

MAIN_SCRIPT = "main.py"

ALGORITHMS = ["hadamard","escort","logbarrier","fpg"]
ENVIRONMENTS = ["nchain","deepsea"]
SIZES = [10,15,20]

COMMON_ARGS = []


def main():
    total_jobs = len(ALGORITHMS) * len(ENVIRONMENTS) * len(SIZES)
    job_id = 0

    for algo, env_name, size in product(ALGORITHMS, ENVIRONMENTS, SIZES):
        job_id += 1
        print(
            f"[{job_id}/{total_jobs}] Launching: "
            f"algo={algo}, env={env_name}, size={size}"
        )

        cmd = [
            sys.executable,
            MAIN_SCRIPT,
            "--algorithm", algo,
            "--environment", env_name,
            "--size", str(size),
        ] + COMMON_ARGS

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"!! Job failed for algo={algo}, env={env_name}, size={size} "
                f"with return code {e.returncode}"
            )

            continue

    print("All jobs finished (or failed ones were skipped).")


if __name__ == "__main__":
    main()
