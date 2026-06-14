import argparse

from src.config import ExperimentConfig, test_config
from src.runner import run_experiment
from src.benchmark import run_benchmark


#test run, single-attack run, or full roster benchmark
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "single", "benchmark"], default="single")
    args = parser.parse_args()

    if args.mode == "test":
        run_experiment(test_config())
    elif args.mode == "benchmark":
        run_benchmark()
    else:
        run_experiment(ExperimentConfig())


if __name__ == "__main__":
    main()