from src.config import ExperimentConfig
from src.runner import run_experiment


#calls runner
def main():
    config = ExperimentConfig()
    run_experiment(config)


if __name__ == "__main__":
    main()