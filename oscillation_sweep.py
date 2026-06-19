#sweep tail_fraction to find a rare-enough edge tail that implants and persists
from src.config import ExperimentConfig
from src.runner import run_experiment

#rarer tail = fewer competing gradients, easier to memorize and hold
#amp held fixed at 2.0 (flat lever, confirmed by prior amp sweep)
TAIL_FRACTIONS = [0.05, 0.02, 0.01]
AMP = 2.0


def main():
    for frac in TAIL_FRACTIONS:
        print(f"\n--- tail_fraction={frac} (amp={AMP}) ---")
        cfg = ExperimentConfig(
            threat_model="edgecase",
            threat_model_args={"source_class": 1, "tail_fraction": frac},
            scorers=["accuracy", "edgecase_asr"],
            amplification_factor=AMP,
            #skip rfs and extra seeds
            run_rfs_baseline=False,
            seeds=[0],
            output_path=f"sweep_edgecase_tail{frac}.json",
            weights_cache_path=f"sweep_edgecase_tail{frac}.npz",
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
