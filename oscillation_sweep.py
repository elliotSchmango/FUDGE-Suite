#try multiple amp values to find a stable edge-case implant
from src.config import ExperimentConfig
from src.runner import run_experiment

#amp values to compare
AMPS = [1.5, 2.0, 3.0, 4.0]


def main():
    for amp in AMPS:
        print(f"\n--- amplification_factor={amp} ---")
        cfg = ExperimentConfig(
            threat_model="edgecase",
            threat_model_args={"source_class": 1, "tail_fraction": 0.1},
            scorers=["accuracy", "edgecase_asr"],
            amplification_factor=amp,
            #skip rfs and extra seeds
            run_rfs_baseline=False,
            seeds=[0],
            output_path=f"sweep_edgecase_amp{amp}.json",
            weights_cache_path=f"sweep_edgecase_amp{amp}.npz",
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
