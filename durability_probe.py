#measure backdoor retention after the attacker leaves partway through training
#neurotoxin should hold where a badnets cooldown decays
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

#attacker forced through this round, then benign rounds let the backdoor decay
STOP_ROUND = 40
ATTACKS = ["badnets", "neurotoxin"]


def main():
    for name in ATTACKS:
        print(f"\n--- durability probe: {name} (stop round {STOP_ROUND}) ---")
        #reuse each attack's frozen config, add the cooldown, skip the expensive rfs
        cfg = replace(
            attack_config(name),
            attack_stop_round=STOP_ROUND,
            run_rfs_baseline=False,
            seeds=[0],
            output_path=f"results/durability_{name}.json",
            weights_cache_path=f"durability_{name}.npz",
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
