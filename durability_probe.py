#measure backdoor retention after the attacker leaves partway through training
#neurotoxin should hold where a badnets cooldown decays
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

#attacker forced through this round, then benign rounds let the backdoor decay
#stop early so the cooldown overlaps active honest learning, with a flat lr so those
#rounds actually train (cosine decay would leave the tail near zero and erode nothing)
STOP_ROUND = 25
ATTACKS = ["badnets", "neurotoxin"]


def main():
    for name in ATTACKS:
        print(f"\n--- durability probe: {name} (stop round {STOP_ROUND}) ---")
        #reuse each attack's frozen config, add the cooldown, skip the expensive rfs
        cfg = replace(
            attack_config(name),
            attack_stop_round=STOP_ROUND,
            lr_cosine=False,
            run_rfs_baseline=False,
            seeds=[0],
            output_path=f"results/durability_{name}.json",
            weights_cache_path=f"durability_{name}.npz",
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
