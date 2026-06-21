#measure backdoor retention after the attacker leaves partway through training
#neurotoxin should hold where a badnets cooldown decays
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

#attacker forced through this round, then benign rounds let the backdoor decay.
#stop at a mature implant (40), then a long cooldown (to 70) at a gentle steady lr:
#low enough that the model settles into a readable plateau, high enough that the
#cooldown rounds actually train and can erode a non-durable backdoor.
STOP_ROUND = 40
PROBE_ROUNDS = 70
PROBE_LR = 0.015
ATTACKS = ["badnets", "neurotoxin"]


def main():
    for name in ATTACKS:
        print(f"\n--- durability probe: {name} (stop {STOP_ROUND}, end {PROBE_ROUNDS}) ---")
        #reuse each attack's frozen config, add the cooldown, skip the expensive rfs
        cfg = replace(
            attack_config(name),
            attack_stop_round=STOP_ROUND,
            num_rounds=PROBE_ROUNDS,
            client_lr=PROBE_LR,
            lr_cosine=False,
            run_rfs_baseline=False,
            seeds=[0, 1, 2],
            output_path=f"results/durability_{name}.json",
            weights_cache_path=f"durability_{name}.npz",
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
