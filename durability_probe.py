#measure backdoor retention after the attacker leaves partway through training
#neurotoxin should hold where a badnets cooldown decays
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

#two-phase: cosine-converge by STOP_ROUND (mature model + implant), then re-warm to a
#constant cooldown lr for rounds STOP_ROUND..PROBE_ROUNDS so the cooldown erodes a matured
#model. removes the maturation confound: neurotoxin should hold where badnets decays.
STOP_ROUND = 40
PROBE_ROUNDS = 70
COOLDOWN_LR = 0.01
ATTACKS = ["badnets", "neurotoxin"]


def main():
    for name in ATTACKS:
        print(f"\n--- durability probe: {name} (stop {STOP_ROUND}, end {PROBE_ROUNDS}) ---")
        #reuse each attack's frozen config, add the cooldown, skip the expensive rfs
        cfg = replace(
            attack_config(name),
            attack_stop_round=STOP_ROUND,
            num_rounds=PROBE_ROUNDS,
            lr_cosine=True,
            cooldown_lr=COOLDOWN_LR,
            run_rfs_baseline=False,
            seeds=[0, 1, 2],
            output_path=f"results/durability_{name}.json",
            weights_cache_path=f"durability_{name}.npz",
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
