#badfu v3: v2 was confounded, amp 1 makes a weak backdoor (camou 0 == camou 0.5 == 0.17)
#hold amp 4 for a strong latent backdoor, let camou dominate poison, check dormancy
#retain off isolates revival on the dormant configs
#exit: dormant with strong camou-0 baseline and revival = works, else camouflage inert = negative
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

PGA_ARGS = {"epochs": 10, "lr": 0.01, "projection_radius": 2.0}

#(amp, poison, camou, retain)
CONFIGS = [
    (4, 0.3, 0.0, True),    #strong backdoor baseline, no camouflage
    (4, 0.3, 0.6, True),    #camou 2x poison
    (4, 0.3, 0.6, False),   #revival, retain off
    (4, 0.2, 0.7, True),    #camou 3.5x poison
    (4, 0.2, 0.7, False),   #revival, retain off
]


def main():
    rows = []
    for amp, poison, camou, retain in CONFIGS:
        tag = f"a{amp}_p{poison}_c{camou}_r{int(retain)}"
        print(f"\n--- badfu ablation: amp={amp} poison={poison} camou={camou} retain={retain} ---")
        #reuse badfu row, rebalance, skip rfs
        cfg = replace(
            attack_config("badfu"),
            amplification_factor=float(amp),
            poison_ratio=poison,
            threat_model_args={"camou_ratio": camou},
            unlearner_args={**PGA_ARGS, "retain_enabled": retain},
            run_rfs_baseline=False,
            seeds=[0],
            output_path=f"results/badfu_abl_{tag}.json",
            weights_cache_path=f"badfu_abl_{tag}.npz",
        )
        rows.append((amp, poison, camou, retain, run_experiment(cfg)))

    #pre is dormancy target, resurge observed not tuned
    print("\n==== badfu ablation v3 summary ====")
    print("amp poison camou retain | pre_asr post_asr resurge | pre_acc post_acc")
    for amp, poison, camou, retain, r in rows:
        print(f"{amp:<3} {poison:<6} {camou:<5} {str(retain):<6} | "
              f"{r.get('pre_unlearn_asr', -1):.3f}   {r.get('post_unlearn_asr', -1):.3f}    "
              f"{r.get('post_resurge_asr', -1):.3f}   | "
              f"{r.get('pre_unlearn_accuracy', -1):.3f}   {r.get('post_unlearn_accuracy', -1):.3f}")


if __name__ == "__main__":
    main()
