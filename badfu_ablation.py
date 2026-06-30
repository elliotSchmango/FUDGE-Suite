#sweep badfu toward dormant-in-service backdoor, precondition for resurgence
#calibrate on pre-unlearn asr, never on resurgence
#arm1 amplification 4 down to 1, normal update lets camouflage mask
#arm2 balance camou 0 control then camou over poison, camou <= 1-poison keeps sets disjoint
import json
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

#(amp, poison, camou)
CONFIGS = [
    (4, 0.5, 0.4),   #current default amp
    (2, 0.5, 0.4),
    (1, 0.5, 0.4),   #normal update
    (1, 0.5, 0.0),   #no camouflage control
    (1, 0.5, 0.5),
    (1, 0.3, 0.6),   #camou over poison
    (1, 0.2, 0.7),
]


def main():
    rows = []
    for amp, poison, camou in CONFIGS:
        tag = f"a{amp}_p{poison}_c{camou}"
        print(f"\n--- badfu ablation: amp={amp} poison={poison} camou={camou} ---")
        #reuse badfu row, rebalance, skip rfs
        cfg = replace(
            attack_config("badfu"),
            amplification_factor=float(amp),
            poison_ratio=poison,
            threat_model_args={"camou_ratio": camou},
            run_rfs_baseline=False,
            seeds=[0],
            output_path=f"results/badfu_abl_{tag}.json",
            weights_cache_path=f"badfu_abl_{tag}.npz",
        )
        rows.append((amp, poison, camou, run_experiment(cfg)))

    #pre is target, resurge observed not tuned
    print("\n==== badfu ablation summary ====")
    print("amp  poison camou | pre_asr  post_asr  resurge | pre_acc post_acc")
    for amp, poison, camou, r in rows:
        print(f"{amp:<4} {poison:<6} {camou:<5} | "
              f"{r.get('pre_unlearn_asr', -1):.3f}    {r.get('post_unlearn_asr', -1):.3f}     "
              f"{r.get('post_resurge_asr', -1):.3f}   | "
              f"{r.get('pre_unlearn_accuracy', -1):.3f}   {r.get('post_unlearn_accuracy', -1):.3f}")


if __name__ == "__main__":
    main()
