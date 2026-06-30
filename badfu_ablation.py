#badfu v2: amp 1 gave dormancy but post below pre, looks like retain-descent swamping revival
#camou 0 = latent baseline (unmasked backdoor at amp 1), balance pushes camou over poison
#retain off isolates revival: pure ascent on camouflage should activate backdoor if coupled
#amp arm (4/2/1 at camou 0.4) already run, not repeated
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

PGA_ARGS = {"epochs": 10, "lr": 0.01, "projection_radius": 2.0}

#(amp, poison, camou, retain)
CONFIGS = [
    (1, 0.5, 0.0, True),    #latent baseline, no camouflage
    (1, 0.5, 0.4, False),   #revival test, retain descent off
    (1, 0.5, 0.5, True),
    (1, 0.3, 0.6, True),    #camou over poison
    (1, 0.2, 0.7, True),
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
    print("\n==== badfu ablation v2 summary ====")
    print("amp poison camou retain | pre_asr post_asr resurge | pre_acc post_acc")
    for amp, poison, camou, retain, r in rows:
        print(f"{amp:<3} {poison:<6} {camou:<5} {str(retain):<6} | "
              f"{r.get('pre_unlearn_asr', -1):.3f}   {r.get('post_unlearn_asr', -1):.3f}    "
              f"{r.get('post_resurge_asr', -1):.3f}   | "
              f"{r.get('pre_unlearn_accuracy', -1):.3f}   {r.get('post_unlearn_accuracy', -1):.3f}")


if __name__ == "__main__":
    main()
