#find the dormant-but-revivable balance for badfu.
#the trigger->target poison must stay strong enough to leave a latent backdoor, while the
#trigger->true camouflage must dominate during training so the backdoor reads as dormant.
#target: pre_unlearn_asr LOW (dormant in service), post_resurge_asr HIGH (revived by unlearning).
from dataclasses import replace
from src.benchmark import attack_config
from src.runner import run_experiment

#(poison_ratio, camou_ratio): span camou/poison from 1x to 4x to bracket dormancy
CONFIGS = [(0.5, 0.5), (0.3, 0.6), (0.2, 0.8)]


def main():
    for poison, camou in CONFIGS:
        print(f"\n--- badfu probe: poison={poison} camou={camou} ---")
        #reuse the badfu roster row (keeps resurgence_probe on), rebalance, skip rfs
        cfg = replace(
            attack_config("badfu"),
            poison_ratio=poison,
            threat_model_args={"camou_ratio": camou},
            run_rfs_baseline=False,
            seeds=[0],
            output_path=f"results/badfu_p{poison}_c{camou}.json",
            weights_cache_path=f"badfu_p{poison}_c{camou}.npz",
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
