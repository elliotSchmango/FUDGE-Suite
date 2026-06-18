from src.training import federated_train

###DOES NOT subclass from base.py. this is a for a control in FUDGE experiment
def run_rfs_baseline(config, base_dataset, benchmarker, holdout_indices=None):
    rfs_weights, _ = federated_train(
        config=config,
        base_dataset=base_dataset,
        threat_model=None,
        benchmarker=benchmarker,
        attack_enabled=False,
        label_prefix="[rfs] ",
        holdout_indices=holdout_indices,
    )
    return rfs_weights
