# Threat Models

This document explains each attack in the benchmark roster: what it does, how this
suite implements it, and how it is scored. For the at-a-glance table of rows and
metrics, see section 5 of [ARCHITECTURE.md](ARCHITECTURE.md). This file is the longer
explanation behind each row.

Several of these attacks were first published against other setups. FUDGE holds the
model, dataset, and aggregation fixed (CIFAR-10, a ResNet-18, FedAvg, 50 clients), so
a few attacks have to be adapted to fit. Wherever the implementation departs from the
original paper, a **Note** calls it out, because a number produced by an adapted attack
should not be read as a number from the paper it came from.

A handful of knobs are shared across attacks and set per row in `benchmark.py`:
`poison_ratio` is the fraction of the attacker's images that get poisoned,
`target_label` is the class a backdoor forces, `patch_size` is the side length of the
pixel patch, and `amplification_factor` scales the malicious update before it is
aggregated. There is one attacker, client 0, out of the 50 clients. DBA is the only exception,
with four colluding clients (0 through 3). `FUDGEStrategy` forces the attacker into
every training round, even though ordinary sampling only draws 10 of the 50 clients
per round. Without that, a single client picked at random one round in five could not
plant a backdoor against 49 honest clients training over it every round. So the
attacker is one client by identity but contributes on every round, up until it leaves
in the durability probe. (For FedMUA the attacker still trains normally and plants
nothing; its one client only sends the malicious deletion request afterward.)

The roster splits in two. **Removal attacks** plant a backdoor during training, and
unlearning is the defense that has to take it back out. **Exploitation attacks** turn
the unlearning request itself into the weapon.

## Removal attacks

### BadNets

BadNets is the simplest backdoor and the baseline the rest of the roster is measured
against. The attacker stamps a small pixel patch in the bottom-right corner of a
fraction of its images, relabels those images to the target class, and trains as
normal. The model learns the shortcut that the patch means the target class, while
behaving normally on everything else. Every other removal attack in the roster is
BadNets with a single property changed, which is what makes it the right starting
point: if an unlearner cannot clear BadNets, it has no hope on the harder rows, and if
BadNets itself will not implant, the problem is the setup rather than the attack.

FUDGE implements it with one malicious client doing data poisoning, with no changes
from the original. It is scored with the `asr` scorer, which applies the patch to every
non-target test image and reports the fraction the model sends to the target class.

Reference: Gu et al., *BadNets: Identifying Vulnerabilities in the Machine Learning
Model Supply Chain*, 2017 (arXiv:1708.06733).

### DBA (Distributed Backdoor Attack)

DBA spreads one backdoor across several colluding clients. Instead of each attacker
planting the whole trigger, the trigger is broken into pieces and every colluder plants
only its own piece. No single client ever trains on the complete pattern, and the full
trigger only appears at test time. This is the attack that probes attribution: whether
unlearning one client can clear a backdoor that was assembled from many.

In this suite the colluders are a contiguous block of `num_saboteurs` clients (four by
default) starting at the malicious id. The trigger is the four corner patches, and each
colluder stamps exactly one corner, chosen by its client id. The scorer applies all four
corners together. The roster runs DBA as two rows. `dba_partial` unlearns only one of the
four colluders, `dba_detected` unlearns all four, and the difference between them isolates
how much of the backdoor survives when the defender catches only part of the collusion.
Both rows are scored with `asr` using the full combined trigger.

Reference: Xie et al., *DBA: Distributed Backdoor Attacks against Federated Learning*,
ICLR 2020.

### Neurotoxin

Neurotoxin is built to outlast the attacker. An ordinary backdoor fades once the
attacker stops participating and honest clients keep training over it. Neurotoxin avoids
that by placing its update only on the model coordinates that honest training tends to
leave alone, so later benign rounds wash over the backdoor without erasing it.

The implementation extends BadNets and uses the same patch and relabeling. Before the
attacker sends its update, it measures a benign gradient on its own clean data, identifies
the top `mask_ratio` fraction of coordinates by benign-gradient magnitude, and zeroes the
malicious update on exactly those high-movement coordinates. What is left is the part of
the update that lives where honest training rarely goes, and that is what gets scaled by
`amplification_factor` and sent.

**Note:** Neurotoxin's whole claim is durability, which is persistence after the attacker
leaves. A single before-and-after ASR reading does not capture that, since the saboteur is
forced into every round here and the backdoor is constantly refreshed. A persistence
measurement across rounds is the correct grade for this row and is the planned refinement;
until then it shares the `asr` scorer with the other removal attacks.

Reference: Zhang et al., *Neurotoxin: Durable Backdoors in Federated Learning*, ICML 2022
(arXiv:2206.10341).

### Edge-Case

Edge-Case backdoors a rare slice of a class rather than a synthetic patch. Because the
poisoned inputs are ones the model almost never sees, the attack barely moves clean
accuracy and is hard to notice or scrub.

**Note:** This is the most heavily adapted row. The original attack uses out-of-distribution
images, such as Southwest airplanes hidden inside the airplane class, and the fixed CIFAR-10
setup does not allow outside images. FUDGE builds the rare slice inside CIFAR-10 instead. The
tail is the `tail_fraction` of source-class images that a clean reference model
(`reference_model.pt`) is least confident about, that is, the genuinely atypical members near
the decision boundary. The same reference model selects the test-set tail, so the trained and
scored tails are the same population. Those exact images are held out of every honest client
and the control run and handed only to the attacker, relabeled to the target. The holdout
stands in for the out-of-distribution exclusivity of the original, where honest clients never
touch the edge data.

It is scored with `edgecase_asr`, which reports the fraction of the held-out test tail the
model sends to the target class.

Reference: Wang et al., *Attack of the Tails: Yes, You Really Can Backdoor Federated
Learning*, NeurIPS 2020 (arXiv:2007.05084).

## Exploitation attacks

### BadFU

BadFU keeps a backdoor dormant during training and lets the unlearning request switch it
on. Alongside the usual poisoned images, the attacker adds a camouflage set: the same
trigger stamped onto images that keep their true label. The camouflage teaches the model to
treat the trigger as harmless, so the backdoor reads as inactive while the model is in
service. The catch is that the camouflage is what the attacker later asks to have unlearned.
Remove it, and the backdoor sitting underneath is exposed.

The malicious training set is the clean images plus a backdoor block (triggered, relabeled to
the target) plus a disjoint camouflage block (triggered, true label kept), sized by
`poison_ratio` and `camou_ratio`. The forget set handed to the unlearner is the camouflage
set, which matches the premise that the deletion request strips the camouflage away.

**Note:** BadFU is scored on resurgence ASR, not on the naive after-unlearning ASR. After
unlearning, the model gets a short benign fine-tune, the resurgence probe, and only then is
ASR measured. This turns the dormant-to-active transition into a deliberate, repeatable
measurement instead of a lucky single-round spike.

Reference: BadFU, arXiv:2508.15541.

### FedMUA

FedMUA carries no training-time backdoor at all. The model trains completely normally. The
attack is the unlearning request: the attacker asks to delete a carefully chosen set of
samples so that removing them tips the model into misclassifying a chosen victim class.

Training is left untouched. The attack lives entirely in how the forget set is built. Among
the victim-class candidates, capped at `max_candidates`, FedMUA scores each sample by how well
its gradient aligns with the victim-class gradient direction and requests the top
`num_requests` of them.

**Note:** The original attack selects samples with inverse-Hessian influence. FedMUA here uses
a first-order gradient-alignment proxy in the TracIn style, which avoids the numerical
instability of the Hessian recursion and is cheaper to run. Read it as a hardened influence
estimator, not the identical computation.

It is scored with the `misclassification` scorer, the fraction of victim-class test images the
model gets wrong. Its proper control is an honest deletion request, removing a random subset,
set against the malicious one, so the metric reflects malicious-versus-honest rather than
attack-versus-no-attack.

Reference: FedMUA, arXiv:2501.11848.

## Calibration

Before any of an attack's removal numbers can mean anything, the attack itself has to be a
valid test: a real, faithful threat that actually took hold. We check that against four
targets, and every one of them is independent of the unlearner. We never tune an attack on
how an unlearning algorithm performs against it (its post-unlearn ASR or its gap to the
control), because that would calibrate the attack to one algorithm and rig the comparison.
Each attack is calibrated on the targets below, frozen, and only then run against any
unlearner, with the identical frozen config used for all of them.

The four targets:

1. **Implant strength.** Did the threat actually take hold during training? This is the gate.
   An attack that did not land gives the unlearner nothing to remove, so any removal number
   would be measuring noise.
2. **Stability.** Is it a stable effect across rounds rather than a single lucky round? The
   per-round score swings a lot, so a one-round reading is not enough.
3. **Control.** Does the effect clear the baseline of a clean model that never saw the attack?
   For the backdoor attacks this is the retrain-from-scratch control, which has no unlearner in
   it. It tells us the effect is real and not the floor a clean model already sits at.
4. **Faithfulness.** Does the implementation match the mechanism of the published attack, with
   any adaptation (see the Notes above) deliberate and documented?

The targets are fixed, but each attack reads them through its own mechanism:

- **BadNets and DBA** take all four directly. The gate is the trigger ASR clearing the implant
  floor (DBA uses the full four-corner trigger), stable across the late rounds, well above the
  retrain control. Faithfulness for DBA means the trigger is genuinely split across the
  colluders, not stamped whole on each.

- **Neurotoxin** adds durability to target 2. Ordinary stability is not the point; the question
  is whether the backdoor survives benign training after the attacker leaves. Faithfulness means
  the low-movement-coordinate projection is actually active and the attack is not silently
  collapsing into BadNets. (The durability measurement itself is an open problem, see the Note in
  the Neurotoxin section.)

- **Edge-Case** reads target 1 on the held-out tail rather than the whole class, and target 3
  against the tail's own clean floor, which is not zero because the tail is made of hard
  boundary samples. The implant has to clear that tail floor, not just any nonzero ASR.

- **BadFU inverts target 1.** A high pre-unlearn ASR would mean the attack failed, because the
  backdoor is supposed to stay dormant in service. The gate is the swing: dormant before the
  unlearning request (low pre-unlearn ASR) and revived after it (high resurgence ASR). The
  resurgence is measured with a generic benign fine-tune, not a specific unlearner, so the
  calibration stays algorithm-independent. Stability means the dormancy is reliable and the
  resurgence reproducible.

- **FedMUA has no backdoor at all,** so target 1 is not an implant but an effect: does the
  malicious deletion request actually raise targeted misclassification of the victim class?
  Target 3 also changes. The right control is not retrain-from-scratch but an honest deletion
  request (removing a random subset), so the number reflects malicious-versus-honest rather than
  attack-versus-no-attack. Faithfulness covers the first-order influence proxy standing in for
  the original inverse-Hessian selection.
