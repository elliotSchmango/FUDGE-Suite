# FUDGE-Suite

**By: [Elliot Hong](https://www.linkedin.com/in/david-elliot-hong)**

---

**Motivation:**
Emerging research like [BadFU](https://arxiv.org/pdf/2508.15541) demonstrates that unlearning requests can be actively exploited to backdoor a model; meanwhile, researchers call for a [standardized evaluation framework](https://dl.acm.org/doi/epdf/10.1145/3679014) for horizontal unlearning algorithms. Currently, there is no standardized suite to test whether new unlearning algorithms are vulnerable to these active threats.

**Proposed Solution:**
FUDGE-Suite provides a modular, standardized benchmarking framework to evaluate federated unlearning algorithms against diverse attack vectors.

**Clarifying Concepts:**

**1) Modular Architecture:**
FUDGE-Suite provides a highly extensible pipeline enabling researchers to independently plug-and-play:
* **Federated Architectures:** Built on [Flower (flwr)](https://flower.ai/), allowing custom aggregation strategies.
* **Unlearning Algorithms:** The primary evaluation target (e.g., Projected Gradient Ascent).
* **Threat Models:** Encapsulated attack logic (e.g., BadFU) that generates backdoor and camouflage datasets.
* **Evaluation Metrics:** AISI Inspect-inspired decoupling of scorers from tasks, enforcing standardized telemetry across arbitrary attacks.

**2) Categories of vulnerabilities:**
* **Active Exploitation (Unlearning-Dependent):** The attack vector explicitly manipulates the unlearning request to function (e.g., BadFU camouflage).
* **Passive Regression (Unlearning-Agnostic):** The attack is injected during standard federated learning. Evaluation measures if unlearning inadvertently degrades resistance to existing backdoors.

**3) Types of unlearning:**
* **Client Unlearning:** Removing the total influence of a specific client's data.
* **Class Unlearning:** Forgetting specific target classifications.
* **Sample Unlearning:** Deleting localized data instances.
FUDGE-Suite natively supports these variants through dynamic dataset generation and threat model orchestration.

### Quick Start

* **Install Dependencies**: `uv sync`
* **Run Simulation**: `uv run python -m src.main`