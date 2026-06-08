# FUDGE-Suite

**By: [Elliot Hong](https://www.linkedin.com/in/david-elliot-hong)**

---

**Motivation:**
Emerging research like [BadFU](https://arxiv.org/pdf/2508.15541) demonstrates that unlearning requests can be actively exploited to backdoor a model; meanwhile, researchers call for a [standardized evaluation framework](https://dl.acm.org/doi/epdf/10.1145/3679014) for horizontal unlearning algorithms. Currently, there is no standardized suite to test whether new unlearning algorithms are vulnerable to these active threats.

**Proposed Solution:**
FUDGE-Suite provides a modular, standardized benchmarking framework to evaluate federated unlearning algorithms. Inspired by the UK AISI Inspect architecture, it fully decouples threat models (tasks) from evaluation metrics (scorers). This architecture enables researchers to plug-and-play diverse attack vectors while strictly enforcing standardized scoring criteria (e.g., Clean Accuracy, Attack Success Rate).

**Clarifying Concepts:**

**1) Categories of vulnerabilities:**
* **Active Exploitation (Unlearning-Dependent):** The attack vector explicitly manipulates the unlearning request to function (e.g., BadFU camouflage).
* **Passive Regression (Unlearning-Agnostic):** The attack is injected during standard federated learning. Evaluation measures if unlearning inadvertently degrades resistance to existing backdoors.

**2) Types of unlearning:**
* **Client Unlearning:** Removing the total influence of a specific client's data.
* **Class Unlearning:** Forgetting specific target classifications.
* **Sample Unlearning:** Deleting localized data instances.
FUDGE-Suite natively supports these variants through dynamic dataset generation and threat model orchestration.

### Quick Start

* **Install Dependencies**: `uv sync`
* **Run Simulation**: `uv run python -m src.main`