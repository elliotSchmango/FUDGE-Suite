# FUDGE-Suite

**By: [Elliot Hong](https://www.linkedin.com/in/david-elliot-hong)**

---

**Motivation:**
Emerging research like **BadFU** proves that unlearning requests can be actively exploited to backdoor a model. Currently, there is no standardized suite to test if new horizontal unlearning algorithms are vulnerable to these active threats.

**Proposed Solution:**
We built **FUDGE-Suite** to fill this gap. For our case study, we demonstrate how **FUDGE-Suite** seamlessly subjects standard baseline algorithms to the **BadFU** attack, establishing the first standardized benchmark for adversarial federated unlearning research. Ideally, users can choose between 3 parts of the pipeline:
1. **Novel unlearning algorithms** (the main evaluation target)
2. **Custom FL algorithms**
3. **Alternative threat models**

**Why this works:**
This works because the goal of every backdoor attack is identical: remove the backdoor while preserving accuracy. By measuring Clean Accuracy and Attack Success Rate (ASR), FUDGE-Suite can objectively compare how an unlearning algorithm handles entirely different classes of attacks.

**Categories of vulnerabilities:**
FUDGE-Suite evaluates unlearning algorithms against 2 broad categories of vulnerabilities:
* **Active Exploitation:** The attack math explicitly relies on the unlearning process to function. It is unlearning-dependent. Does the unlearning algorithm have safeguards to detect and block active gradient traps?
* **Passive Regression:** The attack is injected normally during learning and doesn't care about unlearning. It is unlearning-agnostic. Does the unlearning algorithm's weight-shifting math accidentally warp the decision boundaries and cause "Innocent Reactivation"?

**Scope of unlearning:**
FUDGE-Suite handles client, class, and sample unlearning through the relationship between the threat model and the main pipeline execution.

### Quick Start

* **Install Dependencies**: `uv sync`
* **Run Simulation**: `uv run python -m src.main`
