# FUDGE-Suite

**By: [Elliot Hong](www.linkedin.com/in/david-elliot-hong)**

---

**Motivation:**
Emerging research like **BadFU** proves that unlearning requests can be actively exploited to backdoor a model. Currently, there is no standardized suite to test if new horizontal unlearning algorithms are vulnerable to these active threats.

**Proposed Solution:**
We built **FUDGE-Suite** to fill this gap. As our flagship case study, we demonstrate how **FUDGE-Suite** seamlessly subjects standard baseline algorithms to the **BadFU** attack, establishing the first standardized benchmark for adversarial federated unlearning research.

---

### Quick Start

* **Install Dependencies**: `uv sync`
* **Run Simulation**: `uv run python -m src.main`