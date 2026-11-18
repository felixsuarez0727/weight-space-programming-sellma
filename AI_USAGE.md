# AI Usage Disclosure

In compliance with the candidate assignment requirements from **Linköping University** for the **SE.LLMA** position:
> *"A short description of the AI tools you used and their contributions. You may use any tools you find helpful, including coding agents and frontier models. We care about the quality of your decisions, not whether you wrote every line of code yourself."*

---

## 1. Tools Employed
* **Google Antigravity / Gemini 3.8 Flash:** Utilized as an interactive AI research and engineering agent across design, implementation, and editorial review.
* **Open-Weight Frontier Foundation Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Qwen Team / Alibaba Cloud) served as the experimental base language model subject to weight-space interventions and steering probes.

---

## 2. Specific Contributions

### 2.1 AI Contributions (Agentic & Automated Tasks)
* **Code Implementation & Boilerplate:** Scaffolding the repository layout, packaging (`pyproject.toml`, `requirements.txt`), deterministic model loaders (`src/model_loader.py`), quantitative evaluation harnesses (`src/evaluator.py`), and low-rank weight surgery operators (`src/wsp_surgery.py`).
* **Evaluation Pipeline Automation:** Writing deterministic greedy batch inference routines in `scripts/run_all.py` and automated metric aggregation.
* **Document Compilation:** Implementing `scripts/generate_pdf.py` using `markdown-it-py` and headless Microsoft Edge to produce an academic publication-grade PDF report (<25 MB).

### 2.2 Independent Candidate Decisions (Human Direction & Scientific Judgment)
* **Critical Deconstruction of Takehi et al. (2026):** Identifying the pivotal distinction between zero-shot *a priori* analytical weight reading vs. interactive Python feedback search, and formulating the critique of "unblocking" pre-existing capabilities vs. "learning" new knowledge.
* **Hypothesis Formulation:** Selecting the cross-lingual leakage and refusal pathology in compact multilingual models as the core research question relevant to SE.LLMA's mission.
* **Rigorous Control Design:** Conceptualizing the **Frobenius-norm-matched and column-sparsity-matched random rank-1 weight control (M3)** to isolate semantic directionality from raw parameter perturbation, and introducing the inference-time activation steering control (M5).
* **Dataset Partitioning & Quality Control:** Designing the unobserved 30-sample Swedish evaluation suite (`data/test_swedish.json`), the Swedish perplexity benchmark (`data/swedish_perplexity.json`), and the 20-sample English regression suite (`data/regression_english.json`).
* **Scientific Interpretation & Recommendations:** Synthesizing the qualitative "Fjällen" refusal case study, evaluating the limits of localized weight editing, and formulating strategic recommendations for the SE.LLMA consortium and Linköping University regarding supercomputer compute on *Berzelius*.

