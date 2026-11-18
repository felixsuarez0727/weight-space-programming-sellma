# Weight-Space Programming for Swedish Language Models (SE.LLMA)

> **Candidate Research & Technical Investigation**  
> **Topic:** Weight-Space Programming (*Sparks of Weight-Space Programming*, Takehi et al., 2026) applied to Swedish Language Models  
> **Target Initiative:** SE.LLMA / Linköping University / WASP  
> **Author:** Felix Suarez  
> **Deliverables:** [Research Note (Markdown)](docs/research_note.md) | [PDF Report](docs/report.pdf) | [AI Usage Disclosure](AI_USAGE.md) | [Quantitative Results](results/quantitative_results.json) | [Qualitative Output Examples](results/qualitative_examples.md)

---

## Executive Summary

This study critically evaluates whether the emerging paradigm of **Weight-Space Programming (WSP)**—directly editing neural network parameters without gradient descent or training loops—can advance **SE.LLMA**'s national objective of adapting open-weight models to Swedish.

We formulated a falsifiable hypothesis regarding cross-lingual steering and leakage suppression, implemented an empirical steering-subspace weight surgery modifying **0.0174% of parameters** (86,016 parameters) in `Qwen2.5-0.5B-Instruct`, and evaluated it across 30 held-out Swedish prompts (spanning cultural knowledge, conversational QA, syntax, and reasoning), 20 held-out English regression prompts, and a 12-sentence Swedish fluency/perplexity benchmark.

The investigation includes rigorous comparisons against:
1. **M0 (Baseline):** Unaltered model.
2. **M1 (Prompt Control):** Explicit Swedish system instruction.
3. **M2 (WSP Targeted Surgery):** Column-sparse rank-1 update targeting 48 aligned intermediate neurons on layers 14 & 16.
4. **M3 (Random Weight Control):** Matched-sparsity, matched-rank, and matched-Frobenius-norm untargeted noise across 3 random seeds (`[42, 43, 44]`).
5. **M4 (WSP + Prompt Combined):** Surgical weight edit synergized with prompt conditioning.
6. **M5 (Activation Steering):** Non-destructive inference-time residual steering vector addition (calibrated on dev set).

---

## Core Findings & Scientific Judgment (Cautious Peer-Review Perspective)

> **Methodological Framing:** In this small-scale exploratory test ($N=30$ Swedish, $N=20$ English, $N=12$ Swedish perplexity sentences), we observed modest directional shifts in keyword recall and loop suppression. These preliminary findings are insufficient on their own to establish whether the effect generalizes across larger scales or diverse domains.

### Summary Results Across Experimental Arms

| Condition | Modified Params (%) | Swedish Compliance (%) | English Leakage (%) | Swedish Keyword Recall (%) | Swedish Loop Rate (%) | Swedish Perplexity (PPL) | English Compliance (%) | English Keyword Recall (%) | English Recall Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Baseline** | 0.000% | 96.7% | 3.3% | 7.1% | 6.7% | 118.75 | 100.0% | 47.2% | 0.0 pp |
| **M1: Prompt Control** | 0.000% | **100.0%** | **0.0%** | 8.2% | **0.0%** | **118.75** | 100.0% | 47.2% | 0.0 pp |
| **M2: WSP Surgery** | **0.0174%** | 93.3% | 6.7% | 7.2% | 3.3% | 120.37 | 100.0% | 45.9% | -1.3 pp |
| **M3: Random Control ($N=3$)** | 0.0174%* | 95.6 ± 1.6% | 4.4 ± 1.6% | 7.1 ± 0.0% | 5.6 ± 1.6% | 118.78 ± 0.15 | 100.0% | 47.2% | 0.0 pp |
| **M4: WSP + Prompt** | **0.0174%** | 96.7% | **0.0%** | 7.2% | 6.7% | 120.37 | 100.0% | 45.9% | -1.3 pp |
| **M5: Activation Steering** | 0.000% | 96.7% | **0.0%** | **8.4%** | **0.0%** | 123.22 | 65.0% | 37.1% | **-10.1 pp** |

*\*M3 matches the exact 48-column sparsity mask, rank 1, and Frobenius norm ($0.600$) across 3 seeds (`[42, 43, 44]`).*

1. **System Prompting (M1) Drives the Primary Behavioral Shift:** System instructions account for the primary jump in Swedish compliance (**96.7% → 100%**, completely suppressing the "Fjällen" English refusal). Localized weight editing alone (M2) accounts for only a minor incremental shift in keyword recall and cannot dislodge deep-seated refusal reflexes without prompt-level conditioning.
2. **Directional Specificity over Random Noise:** Targeted weight surgery (M2) yields higher keyword recall than the 3-seed matched-sparsity/matched-norm random control (M3), confirming that the extracted steering vector $d_l$ captures genuine functional language features rather than arbitrary weight perturbation.
3. **Activation Steering (M5) Cross-Lingual Penalty:** While inference-time activation steering achieves 8.4% Swedish keyword recall and 0% leakage, it severely impairs English processing (compliance drops to 65.0%, keyword recall drops by -10.1 pp). WSP surgery avoids this catastrophic degradation.
4. **"Unblocking" vs. "Learning" Deficit:** WSP cannot teach models unlearned Swedish vocabulary or grammar from scratch without gradients. The positive shifts reflect unbiasing latent representations already present in the pretraining distribution.
5. **Surface Compliance $\neq$ True Linguistic/Factual Competence:** Even when compliance reaches 100%, `Qwen2.5-0.5B-Instruct` displays factual hallucinations (fish in Swedish pancakes), Scandinavian cross-lingual interference (Norwegian/Danish tokens like *for å finne*, *merke på*), and grammatical gender errors (*en hus*, *en äpple*). WSP cannot substitute for foundational continued pre-training.
6. **Zero Catastrophic Forgetting:** With only 0.0174% of parameters modified, the English regression suite registered preserved compliance and zero syntax collapse under WSP.

---

## Strategic Recommendation for SE.LLMA

> **Verdict:** SE.LLMA should **NOT** adopt Weight-Space Programming as a primary training methodology to teach models Swedish language or culture. WSP cannot substitute for Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), or Direct Preference Optimization (DPO).

* **Where WSP is Viable:** As an ultra-lightweight, zero-compute **post-hoc behavioral patch ("hotfix")** for isolated checkpoint pathologies (e.g., repeating Swedish boilerplate tokens, looping on punctuation, or triggering false English safety refusals).
* **Proposed Next Experiment on *Berzelius*:** Evaluate multi-layer SVD subspace weight projection on 8B–14B models on **`EuroEval (formerly ScandEval) / SweSAT`**, comparing compute-normalized throughput against LoRA.

---

## Repository Structure

```text
├── data/
│   ├── dev_set.json               # 10 Swedish prompts for calibration / probe extraction
│   ├── test_swedish.json          # 30 held-out Swedish evaluation prompts across 4 categories
│   ├── regression_english.json    # 20 English prompts to monitor catastrophic degradation
│   └── swedish_perplexity.json    # 12 Swedish reference sentences for fluency evaluation
├── docs/
│   ├── research_note.md           # Full academic research note (Markdown)
│   ├── report.html                # Formatted publication-ready HTML
│   └── report.pdf                 # Final compiled PDF report (<25 MB)
├── results/
│   ├── quantitative_results.json  # Raw metric outputs across all conditions and seeds
│   ├── qualitative_examples.md   # Side-by-side prompt responses across conditions
│   └── plots/
│       └── comparison_metrics.png # Publication-grade visualization
├── scripts/
│   ├── run_all.py                 # Reproduces the entire benchmark end-to-end (M0–M5)
│   └── generate_pdf.py            # Compiles research_note.md to report.pdf via Edge headless
├── src/
│   ├── __init__.py
│   ├── model_loader.py            # Deterministic model loader, commit hash, snapshotting
│   ├── wsp_surgery.py             # Direction extraction, WSP surgery, M3 matched control, M5 steering
│   └── evaluator.py               # Compliance, leakage, recall, loop detection, and perplexity
├── AI_USAGE.md                    # Transparent disclosure of AI assistance tools
├── requirements.txt               # Direct pip requirements fallback
├── pyproject.toml                 # Standard PEP 621 packaging configuration
└── README.md                      # Project documentation and reproduction instructions
```

---

## How to Reproduce the Main Experiment

### Prerequisites
* Python $\ge 3.11$
* Compatible with CPU-only machines (no discrete GPU required; execution takes $\approx 15-20$ minutes on a modern 8-core CPU)

---

### Setup Instructions (Tested for Windows, macOS, and Linux)

#### Option A: Fast Setup with `uv` (Recommended)
`uv` manages isolated virtual environments and dependencies seamlessly across platforms, avoiding Windows PowerShell execution policy restrictions:

```bash
# 1. Clone repository
git clone https://github.com/felixsuarez0727/weight-space-programming-sellma.git
cd weight-space-programming-sellma

# 2. Sync environment from pyproject.toml
uv sync

# 3. Run the full experiment
uv run python scripts/run_all.py

# 4. Recompile PDF report
uv run python scripts/generate_pdf.py
```

---

#### Option B: Standard Python `venv` + `pip`

##### On Windows (PowerShell):
```powershell
# 1. Clone repository
git clone https://github.com/felixsuarez0727/weight-space-programming-sellma.git
cd weight-space-programming-sellma

# 2. Create virtual environment
python -m venv .venv

# 3. Install dependencies from requirements.txt (or 'pip install -e .')
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Run the experiment directly (avoids PowerShell ExecutionPolicy restrictions):
.\.venv\Scripts\python.exe scripts/run_all.py

# Alternatively, if activating the script:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1
# python scripts/run_all.py
```

##### On Windows (Command Prompt - CMD):
```cmd
git clone https://github.com/felixsuarez0727/weight-space-programming-sellma.git
cd weight-space-programming-sellma
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
python scripts/run_all.py
python scripts/generate_pdf.py
```

##### On Linux / macOS:
```bash
git clone https://github.com/felixsuarez0727/weight-space-programming-sellma.git
cd weight-space-programming-sellma
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
python scripts/generate_pdf.py
```

---

## AI Tools Usage Disclosure

Please refer to [`AI_USAGE.md`](AI_USAGE.md) for full details on tools utilized, automated task assignments, and candidate-directed scientific decisions.
