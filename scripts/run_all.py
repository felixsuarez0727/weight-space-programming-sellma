"""
Main experiment runner for SE.LLMA Weight-Space Programming study.

Executes all 5 experimental conditions across:
- Held-out Swedish test set (30 prompts)
- Held-out English regression set (20 prompts)

Generates:
- results/quantitative_results.json
- results/qualitative_examples.md
- results/plots/comparison_metrics.png
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model_loader import (
    load_model_and_tokenizer,
    get_weights_snapshot,
    restore_weights_snapshot,
    set_seed,
)
from src.wsp_surgery import (
    extract_language_steering_vectors,
    apply_wsp_surgery,
    apply_random_surgery,
)
from src.evaluator import evaluate_model_on_dataset


SYSTEM_PROMPT_SWEDISH = (
    "Du är en svenskspråkig AI-assistent. Svara alltid och uteslutande på flytande och korrekt svenska."
)


def load_datasets():
    with open(ROOT_DIR / "data" / "dev_set.json", "r", encoding="utf-8") as f:
        dev_set = json.load(f)
    with open(ROOT_DIR / "data" / "test_swedish.json", "r", encoding="utf-8") as f:
        test_swedish = json.load(f)
    with open(ROOT_DIR / "data" / "regression_english.json", "r", encoding="utf-8") as f:
        reg_english = json.load(f)
    return dev_set, test_swedish, reg_english


def run_experiment():
    set_seed(42)
    dev_set, test_swedish, reg_english = load_datasets()

    print("=" * 80)
    print("WEIGHT-SPACE PROGRAMMING FOR SE.LLMA: EXPERIMENTAL RUN")
    print(f"Model: Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Dev samples (calibration): {len(dev_set)}")
    print(f"Swedish test samples (held-out): {len(test_swedish)}")
    print(f"English regression samples (held-out): {len(reg_english)}")
    print("=" * 80)

    print("\n[Step 1/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    clean_snapshot = get_weights_snapshot(model)

    # 1. Dev extraction for WSP
    print("\n[Step 2/5] Extracting language steering directions from Dev set...")
    dev_sv_prompts = [d["prompt"] for d in dev_set]
    # Matched English prompts for contrast
    dev_en_prompts = [
        "What is the difference between a lake and an ocean?",
        "What is the right of public access in Sweden?",
        "Explain briefly how photosynthesis works.",
        "If Anna is older than Björn, and Björn is older than Carl, who is youngest?",
        "What does the saying 'There is no place like home' mean?",
        "How many members are in parliament and how often are elections held?",
        "Give me three tips to improve sleep.",
        "Why do tree leaves change color in autumn?",
        "Describe a classic coffee break and what people eat.",
        "What is the capital of Denmark and the strait between Denmark and Sweden?",
    ]

    target_layers = [14, 16]
    steering_dirs = extract_language_steering_vectors(
        model=model,
        tokenizer=tokenizer,
        dev_prompts_sv=dev_sv_prompts,
        dev_prompts_en=dev_en_prompts,
        layer_indices=target_layers,
    )
    print(f"Steering directions computed for layers: {target_layers}")

    experiment_arms = {}

    # --- Condition 0: Baseline ---
    print("\n[Evaluating Condition 0: Baseline (Unchanged Model)]")
    restore_weights_snapshot(model, clean_snapshot)
    res_sv_m0 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=None)
    res_en_m0 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    experiment_arms["M0_Baseline"] = {
        "description": "Unaltered base weights, standard user prompt",
        "swedish_eval": res_sv_m0,
        "english_regression_eval": res_en_m0,
        "surgery_stats": None,
    }

    # --- Condition 1: Prompt Control ---
    print("\n[Evaluating Condition 1: Prompt Control (System Instruction)]")
    restore_weights_snapshot(model, clean_snapshot)
    res_sv_m1 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=SYSTEM_PROMPT_SWEDISH)
    res_en_m1 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    experiment_arms["M1_Prompt_Control"] = {
        "description": "Unaltered base weights, explicit Swedish system prompt",
        "swedish_eval": res_sv_m1,
        "english_regression_eval": res_en_m1,
        "surgery_stats": None,
    }

    # --- Condition 2: WSP Targeted Surgery ---
    print("\n[Evaluating Condition 2: WSP Targeted Weight Surgery]")
    restore_weights_snapshot(model, clean_snapshot)
    wsp_stats = apply_wsp_surgery(model, steering_dirs, alpha=0.6, top_k_neurons=48)
    print(f"WSP applied: {wsp_stats.total_params_modified} parameters modified ({wsp_stats.fraction_params_modified*100:.4f}%)")
    res_sv_m2 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=None)
    res_en_m2 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    experiment_arms["M2_WSP_Surgery"] = {
        "description": "Surgical rank-1 top-48 neuron weight edit on layers 14, 16 (0.017% params)",
        "swedish_eval": res_sv_m2,
        "english_regression_eval": res_en_m2,
        "surgery_stats": {
            "total_params_modified": wsp_stats.total_params_modified,
            "fraction_params_modified": wsp_stats.fraction_params_modified,
            "frobenius_norms": wsp_stats.frobenius_norms,
        },
    }

    # --- Condition 3: Random Weight Perturbation Control ---
    print("\n[Evaluating Condition 3: Random Weight Perturbation Control (Matched Norm)]")
    restore_weights_snapshot(model, clean_snapshot)
    rand_stats = apply_random_surgery(model, wsp_stats, seed=42)
    print(f"Random surgery applied: {rand_stats.total_params_modified} parameters modified with matched Frobenius norm")
    res_sv_m3 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=None)
    res_en_m3 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    experiment_arms["M3_Random_Weight_Control"] = {
        "description": "Untargeted Gaussian weight noise with identical Frobenius norm on layers 14, 16",
        "swedish_eval": res_sv_m3,
        "english_regression_eval": res_en_m3,
        "surgery_stats": {
            "total_params_modified": rand_stats.total_params_modified,
            "frobenius_norms": rand_stats.frobenius_norms,
        },
    }

    # --- Condition 4: WSP + Prompt Combined ---
    print("\n[Evaluating Condition 4: WSP Surgery + Prompt Combined]")
    restore_weights_snapshot(model, clean_snapshot)
    apply_wsp_surgery(model, steering_dirs, alpha=0.6, top_k_neurons=48)
    res_sv_m4 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=SYSTEM_PROMPT_SWEDISH)
    res_en_m4 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    experiment_arms["M4_WSP_Plus_Prompt"] = {
        "description": "WSP surgical weight edit combined with explicit system prompt",
        "swedish_eval": res_sv_m4,
        "english_regression_eval": res_en_m4,
        "surgery_stats": {
            "total_params_modified": wsp_stats.total_params_modified,
            "fraction_params_modified": wsp_stats.fraction_params_modified,
        },
    }

    # Restore clean state
    restore_weights_snapshot(model, clean_snapshot)

    # Save results
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    results_path = ROOT_DIR / "results" / "quantitative_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(experiment_arms, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Quantitative results saved to: {results_path}")

    # Generate Qualitative Markdown Comparison
    generate_qualitative_report(experiment_arms, test_swedish, reg_english)

    # Generate Plots
    generate_plots(experiment_arms)

    # Print Summary Table
    print_summary_table(experiment_arms)


def print_summary_table(experiment_arms):
    headers = [
        "Condition",
        "Params Mod (%)",
        "SV Compliance (%)",
        "SV Leakage (%)",
        "SV Keyw. Recall (%)",
        "EN Compliance (%)",
        "EN Regression",
    ]
    rows = []
    base_en_comp = experiment_arms["M0_Baseline"]["english_regression_eval"]["compliance_rate"]

    for name, data in experiment_arms.items():
        sv = data["swedish_eval"]
        en = data["english_regression_eval"]
        stats = data["surgery_stats"]
        p_mod = f"{stats['fraction_params_modified']*100:.3f}%" if stats and "fraction_params_modified" in stats else "0.0%"

        en_drop = base_en_comp - en["compliance_rate"]
        reg_str = f"-{en_drop*100:.1f} pp" if en_drop > 0 else "0.0 pp"

        rows.append([
            name,
            p_mod,
            f"{sv['compliance_rate']*100:.1f}%",
            f"{sv['leakage_rate']*100:.1f}%",
            f"{sv['avg_keyword_recall']*100:.1f}%",
            f"{en['compliance_rate']*100:.1f}%",
            reg_str,
        ])

    print("\n" + "=" * 90)
    print("EXPERIMENTAL SUMMARY TABLE (SE.LLMA / WEIGHT-SPACE PROGRAMMING)")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("=" * 90 + "\n")


def generate_qualitative_report(experiment_arms, test_swedish, reg_english):
    md_lines = [
        "# Qualitative Evaluation: Model Responses Across Conditions",
        "",
        "This document presents paired sample outputs across the experimental conditions:",
        "- **M0**: Baseline (Unchanged)",
        "- **M1**: Prompt Control (System Instruction)",
        "- **M2**: WSP Surgery (Targeted Parameter Edit)",
        "- **M3**: Random Weight Control (Matched Frobenius Norm)",
        "- **M4**: WSP + Prompt (Combined)",
        "",
        "---",
        "",
        "## Swedish Test Samples (Held-Out)",
        "",
    ]

    # Select representative samples
    sample_indices = [0, 4, 7, 14, 20]  # Cultural, Nobel, Conversational, Morphology, Logic

    for idx in sample_indices:
        if idx >= len(test_swedish):
            continue
        item = test_swedish[idx]
        prompt = item["prompt"]
        cat = item["category"]

        md_lines.append(f"### Prompt {idx+1} `[{cat}]`: *\"{prompt}\"*")
        md_lines.append("")

        for arm_name in ["M0_Baseline", "M1_Prompt_Control", "M2_WSP_Surgery", "M3_Random_Weight_Control", "M4_WSP_Plus_Prompt"]:
            rec = experiment_arms[arm_name]["swedish_eval"]["records"][idx]
            resp = rec["response"].replace("\n", " ")
            eval_info = rec["eval"]
            status = "PASS (SV)" if eval_info["is_compliant"] else ("LEAK (EN)" if eval_info["is_leakage"] else "MIXED")
            md_lines.append(f"- **{arm_name}** `[{status}]`: {resp[:200]}...")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    qual_path = ROOT_DIR / "results" / "qualitative_examples.md"
    with open(qual_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Saved] Qualitative comparison saved to: {qual_path}")


def generate_plots(experiment_arms):
    plot_dir = ROOT_DIR / "results" / "plots"
    names = ["M0_Base", "M1_Prompt", "M2_WSP", "M3_Random", "M4_WSP+P"]
    
    sv_comp = [experiment_arms[k]["swedish_eval"]["compliance_rate"] * 100 for k in experiment_arms]
    sv_leak = [experiment_arms[k]["swedish_eval"]["leakage_rate"] * 100 for k in experiment_arms]
    en_comp = [experiment_arms[k]["english_regression_eval"]["compliance_rate"] * 100 for k in experiment_arms]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    rects1 = ax.bar(x - width, sv_comp, width, label="Swedish Compliance (%)", color="#1f77b4")
    rects2 = ax.bar(x, sv_leak, width, label="English Leakage (%)", color="#d62728")
    rects3 = ax.bar(x + width, en_comp, width, label="English Control/Preservation (%)", color="#2ca02c")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Weight-Space Programming vs Controls (SE.LLMA / Qwen2.5-0.5B)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 115)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plot_file = plot_dir / "comparison_metrics.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"[Saved] Plot saved to: {plot_file}")


if __name__ == "__main__":
    run_experiment()

