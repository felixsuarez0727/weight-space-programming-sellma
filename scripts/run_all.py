"""
Main experiment runner for SE.LLMA Weight-Space Programming study.

Executes all 6 experimental conditions across:
- Held-out Swedish test set (30 prompts)
- Held-out English regression set (20 prompts)
- Swedish perplexity reference set (12 sentences)

Conditions:
- M0: Baseline (Unaltered)
- M1: Prompt Control (System Instruction)
- M2: WSP Targeted Surgery (0.0174% parameters modified)
- M3: Random Weight Control (Matched column sparsity, rank-1, matched Frobenius norm, 3 seeds: 42, 43, 44)
- M4: WSP + Prompt (Combined)
- M5: Activation Steering (Inference-time residual steering vector)

Generates:
- results/quantitative_results.json (including full config, model commit hash, and per-seed metrics)
- results/qualitative_examples.md (including indices [0, 4, 7, 8, 9, 14, 20])
- results/plots/comparison_metrics.png
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.model_loader import (
    load_model_and_tokenizer,
    get_weights_snapshot,
    restore_weights_snapshot,
    get_model_revision,
    set_seed,
    generate_response,
)
from src.wsp_surgery import (
    extract_language_steering_vectors,
    apply_wsp_surgery,
    apply_random_surgery,
    register_activation_steering_hooks,
)
from src.evaluator import (
    evaluate_model_on_dataset,
    compute_perplexity,
)


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
    with open(ROOT_DIR / "data" / "swedish_perplexity.json", "r", encoding="utf-8") as f:
        ppl_sentences = json.load(f)
    return dev_set, test_swedish, reg_english, ppl_sentences


def run_experiment():
    set_seed(42)
    dev_set, test_swedish, reg_english, ppl_sentences = load_datasets()

    print("=" * 85)
    print("WEIGHT-SPACE PROGRAMMING FOR SE.LLMA: COMPREHENSIVE BENCHMARK RUN")
    print(f"Model: Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Dev samples (calibration): {len(dev_set)}")
    print(f"Swedish test samples (held-out): {len(test_swedish)}")
    print(f"English regression samples (held-out): {len(reg_english)}")
    print(f"Swedish perplexity reference sentences: {len(ppl_sentences)}")
    print("=" * 85)

    print("\n[Step 1/6] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    clean_snapshot = get_weights_snapshot(model)
    model_revision = get_model_revision(model, "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Hugging Face Model Revision (commit hash): {model_revision}")

    # 1. Dev extraction for WSP
    print("\n[Step 2/6] Extracting language steering directions from Dev set (using chat template)...")
    dev_sv_prompts = [d["prompt"] for d in dev_set]
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
        use_chat_template=True,
    )
    print(f"Steering directions computed for layers: {target_layers}")

    # 2. Select beta for Activation Steering strictly on Dev set
    print("\n[Step 3/6] Calibrating activation steering beta strictly on Dev set...")
    candidate_betas = [0.5, 1.0, 2.0]
    dev_selection_log = {}
    best_beta = 1.0
    best_dev_recall = -1.0

    for b in candidate_betas:
        restore_weights_snapshot(model, clean_snapshot)
        with register_activation_steering_hooks(model, steering_dirs, beta=b):
            res_dev = evaluate_model_on_dataset(model, tokenizer, dev_set, system_prompt=None)
        dev_selection_log[str(b)] = {
            "compliance_rate": res_dev["compliance_rate"],
            "leakage_rate": res_dev["leakage_rate"],
            "avg_keyword_recall": res_dev["avg_keyword_recall"],
        }
        print(f"  Beta = {b}: Compliance={res_dev['compliance_rate']*100:.1f}%, Recall={res_dev['avg_keyword_recall']*100:.1f}%")
        if res_dev["avg_keyword_recall"] > best_dev_recall:
            best_dev_recall = res_dev["avg_keyword_recall"]
            best_beta = b

    print(f"Calibrated best beta for M5: {best_beta} (selected based on Dev set performance)")

    experiment_arms = {}
    config_dict = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": model_revision,
        "seed": 42,
        "target_layers": target_layers,
        "alpha": 0.6,
        "top_k_neurons": 48,
        "steering_beta": best_beta,
        "candidate_betas_dev_search": candidate_betas,
        "dev_selection_results": dev_selection_log,
        "max_new_tokens": 80,
        "do_sample": False,
        "use_chat_template": True,
        "m3_seeds": [42, 43, 44],
    }

    # --- Condition 0: Baseline ---
    print("\n[Evaluating Condition 0: Baseline (Unchanged Model)]")
    restore_weights_snapshot(model, clean_snapshot)
    res_sv_m0 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=None)
    res_en_m0 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    ppl_m0 = compute_perplexity(model, tokenizer, ppl_sentences)
    experiment_arms["M0_Baseline"] = {
        "description": "Unaltered base weights, standard user prompt",
        "swedish_eval": res_sv_m0,
        "english_regression_eval": res_en_m0,
        "swedish_perplexity": ppl_m0,
        "surgery_stats": None,
    }

    # --- Condition 1: Prompt Control ---
    print("\n[Evaluating Condition 1: Prompt Control (System Instruction)]")
    restore_weights_snapshot(model, clean_snapshot)
    res_sv_m1 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=SYSTEM_PROMPT_SWEDISH)
    res_en_m1 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    ppl_m1 = compute_perplexity(model, tokenizer, ppl_sentences)
    experiment_arms["M1_Prompt_Control"] = {
        "description": "Unaltered base weights, explicit Swedish system prompt",
        "swedish_eval": res_sv_m1,
        "english_regression_eval": res_en_m1,
        "swedish_perplexity": ppl_m1,
        "surgery_stats": None,
    }

    # --- Condition 2: WSP Targeted Surgery ---
    print("\n[Evaluating Condition 2: WSP Targeted Weight Surgery]")
    restore_weights_snapshot(model, clean_snapshot)
    wsp_stats = apply_wsp_surgery(model, steering_dirs, alpha=0.6, top_k_neurons=48)
    print(f"WSP applied: {wsp_stats.total_params_modified} parameters modified ({wsp_stats.fraction_params_modified*100:.4f}%)")
    print(f"Frobenius norms: {wsp_stats.frobenius_norms}")
    res_sv_m2 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=None)
    res_en_m2 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    ppl_m2 = compute_perplexity(model, tokenizer, ppl_sentences)
    experiment_arms["M2_WSP_Surgery"] = {
        "description": "Surgical rank-1 top-48 neuron weight edit on layers 14, 16 (0.0174% params)",
        "swedish_eval": res_sv_m2,
        "english_regression_eval": res_en_m2,
        "swedish_perplexity": ppl_m2,
        "surgery_stats": {
            "total_params_modified": wsp_stats.total_params_modified,
            "fraction_params_modified": wsp_stats.fraction_params_modified,
            "frobenius_norms": wsp_stats.frobenius_norms,
            "alpha": wsp_stats.alpha,
            "top_k_neurons": wsp_stats.top_k_neurons,
        },
    }

    # --- Condition 3: Random Weight Perturbation Control (Matched Sparse Rank-1, 3 seeds) ---
    print("\n[Evaluating Condition 3: Random Weight Control (Matched Sparsity & Rank-1, seeds=[42, 43, 44])]")
    m3_seed_runs = {}
    m3_sv_comp_list = []
    m3_sv_leak_list = []
    m3_sv_recall_list = []
    m3_sv_loop_list = []
    m3_en_comp_list = []
    m3_en_recall_list = []
    m3_ppl_list = []

    m3_primary_swedish_eval = None
    m3_primary_english_eval = None

    for s in [42, 43, 44]:
        restore_weights_snapshot(model, clean_snapshot)
        r_stats = apply_random_surgery(model, wsp_stats, seed=s)
        sv_eval = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=None)
        en_eval = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
        ppl = compute_perplexity(model, tokenizer, ppl_sentences)

        m3_seed_runs[str(s)] = {
            "swedish_eval": sv_eval,
            "english_regression_eval": en_eval,
            "swedish_perplexity": ppl,
            "frobenius_norms": r_stats.frobenius_norms,
            "modified_params": r_stats.total_params_modified,
        }
        m3_sv_comp_list.append(sv_eval["compliance_rate"])
        m3_sv_leak_list.append(sv_eval["leakage_rate"])
        m3_sv_recall_list.append(sv_eval["avg_keyword_recall"])
        m3_sv_loop_list.append(sv_eval["loop_rate"])
        m3_en_comp_list.append(en_eval["compliance_rate"])
        m3_en_recall_list.append(en_eval["avg_keyword_recall"])
        m3_ppl_list.append(ppl)

        if s == 42:
            m3_primary_swedish_eval = sv_eval
            m3_primary_english_eval = en_eval

        print(f"  Seed {s}: SV Comp={sv_eval['compliance_rate']*100:.1f}%, SV Recall={sv_eval['avg_keyword_recall']*100:.1f}%, PPL={ppl:.2f}")

    experiment_arms["M3_Random_Weight_Control"] = {
        "description": "Untargeted sparse rank-1 Gaussian noise on top-48 columns of layers 14, 16 with matching Frobenius norm (3-seed mean ± std)",
        "swedish_eval": m3_primary_swedish_eval,  # seed 42 for records
        "english_regression_eval": m3_primary_english_eval,
        "swedish_perplexity": float(np.mean(m3_ppl_list)),
        "seed_runs": m3_seed_runs,
        "aggregated_stats": {
            "sv_compliance_mean": float(np.mean(m3_sv_comp_list)),
            "sv_compliance_std": float(np.std(m3_sv_comp_list)),
            "sv_leakage_mean": float(np.mean(m3_sv_leak_list)),
            "sv_leakage_std": float(np.std(m3_sv_leak_list)),
            "sv_keyword_recall_mean": float(np.mean(m3_sv_recall_list)),
            "sv_keyword_recall_std": float(np.std(m3_sv_recall_list)),
            "sv_loop_rate_mean": float(np.mean(m3_sv_loop_list)),
            "sv_loop_rate_std": float(np.std(m3_sv_loop_list)),
            "en_compliance_mean": float(np.mean(m3_en_comp_list)),
            "en_compliance_std": float(np.std(m3_en_comp_list)),
            "en_keyword_recall_mean": float(np.mean(m3_en_recall_list)),
            "en_keyword_recall_std": float(np.std(m3_en_recall_list)),
            "swedish_perplexity_mean": float(np.mean(m3_ppl_list)),
            "swedish_perplexity_std": float(np.std(m3_ppl_list)),
        },
        "surgery_stats": {
            "total_params_modified": wsp_stats.total_params_modified,
            "fraction_params_modified": wsp_stats.fraction_params_modified,
            "frobenius_norms": wsp_stats.frobenius_norms,
            "seeds": [42, 43, 44],
        },
    }

    # --- Condition 4: WSP + Prompt Combined ---
    print("\n[Evaluating Condition 4: WSP Surgery + Prompt Combined]")
    restore_weights_snapshot(model, clean_snapshot)
    apply_wsp_surgery(model, steering_dirs, alpha=0.6, top_k_neurons=48)
    res_sv_m4 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=SYSTEM_PROMPT_SWEDISH)
    res_en_m4 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
    ppl_m4 = compute_perplexity(model, tokenizer, ppl_sentences)
    experiment_arms["M4_WSP_Plus_Prompt"] = {
        "description": "WSP surgical weight edit combined with explicit system prompt",
        "swedish_eval": res_sv_m4,
        "english_regression_eval": res_en_m4,
        "swedish_perplexity": ppl_m4,
        "surgery_stats": {
            "total_params_modified": wsp_stats.total_params_modified,
            "fraction_params_modified": wsp_stats.fraction_params_modified,
            "frobenius_norms": wsp_stats.frobenius_norms,
        },
    }

    # --- Condition 5: Activation Steering (Inference-Time) ---
    print(f"\n[Evaluating Condition 5: Activation Steering Control (beta={best_beta})]")
    restore_weights_snapshot(model, clean_snapshot)
    with register_activation_steering_hooks(model, steering_dirs, beta=best_beta):
        res_sv_m5 = evaluate_model_on_dataset(model, tokenizer, test_swedish, system_prompt=None)
        res_en_m5 = evaluate_model_on_dataset(model, tokenizer, reg_english, system_prompt=None)
        ppl_m5 = compute_perplexity(model, tokenizer, ppl_sentences)
    experiment_arms["M5_Activation_Steering"] = {
        "description": f"Inference-time residual steering vector addition without weight edits (beta={best_beta})",
        "swedish_eval": res_sv_m5,
        "english_regression_eval": res_en_m5,
        "swedish_perplexity": ppl_m5,
        "surgery_stats": {
            "total_params_modified": 0,
            "fraction_params_modified": 0.0,
            "steering_beta": best_beta,
        },
    }

    # Restore clean state
    restore_weights_snapshot(model, clean_snapshot)

    # Save quantitative results including full config
    os.makedirs(ROOT_DIR / "results" / "plots", exist_ok=True)
    full_output = {
        "config": config_dict,
        "conditions": experiment_arms,
    }
    results_path = ROOT_DIR / "results" / "quantitative_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Quantitative results saved to: {results_path}")

    # Generate Qualitative Markdown Comparison
    generate_qualitative_report(experiment_arms, test_swedish)

    # Generate Plots
    generate_plots(experiment_arms)

    # Print Summary Table
    print_summary_table(experiment_arms)


def print_summary_table(experiment_arms):
    headers = [
        "Condition",
        "Params Mod (%)",
        "SV Comp (%)",
        "SV Leak (%)",
        "SV Keyw (%)",
        "SV Loop (%)",
        "SV PPL",
        "EN Comp (%)",
        "EN Keyw (%)",
        "EN Recall Δ",
    ]
    rows = []
    base_en_kw = experiment_arms["M0_Baseline"]["english_regression_eval"]["avg_keyword_recall"]

    for name, data in experiment_arms.items():
        sv = data["swedish_eval"]
        en = data["english_regression_eval"]
        stats = data["surgery_stats"]
        ppl = data.get("swedish_perplexity", 0.0)

        if name == "M3_Random_Weight_Control" and "aggregated_stats" in data:
            agg = data["aggregated_stats"]
            p_mod = "0.017%*"
            sv_comp_str = f"{agg['sv_compliance_mean']*100:.1f}±{agg['sv_compliance_std']*100:.1f}%"
            sv_leak_str = f"{agg['sv_leakage_mean']*100:.1f}±{agg['sv_leakage_std']*100:.1f}%"
            sv_kw_str = f"{agg['sv_keyword_recall_mean']*100:.1f}±{agg['sv_keyword_recall_std']*100:.1f}%"
            sv_loop_str = f"{agg['sv_loop_rate_mean']*100:.1f}%"
            ppl_str = f"{agg['swedish_perplexity_mean']:.2f}±{agg['swedish_perplexity_std']:.2f}"
            en_comp_str = f"{agg['en_compliance_mean']*100:.1f}%"
            en_kw_val = agg['en_keyword_recall_mean']
            en_kw_str = f"{en_kw_val*100:.1f}%"
            delta = (en_kw_val - base_en_kw) * 100
            en_delta_str = f"{delta:+.1f} pp" if abs(delta) >= 0.05 else "0.0 pp"
        else:
            p_mod = f"{stats['fraction_params_modified']*100:.3f}%" if stats and "fraction_params_modified" in stats else "0.0%"
            sv_comp_str = f"{sv['compliance_rate']*100:.1f}%"
            sv_leak_str = f"{sv['leakage_rate']*100:.1f}%"
            sv_kw_str = f"{sv['avg_keyword_recall']*100:.1f}%"
            sv_loop_str = f"{sv['loop_rate']*100:.1f}%"
            ppl_str = f"{ppl:.2f}"
            en_comp_str = f"{en['compliance_rate']*100:.1f}%"
            en_kw_val = en['avg_keyword_recall']
            en_kw_str = f"{en_kw_val*100:.1f}%"
            delta = (en_kw_val - base_en_kw) * 100
            en_delta_str = f"{delta:+.1f} pp" if abs(delta) >= 0.05 else "0.0 pp"

        rows.append([
            name,
            p_mod,
            sv_comp_str,
            sv_leak_str,
            sv_kw_str,
            sv_loop_str,
            ppl_str,
            en_comp_str,
            en_kw_str,
            en_delta_str,
        ])

    print("\n" + "=" * 115)
    print("EXPERIMENTAL SUMMARY TABLE (SE.LLMA / WEIGHT-SPACE PROGRAMMING AUDIT)")
    print("=" * 115)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("=" * 115 + "\n")


def generate_qualitative_report(experiment_arms, test_swedish):
    md_lines = [
        "# Qualitative Evaluation: Model Responses Across Conditions",
        "",
        "This document presents paired sample outputs across all experimental conditions:",
        "- **M0**: Baseline (Unchanged)",
        "- **M1**: Prompt Control (System Instruction)",
        "- **M2**: WSP Surgery (Targeted Parameter Edit, 0.0174% params)",
        "- **M3**: Random Weight Control (Matched Sparsity & Frobenius Norm, Seed 42)",
        "- **M4**: WSP + Prompt (Combined)",
        "- **M5**: Activation Steering (Inference-Time Residual Vector)",
        "",
        "---",
        "",
        "## Swedish Test Samples (Held-Out Audit Points)",
        "",
    ]

    # Indices: [0, 4, 7, 8, 9, 14, 20]
    # Includes sv_test_09 (index 8: pancake loop) and sv_test_10 (index 9: Fjällen refusal)
    sample_indices = [0, 4, 7, 8, 9, 14, 20]

    for idx in sample_indices:
        if idx >= len(test_swedish):
            continue
        item = test_swedish[idx]
        prompt = item["prompt"]
        cat = item["category"]
        test_id = item.get("id", f"prompt_{idx+1}")

        md_lines.append(f"### Prompt {idx+1} (`{test_id}`) `[{cat}]`: *\"{prompt}\"*")
        md_lines.append("")

        for arm_name in ["M0_Baseline", "M1_Prompt_Control", "M2_WSP_Surgery", "M3_Random_Weight_Control", "M4_WSP_Plus_Prompt", "M5_Activation_Steering"]:
            rec = experiment_arms[arm_name]["swedish_eval"]["records"][idx]
            resp = rec["response"].replace("\n", " ")
            eval_info = rec["eval"]
            status = "PASS (SV)" if eval_info["is_compliant"] else ("LEAK (EN)" if eval_info["is_leakage"] else "MIXED")
            md_lines.append(f"- **{arm_name}** `[{status}]`: {resp[:220]}...")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    qual_path = ROOT_DIR / "results" / "qualitative_examples.md"
    with open(qual_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Saved] Qualitative comparison saved to: {qual_path}")


def generate_plots(experiment_arms):
    plot_dir = ROOT_DIR / "results" / "plots"
    names = ["M0_Base", "M1_Prompt", "M2_WSP", "M3_Random", "M4_WSP+P", "M5_Steer"]
    arms_keys = ["M0_Baseline", "M1_Prompt_Control", "M2_WSP_Surgery", "M3_Random_Weight_Control", "M4_WSP_Plus_Prompt", "M5_Activation_Steering"]

    sv_comp = []
    sv_recall = []
    sv_loop = []
    en_recall = []
    m3_recall_err = 0.0

    for k in arms_keys:
        arm = experiment_arms[k]
        if k == "M3_Random_Weight_Control" and "aggregated_stats" in arm:
            agg = arm["aggregated_stats"]
            sv_comp.append(agg["sv_compliance_mean"] * 100)
            sv_recall.append(agg["sv_keyword_recall_mean"] * 100)
            sv_loop.append(agg["sv_loop_rate_mean"] * 100)
            en_recall.append(agg["en_keyword_recall_mean"] * 100)
            m3_recall_err = agg["sv_keyword_recall_std"] * 100
        else:
            sv = arm["swedish_eval"]
            en = arm["english_regression_eval"]
            sv_comp.append(sv["compliance_rate"] * 100)
            sv_recall.append(sv["avg_keyword_recall"] * 100)
            sv_loop.append(sv["loop_rate"] * 100)
            en_recall.append(en["avg_keyword_recall"] * 100)

    x = np.arange(len(names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    rects1 = ax.bar(x - 1.5 * width, sv_comp, width, label="Swedish Compliance (%)", color="#1f77b4")
    
    # Error bar for M3 keyword recall
    yerr = [0.0] * len(names)
    yerr[3] = m3_recall_err
    rects2 = ax.bar(x - 0.5 * width, sv_recall, width, yerr=yerr, capsize=3, label="Swedish Keyword Recall (%)", color="#ff7f0e")
    rects3 = ax.bar(x + 0.5 * width, sv_loop, width, label="Swedish Loop Rate (%)", color="#d62728")
    rects4 = ax.bar(x + 1.5 * width, en_recall, width, label="English Keyword Recall (%)", color="#2ca02c")

    ax.set_ylabel("Metric Value (%)", fontsize=10, fontweight="bold")
    ax.set_title("Weight-Space Programming vs Controls (SE.LLMA / Qwen2.5-0.5B)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 115)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.1:
                ax.annotate(f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=0)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    plot_file = plot_dir / "comparison_metrics.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"[Saved] Updated plot saved to: {plot_file}")


if __name__ == "__main__":
    run_experiment()
