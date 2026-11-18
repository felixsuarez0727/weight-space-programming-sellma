"""
Weight-Space Programming (WSP) Surgery Module.

Implements direct weight parameter interventions without gradient descent or backpropagation:
1. Directional extraction from hidden states / activation subspaces across development examples.
2. Targeted rank-1 / top-k neuron weight modification on MLP down-projection matrices.
3. Strict matched-norm and matched-sparsity random rank-1 weight perturbation control.
4. Inference-time activation steering hook helper for non-destructive representation steering.
"""

from dataclasses import dataclass
import contextlib
import torch
import numpy as np


@dataclass
class SurgeryStats:
    layer_indices: list[int]
    total_params_modified: int
    fraction_params_modified: float
    frobenius_norms: dict[int, float]
    intervention_type: str
    alpha: float | None = None
    top_k_neurons: int | None = None
    seeds: list[int] | None = None


def extract_language_steering_vectors(
    model: torch.nn.Module,
    tokenizer,
    dev_prompts_sv: list[str],
    dev_prompts_en: list[str],
    layer_indices: list[int] = [14, 15, 16],
    use_chat_template: bool = True,
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Extract language-differentiating direction vectors at specified layers
    using forward passes over development examples (without gradient tracking).
    
    When use_chat_template=True, formats prompts via chat template to measure
    hidden states at the exact assistant generation boundary.
    
    Returns a dict mapping layer_idx -> normalized direction vector d_l in R^{hidden_size}.
    """
    hidden_sv: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}
    hidden_en: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}

    def make_hook(layer_idx: int, storage: dict[int, list[torch.Tensor]]):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                last_tok_h = h[0, -1, :].detach().cpu()  # [hidden_dim]
            elif h.dim() == 2:
                last_tok_h = h[-1, :].detach().cpu()
            else:
                last_tok_h = h.detach().cpu()
            storage[layer_idx].append(last_tok_h)
        return hook

    def format_text(p: str) -> str:
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return p

    # Register forward hooks on target layers
    hooks = []
    for l in layer_indices:
        h = model.model.layers[l].register_forward_hook(make_hook(l, hidden_sv))
        hooks.append(h)

    # Forward Swedish prompts
    with torch.no_grad():
        for prompt in dev_prompts_sv:
            inp = tokenizer(format_text(prompt), return_tensors="pt").to(device)
            model(**inp)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Register hooks for English prompts
    hooks = []
    for l in layer_indices:
        h = model.model.layers[l].register_forward_hook(make_hook(l, hidden_en))
        hooks.append(h)

    # Forward English prompts
    with torch.no_grad():
        for prompt in dev_prompts_en:
            inp = tokenizer(format_text(prompt), return_tensors="pt").to(device)
            model(**inp)

    for h in hooks:
        h.remove()

    steering_dirs = {}
    for l in layer_indices:
        mean_sv = torch.stack(hidden_sv[l]).mean(dim=0)
        mean_en = torch.stack(hidden_en[l]).mean(dim=0)
        diff = mean_sv - mean_en
        norm = torch.norm(diff, p=2)
        if norm > 1e-8:
            direction = diff / norm
        else:
            direction = diff
        steering_dirs[l] = direction.to(device)

    return steering_dirs


def apply_wsp_surgery(
    model: torch.nn.Module,
    steering_dirs: dict[int, torch.Tensor],
    alpha: float = 0.6,
    top_k_neurons: int = 48,
) -> SurgeryStats:
    """
    Apply targeted Weight-Space Programming edit directly to MLP down_proj matrices:
    W_down is in R^{hidden_size, intermediate_size}.
    
    Identifies intermediate neurons whose output projection aligns most with the
    target direction, and steers them via a localized column-sparse rank-1 weight addition.
    
    W_down' = W_down + alpha * (direction (x) neuron_weights_projection)
    """
    total_model_params = sum(p.numel() for p in model.parameters())
    modified_params = 0
    frobenius_norms = {}
    layer_indices = list(steering_dirs.keys())

    for l, d_l in steering_dirs.items():
        w_down = model.model.layers[l].mlp.down_proj.weight  # shape: [896, 4864]
        d_l_col = d_l.unsqueeze(1)  # [896, 1]

        # Project w_down onto d_l to find neuron alignment
        alignments = torch.matmul(d_l.unsqueeze(0), w_down).squeeze(0)  # [4864]
        
        # Select top-k aligned neurons
        _, top_indices = torch.topk(torch.abs(alignments), k=min(top_k_neurons, alignments.shape[0]))
        
        # Construct column-sparse rank-1 update vector
        u_l = torch.zeros_like(alignments)
        u_l[top_indices] = alignments[top_indices]
        u_norm = torch.norm(u_l, p=2)
        if u_norm > 1e-8:
            u_l = u_l / u_norm

        # Delta W: [896, 4864]
        delta_w = alpha * torch.matmul(d_l_col, u_l.unsqueeze(0))
        
        with torch.no_grad():
            w_down.add_(delta_w)

        f_norm = float(torch.norm(delta_w, p="fro").item())
        frobenius_norms[l] = f_norm
        
        nonzero_count = int((delta_w != 0).sum().item())
        modified_params += nonzero_count

    return SurgeryStats(
        layer_indices=layer_indices,
        total_params_modified=modified_params,
        fraction_params_modified=modified_params / total_model_params,
        frobenius_norms=frobenius_norms,
        intervention_type="wsp_targeted",
        alpha=alpha,
        top_k_neurons=top_k_neurons,
        seeds=None,
    )


def apply_random_surgery(
    model: torch.nn.Module,
    reference_stats: SurgeryStats,
    seed: int = 42,
) -> SurgeryStats:
    """
    Rigorous Control: Apply untargeted random weight edit to the EXACT same layers,
    matching BOTH column sparsity (top_k_neurons) and rank-1 structure,
    scaled to the EXACT same Frobenius norm as the WSP surgery.
    """
    generator = torch.Generator().manual_seed(seed)
    total_model_params = sum(p.numel() for p in model.parameters())
    modified_params = 0
    frobenius_norms = {}
    k = reference_stats.top_k_neurons or 48

    for l in reference_stats.layer_indices:
        target_norm = reference_stats.frobenius_norms[l]
        w_down = model.model.layers[l].mlp.down_proj.weight  # [896, 4864]
        hidden_dim, intermediate_dim = w_down.shape

        # Sample k random column indices without replacement
        perm = torch.randperm(intermediate_dim, generator=generator)
        selected_cols = perm[:k]

        # Sample random unit direction vector in R^{hidden_dim}
        d_rand = torch.randn(hidden_dim, generator=generator)
        d_rand = d_rand / torch.norm(d_rand, p=2)

        # Sample random unit sparse vector supported ONLY on selected_cols
        u_rand = torch.zeros(intermediate_dim)
        u_vals = torch.randn(k, generator=generator)
        u_rand[selected_cols] = u_vals / torch.norm(u_vals, p=2)

        # Construct rank-1 delta W with exact target Frobenius norm
        rand_delta = target_norm * torch.matmul(d_rand.unsqueeze(1), u_rand.unsqueeze(0)).to(w_down.device)

        with torch.no_grad():
            w_down.add_(rand_delta)

        frobenius_norms[l] = float(torch.norm(rand_delta, p="fro").item())
        nonzero_count = int((rand_delta != 0).sum().item())
        modified_params += nonzero_count

    return SurgeryStats(
        layer_indices=reference_stats.layer_indices,
        total_params_modified=modified_params,
        fraction_params_modified=modified_params / total_model_params,
        frobenius_norms=frobenius_norms,
        intervention_type="random_untargeted_matched_sparse",
        alpha=reference_stats.alpha,
        top_k_neurons=k,
        seeds=[seed],
    )


@contextlib.contextmanager
def register_activation_steering_hooks(
    model: torch.nn.Module,
    steering_dirs: dict[int, torch.Tensor],
    beta: float = 1.0,
):
    """
    Inference-time Activation Steering Control (Condition M5).
    Non-destructively adds beta * d_l to the residual stream output of specified layers
    during forward passes without modifying any weights.
    """
    hooks = []
    for l, d_l in steering_dirs.items():
        layer = model.model.layers[l]
        vec = (beta * d_l).detach()

        def make_steer_hook(steer_v):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    steer_device = steer_v.to(h.device, dtype=h.dtype)
                    h = h + steer_device
                    return (h,) + output[1:]
                else:
                    steer_device = steer_v.to(output.device, dtype=output.dtype)
                    return output + steer_device
            return hook

        h = layer.register_forward_hook(make_steer_hook(vec))
        hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()
