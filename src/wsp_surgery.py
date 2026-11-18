"""
Weight-Space Programming (WSP) Surgery Module.

Implements direct weight parameter interventions without gradient descent or backpropagation:
1. Directional extraction from hidden states / activation subspaces across development examples.
2. Targeted rank-1 / top-k neuron weight modification on MLP down-projection matrices.
3. Strict matched-norm random weight perturbation control (same Frobenius norm, random direction).
"""

from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class SurgeryStats:
    layer_indices: list[int]
    total_params_modified: int
    fraction_params_modified: float
    frobenius_norms: dict[int, float]
    intervention_type: str


def extract_language_steering_vectors(
    model: torch.nn.Module,
    tokenizer,
    dev_prompts_sv: list[str],
    dev_prompts_en: list[str],
    layer_indices: list[int] = [14, 15, 16],
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Extract language-differentiating direction vectors at specified layers
    using forward passes over development examples (without gradient tracking).
    
    Returns a dict mapping layer_idx -> normalized direction vector d_l in R^{hidden_size}.
    """
    hidden_sv: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}
    hidden_en: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}

    def make_hook(layer_idx: int, storage: dict[int, list[torch.Tensor]]):
        def hook(module, input, output):
            # In transformers, layer output may be a tuple or directly a tensor
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3:
                last_tok_h = h[0, -1, :].detach().cpu()  # [hidden_dim]
            elif h.dim() == 2:
                last_tok_h = h[-1, :].detach().cpu()
            else:
                last_tok_h = h.detach().cpu()
            storage[layer_idx].append(last_tok_h)
        return hook

    # Register forward hooks on target layers
    hooks = []
    for l in layer_indices:
        h = model.model.layers[l].register_forward_hook(make_hook(l, hidden_sv))
        hooks.append(h)

    # Forward Swedish prompts
    with torch.no_grad():
        for prompt in dev_prompts_sv:
            inp = tokenizer(prompt, return_tensors="pt").to(device)
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
            inp = tokenizer(prompt, return_tensors="pt").to(device)
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
    alpha: float = 0.5,
    top_k_neurons: int = 64,
) -> SurgeryStats:
    """
    Apply targeted Weight-Space Programming edit directly to MLP down_proj matrices:
    W_down is in R^{hidden_size, intermediate_size}.
    
    We identify the intermediate neurons whose output projection aligns most with the
    target direction, and boost/steer them via a localized low-rank weight addition.
    
    W_down' = W_down + alpha * (direction (x) neuron_weights_projection)
    """
    total_model_params = sum(p.numel() for p in model.parameters())
    modified_params = 0
    frobenius_norms = {}
    layer_indices = list(steering_dirs.keys())

    for l, d_l in steering_dirs.items():
        w_down = model.model.layers[l].mlp.down_proj.weight  # shape: [896, 4864]
        # d_l shape: [896]
        d_l_col = d_l.unsqueeze(1)  # [896, 1]

        # Project w_down onto d_l to find neuron alignment:
        # alignment_j = (d_l^T * w_down[:, j])
        alignments = torch.matmul(d_l.unsqueeze(0), w_down).squeeze(0)  # [4864]
        
        # Select top-k aligned neurons
        _, top_indices = torch.topk(torch.abs(alignments), k=min(top_k_neurons, alignments.shape[0]))
        
        # Construct sparse rank-1 update mask
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
        
        # Count only non-zero modified parameters
        nonzero_count = int((delta_w != 0).sum().item())
        modified_params += nonzero_count

    return SurgeryStats(
        layer_indices=layer_indices,
        total_params_modified=modified_params,
        fraction_params_modified=modified_params / total_model_params,
        frobenius_norms=frobenius_norms,
        intervention_type="wsp_targeted",
    )


def apply_random_surgery(
    model: torch.nn.Module,
    reference_stats: SurgeryStats,
    seed: int = 42,
) -> SurgeryStats:
    """
    Rigorous Control: Apply untargeted random weight edit to the EXACT same layers
    and with the EXACT same Frobenius norm as the WSP surgery.
    """
    torch.manual_seed(seed)
    total_model_params = sum(p.numel() for p in model.parameters())
    modified_params = 0
    frobenius_norms = {}

    for l in reference_stats.layer_indices:
        target_norm = reference_stats.frobenius_norms[l]
        w_down = model.model.layers[l].mlp.down_proj.weight
        
        # Generate random Gaussian noise with same shape
        rand_delta = torch.randn_like(w_down)
        current_norm = torch.norm(rand_delta, p="fro")
        if current_norm > 1e-8:
            rand_delta = rand_delta * (target_norm / current_norm)
        
        with torch.no_grad():
            w_down.add_(rand_delta)

        frobenius_norms[l] = float(torch.norm(rand_delta, p="fro").item())
        modified_params += rand_delta.numel()

    return SurgeryStats(
        layer_indices=reference_stats.layer_indices,
        total_params_modified=modified_params,
        fraction_params_modified=modified_params / total_model_params,
        frobenius_norms=frobenius_norms,
        intervention_type="random_untargeted",
    )
