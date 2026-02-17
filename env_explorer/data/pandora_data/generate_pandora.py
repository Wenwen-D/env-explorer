import json
import uuid
from typing import List, Dict, Union, Optional
import numpy as np


def solve_three_boxes(priors, gamma, ground_truth):
    """
    priors: dict like {"A": 0.2, "B": 0.3, "C": 0.5}
    gamma: discount factor in [0, 1]
    ground_truth: the true prize box (e.g., "C")

    Returns:
        expected_value: optimal expected value from initial state
        action_sequence: realized sequence given ground truth
    """

    items = sorted(priors.items(), key=lambda x: x[1], reverse=True)

    # --- recursive value function (does NOT depend on ground truth) ---
    def value_function(current_items):
        if len(current_items) == 1:
            return 1.0

        W = sum(p for _, p in current_items)
        label_star, p_star = max(current_items, key=lambda x: x[1])
        q = p_star / W

        V_guess = q

        remaining = [(l, p) for l, p in current_items if l != label_star]
        V_fail = value_function(remaining)

        V_verify = gamma * (q + (1 - q) * V_fail)

        return max(V_guess, V_verify)

    # --- recursive rollout using ground truth ---
    def rollout(current_items):
        if len(current_items) == 1:
            label, _ = current_items[0]
            return [("COMMIT", label)]

        W = sum(p for _, p in current_items)
        label_star, p_star = max(current_items, key=lambda x: x[1])
        q = p_star / W

        V_guess = q
        remaining = [(l, p) for l, p in current_items if l != label_star]
        V_fail = value_function(remaining)
        V_verify = gamma * (q + (1 - q) * V_fail)

        if V_guess >= V_verify:
            return [("COMMIT", label_star)]
        else:
            # Verify
            if label_star == ground_truth:
                return [("VERIFY", label_star), ("COMMIT", label_star)]
            else:
                return [("VERIFY", label_star)] + rollout(remaining)

    expected_value = value_function(items)
    action_sequence = rollout(items)

    return expected_value, action_sequence


def sample_dirichlet_priors(
    labels: List[str],
    alpha: Union[float, List[float]],
    rng_seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Sample prior probabilities over labels from a Dirichlet distribution.

    Args:
        labels (List[str]): Arm labels, e.g. ["A", "B", "C"].
        alpha (float or List[float]): 
            - If float: symmetric Dirichlet with concentration alpha.
            - If list: per-arm concentration parameters.
        rng_seed (int, optional): Random seed for reproducibility.

    Returns:
        Dict[str, float]: Mapping label -> sampled probability.
    """

    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    K = len(labels)

    # Handle symmetric vs asymmetric Dirichlet
    if isinstance(alpha, (int, float)):
        alpha_vec = np.full(K, float(alpha))
    else:
        if len(alpha) != K:
            raise ValueError("Length of alpha list must match number of labels.")
        alpha_vec = np.array(alpha, dtype=float)

    if np.any(alpha_vec <= 0):
        raise ValueError("Dirichlet concentration parameters must be positive.")

    probs = rng.dirichlet(alpha_vec)

    sampled_priors = {label: float(round(p, 2)) for label, p in zip(labels, probs)}
    return sampled_priors

def sample_pandora_tasks(
    num_tasks: int = 200,
    labels: List[str] = None,
    alpha: float = 1.0,
    output_file: str = "pandora_tasks.json",
    rng_seed: Optional[int] = None,
):
    """
    Generate Pandora's-box tasks with 3 boxes (A/B/C).

    - Priors are sampled from Dirichlet(alpha) each task.
    - Ground-truth box is sampled from the *unrounded* Dirichlet probabilities.
    - gamma is sampled uniformly from {0.0, 0.1, ..., 1.0}.
    - Output schema matches your bandit JSON example, but without decision_boundary_r.
    """
    if labels is None:
        labels = ["A", "B", "C"]
    if len(labels) != 3:
        raise ValueError("This helper is specialized to exactly 3 labels (A/B/C).")

    rng = np.random.default_rng(rng_seed)
    gamma_choices = [round(x / 10, 1) for x in range(0, 11)]  # 0.0 ... 1.0

    def _rounded_priors_sum_to_one(raw_probs: np.ndarray, ndigits: int = 2) -> Dict[str, float]:
        """
        Round to ndigits for JSON readability but force sum==1.0 by adjusting the max entry.
        """
        rounded = np.round(raw_probs, ndigits)
        diff = 1.0 - float(np.sum(rounded))
        # Add diff to the largest component (stable, avoids negative small probs in practice)
        idx = int(np.argmax(rounded))
        rounded[idx] = rounded[idx] + diff
        # Guard tiny negative due to rounding edge cases
        rounded = np.clip(rounded, 0.0, 1.0)
        # Renormalize just in case clipping changed the sum
        rounded = rounded / float(np.sum(rounded))
        # Final round for display
        rounded = np.round(rounded, ndigits)
        # Fix any last rounding drift
        diff2 = 1.0 - float(np.sum(rounded))
        idx2 = int(np.argmax(rounded))
        rounded[idx2] = rounded[idx2] + diff2
        return {lab: float(p) for lab, p in zip(labels, rounded)}

    tasks = []

    for i in range(num_tasks):
        # --- Sample Dirichlet priors (raw for sampling GT) ---
        raw_probs = rng.dirichlet(np.full(len(labels), float(alpha)))

        # Store a rounded-but-summing-to-1 version in JSON
        priors = _rounded_priors_sum_to_one(raw_probs, ndigits=2)

        # Sample ground truth from the *raw* probs (more faithful)
        true_arm = rng.choice(labels, p=raw_probs)

        # Sample gamma with one decimal place
        gamma = float(rng.choice(gamma_choices))

        # Oracle rollout conditioned on ground truth
        _, action_seq = solve_three_boxes(priors=priors, gamma=gamma, ground_truth=true_arm)  # :contentReference[oaicite:1]{index=1}

        # Convert to your string format: COMMIT -> GUESS
        optimal_strategy = []
        for act, lab in action_seq:
            if act == "VERIFY":
                optimal_strategy.append(f"VERIFY {lab}")
            elif act == "COMMIT":
                optimal_strategy.append(f"GUESS {lab}")
            else:
                raise ValueError(f"Unknown action {act}")

        strategy_type = "guess_immediately" if optimal_strategy[0].startswith("GUESS") else "verification_then_guess"

        task_uuid = str(uuid.uuid4())[:8]
        # mimic your bandit naming convention (replace '.' so it's filesystem-friendly)
        task_id_base = f"pandora_{i:04d}_{true_arm}_g{gamma}".replace(".", "")
        task_id = f"{task_id_base}_{task_uuid}"

        task = {
            "task_id": task_id,
            "env": {
                "num_arms": len(labels),
                "labels": labels,
                "priors": priors,
                "true_arm": true_arm,
            },
            "discount_factor": gamma,
            "optimal_policy": {
                "strategy_type": strategy_type,
                "optimal_strategy": optimal_strategy,
            },
        }
        tasks.append(task)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(tasks)} tasks and saved to {output_file}")
    print(f" - labels: {labels}")
    print(f" - alpha: {alpha}")
    print(f" - gamma choices: {gamma_choices}")
# for _ in range(5):
#     priors = sample_dirichlet_priors(labels=["A", "B", "C"], alpha=0.5)
#     max_label = max(priors, key=priors.get)
#     print(f'alpha=0.5 sampled priors: {priors}, max label: {max_label}')

# for _ in range(5):
#     priors = sample_dirichlet_priors(labels=["A", "B", "C"], alpha=1.0)
#     max_label = max(priors, key=priors.get)
#     print(f'alpha=1.0 sampled priors: {priors}, max label: {max_label}')

# for _ in range(5):
#     priors = sample_dirichlet_priors(labels=["A", "B", "C"], alpha=1.5)
#     max_label = max(priors, key=priors.get)
#     print(f'alpha=1.5 sampled priors: {priors}, max label: {max_label}')
# priors = {"A": 0.2, "B": 0.3, "C": 0.5}

# for gamma in [0.61, 0.617, 0.618, 0.619,0.62, 0.9]:
#     for gt in ["A", "B", "C"]:
#         print(f"Gamma: {gamma}, Ground Truth: {gt}")
#         value, actions = solve_three_boxes(priors, gamma, gt)
#         print("Expected value:", value)
#         print("Action sequence:", actions)


if __name__ == "__main__":
    sample_pandora_tasks(
        num_tasks=200,
        labels=["A", "B", "C"],
        alpha=1.2,
        output_file="pandora_tasks_dirichlet.json",
        rng_seed=42,
    )