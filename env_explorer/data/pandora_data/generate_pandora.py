# generate_bandit.py
# p = 0.2, 0.3, 0.5 priors = [0.2, 0.3, 0.5], labels = [A, B, C]
# total samples = 10
# => truth = ...
'''
{
  "task_id": "string",                 // Unique identifier (e.g. "bandit_3bags_r0.7_20251013_001")
  
  "env": {
    "labels": ["A", "B", "C"],         // Arm labels, for readability
    "priors": {"A": 0.2, "B": 0.3, "C": 0.5}, // True prior probabilities
    "true_arm": "C"                    // (Optional, for simulation) which arm contains the prize
  },
  "game_params":{
    "discount_factor": 0.7,              // r
    "time_limit": 10,                    // (Optional) maximum steps allowed
  },
  "optimal_policy": {
    "optimal_action_sequence": ["VERIFY C", "GUESS C"],
    "strategy_type": "verification_then_guess", // e.g. "always_guess", "verification_then_guess"
    "first_action": "VERIFY C",           // optimal first move
    "value_star": 0.49,                   // expected reward under optimal play
    "decision_boundary_r": [0.625]        // threshold(s) at which optimal policy changes
  },
  "evaluation": {
    "gold_action_sequence": ["VERIFY C", "GUESS C"], // optional canonical optimal sequence
    "regret_tolerance": 0.05,                        // acceptable regret margin for success
    "scoring_metric": "expected_return"              // could also be "first_action_accuracy"
  },
}
'''
import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

def best_strategy(priors, r, ground_truth):
    # if the probability values in priors is [0.2, 0.3, 0.5], then do this:
    labels_sorted = sorted(priors, key=priors.get, reverse=True)
    if sorted(list(priors.values())) == [0.2, 0.3, 0.5]:
        if r < 0.618:
            return "guess_immediately", [f"GUESS {labels_sorted[0]}"], 0.618
        else:
            strategy = "verification_then_guess"
            Step_1 = f"VERIFY {labels_sorted[0]}"
            Step_2 = f"VERIFY {labels_sorted[1]},{labels_sorted[2]}"
            if ground_truth == labels_sorted[0]:
                return strategy, [f"{Step_1}", f"GUESS {labels_sorted[0]}"], 0.618
            else:
                return strategy, [f"{Step_1}", f"{Step_2}", f"GUESS {ground_truth}"], 0.618
            # return "verification_twice_then_guess", f"VERIFY {ground_truth}"
    else:
        raise NotImplementedError("Best strategy not implemented for these priors.")
    
def sample_bandit_tasks(
    priors: Dict[str, float],
    num_samples: int,
    r_list: List[float],
    output_file: str = "pandora_tasks.json",
):
    """
    Generate a list of bandit-style decision tasks for model evaluation.

    Args:
        priors (dict): Mapping from arm labels to prior probabilities.
                       e.g., {"A": 0.2, "B": 0.3, "C": 0.5}.
        num_samples (int): Number of base samples (each will be combined with all r in r_list).
        r_list (list): List of discount factors (e.g., [0.1, 0.2, ..., 0.9]).
        output_file (str): JSON output file path.
    """

    labels = list(priors.keys())
    probs = [priors[l] for l in labels]

    # --- Compute deterministic counts of true_arms ---
    total_assigned = 0
    counts = []
    for i, p in enumerate(probs):
        n = int(round(num_samples * p))
        counts.append(n)
        total_assigned += n

    # Adjust rounding errors to make sum == num_samples
    while total_assigned < num_samples:
        # Give the remaining slots to the arm with highest residual probability
        idx = probs.index(max(probs))
        counts[idx] += 1
        total_assigned += 1
    while total_assigned > num_samples:
        idx = probs.index(min(probs))
        counts[idx] -= 1
        total_assigned -= 1

    true_arms_list = []
    for label, count in zip(labels, counts):
        true_arms_list.extend([label] * count)

    # --- Deterministic ordering ---
    true_arms_list.sort()
    print(true_arms_list)

    now_est = datetime.now(ZoneInfo("America/New_York")).isoformat() # datetime.utcnow().isoformat()
    tasks = []

    for i, true_arm in enumerate(true_arms_list):
        for r in r_list:
            task_uuid = str(uuid.uuid4())[:8]
            task_id = f"bandit_{i:04d}_{true_arm}_r{r}".replace(".", "")

            strategy_type, opt_strategy, r_boundary = best_strategy(priors, r, true_arm)
            task = {
                "task_id": f"{task_id}_{task_uuid}",
                "env": {
                    "num_arms": len(labels),
                    "labels": labels,
                    "priors": priors,
                    "true_arm": true_arm
                },
                "discount_factor": r,
                "optimal_policy": {
                    "strategy_type": strategy_type,
                    "optimal_strategy": opt_strategy,
                    "decision_boundary_r": r_boundary,
                },
            }

            tasks.append(task)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(tasks)} tasks and saved to {output_file}")
    print(f" - # arms: {len(labels)}")
    print(f" - priors: {priors}")
    print(f" - count per arm: {dict(zip(labels, counts))}")
    print(f" - discount factors: {r_list}")
    print(f" - timestamp (EST): {now_est}")


# Example usage
if __name__ == "__main__":
    priors = {"A": 0.2, "B": 0.3, "C": 0.5}
    r_values = [round(x / 10, 1) for x in range(0, 11)]  # [0.0, 0.1, ..., 1.0]
    sample_bandit_tasks(priors=priors, num_samples=10, r_list=r_values)
