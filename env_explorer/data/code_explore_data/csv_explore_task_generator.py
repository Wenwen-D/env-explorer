'''
Script to generate synthetic CSV exploration tasks with various features.
Each task involves creating a CSV file with specific formatting and content,
and defining a reasoning task based on the CSV data.
'''

import itertools
import json
import random
from pathlib import Path
from faker import Faker
from csv_sampler_instantiate import build_model_more, build_model_independent, build_model_independent_hard
from content_sampler import CSVContentSampler, CSVContentSampler_independent
import uuid
import pandas as pd
from tqdm import tqdm
import re
from scipy.stats import truncnorm

def generate_short_uuid(max_length=6):
    return str(uuid.uuid4())[:max_length]

def generate_all_combinations(features):
    """Return all 2^len(features) binary combinations as dicts."""
    keys = list(features)
    combos = list(itertools.product([0, 1], repeat=len(keys)))
    return [{k: bool(v) for k, v in zip(keys, combo)} for combo in combos]

def generate_filename_suffix(feature_combo, feature_dict):
    """Generate a filename suffix given a feature combination."""
    active_feats = [k for k, v in feature_combo.items() if v]
    suffix_parts = []
    for feat in active_feats:
        tokens = feature_dict.get(feat, [feat])
        token = random.choice(tokens)
        suffix_parts.append(f"_{token}")
    random.shuffle(suffix_parts)

    # Choose extension: if has_tsv -> .tsv else .csv
    ext = ".tsv" if feature_combo.get("has_tsv") else ".csv"
    return "".join(suffix_parts) + ext

def generate_filename(fake, feature_combo, feature_dict):
    """Generate full filename with random prefix and feature-based suffix."""
    prefix = Path(fake.file_name(extension='')).stem
    suffix = generate_filename_suffix(feature_combo, feature_dict)
    return prefix + suffix

def assign_task_counts(n_tasks, n_combos):
    """Roughly distribute n_tasks among all combinations."""
    base = n_tasks // n_combos
    rem = n_tasks % n_combos
    counts = [base + (1 if i < rem else 0) for i in range(n_combos)]
    return counts

def generate_tasks(base_path, total_tasks=1000):
    """Main orchestrator: generate synthetic CSV exploration tasks."""
    fake = Faker()
    model, feature_dict = build_model_more()
    feature_names = list(feature_dict.keys())
    combos = generate_all_combinations(feature_names)
    counts = assign_task_counts(total_tasks, len(combos))

    tasks = []
    # task_id = 0
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for combo_idx, (combo, n_each) in enumerate(zip(combos, counts), 1):
        combo_desc = ", ".join([f"{k}={int(v)}" for k, v in combo.items()])
        for _ in tqdm(
                    range(n_each),
                    desc=f"[{combo_idx}/{len(combos)}] {combo_desc}",
                    leave=False,
                    ncols=90
                ):
            task_id = generate_short_uuid(6)
            # Step 1: build filename
            filename = generate_filename(fake, combo, feature_dict)

            # Step 2: compute model priors and sample formats
            feats, probs = model.predict_all(filename)
            feats, sampled = model.sample_all(filename)

            # Step 3: instantiate a CSV content sampler
            task_dir = base_path / f"task_csv_{task_id}"
            task_dir.mkdir(parents=True, exist_ok=True)
            file_path = task_dir / filename
            relative_path = file_path.relative_to(base_path)

            sampler = CSVContentSampler(
                delimiter=sampled["Delimiter"],
                quotechar=sampled["Quotechar"],
                encoding=sampled["Encoding"],
                path=str(file_path)
            )
            _, meta_info = sampler.write_csv()

            # Step 4: store full record
            record = {
                "task_id": task_id,
                "filename": filename,
                "features": combo,
                "priors": {name: probs[name] for name in probs},
                "sampled_format": sampled,
                "meta_info": meta_info,
                "path": str(relative_path),
            }
            tasks.append(record)

    # Step 5: store all records
    output_path = base_path / "csv_explore_tasks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Generated {len(tasks)} tasks → {output_path}")
    return tasks, output_path

def generate_tasks_independent(base_path, total_tasks=1000):
    """Main orchestrator: generate synthetic CSV exploration tasks."""
    fake = Faker()
    model, feature_dict = build_model_independent()
    feature_names = list(feature_dict.keys())
    combos = generate_all_combinations(feature_names)
    counts = assign_task_counts(total_tasks, len(combos))

    tasks = []
    # task_id = 0
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for combo_idx, (combo, n_each) in enumerate(zip(combos, counts), 1):
        combo_desc = ", ".join([f"{k}={int(v)}" for k, v in combo.items()])
        for _ in tqdm(
                    range(n_each),
                    desc=f"[{combo_idx}/{len(combos)}] {combo_desc}",
                    leave=False,
                    ncols=90
                ):
            task_id = generate_short_uuid(6)
            # Step 1: build filename
            filename = generate_filename(fake, combo, feature_dict)

            # Step 2: compute model priors and sample formats
            feats, probs = model.predict_all(filename)
            feats, sampled = model.sample_all(filename)

            # Step 3: instantiate a CSV content sampler
            task_dir = base_path / f"task_csv_{task_id}"
            task_dir.mkdir(parents=True, exist_ok=True)
            file_path = task_dir / filename
            relative_path = file_path.relative_to(base_path)

            sampler = CSVContentSampler_independent(
                delimiter=sampled["Delimiter"],
                quotechar=sampled["Quotechar"],
                skiprows=sampled["Skiprows"],
                path=str(file_path)
            )
            _, meta_info = sampler.write_csv()

            # Step 4: store full record
            record = {
                "task_id": task_id,
                "filename": filename,
                "features": combo,
                "priors": {name: probs[name] for name in probs},
                "sampled_format": sampled,
                "meta_info": meta_info,
                "path": str(relative_path),
            }
            tasks.append(record)

    # Step 5: store all records
    output_path = base_path / "csv_explore_tasks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Generated {len(tasks)} tasks → {output_path}")
    return tasks, output_path

def generate_tasks_independent_hard(base_path, total_tasks=1000):
    """Main orchestrator: generate synthetic CSV exploration tasks."""
    fake = Faker()
    model, feature_dict = build_model_independent_hard()
    feature_names = list(feature_dict.keys())
    combos = generate_all_combinations(feature_names)
    counts = assign_task_counts(total_tasks, len(combos))

    tasks = []
    # task_id = 0
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for combo_idx, (combo, n_each) in enumerate(zip(combos, counts), 1):
        combo_desc = ", ".join([f"{k}={int(v)}" for k, v in combo.items()])
        for _ in tqdm(
                    range(n_each),
                    desc=f"[{combo_idx}/{len(combos)}] {combo_desc}",
                    leave=False,
                    ncols=90
                ):
            task_id = generate_short_uuid(6)
            # Step 1: build filename
            filename = generate_filename(fake, combo, feature_dict)

            # Step 2: compute model priors and sample formats
            feats, probs = model.predict_all(filename)
            feats, sampled = model.sample_all(filename)

            # Step 3: instantiate a CSV content sampler
            task_dir = base_path / f"task_csv_{task_id}"
            task_dir.mkdir(parents=True, exist_ok=True)
            file_path = task_dir / filename
            relative_path = file_path.relative_to(base_path)

            sampler = CSVContentSampler_independent(
                delimiter=sampled["Delimiter"],
                quotechar=sampled["Quotechar"],
                skiprows=sampled["Skiprows"],
                path=str(file_path)
            )
            _, meta_info = sampler.write_csv()

            # Step 4: store full record
            record = {
                "task_id": task_id,
                "filename": filename,
                "features": combo,
                "priors": {name: probs[name] for name in probs},
                "sampled_format": sampled,
                "meta_info": meta_info,
                "path": str(relative_path),
            }
            tasks.append(record)

    # Step 5: store all records
    output_path = base_path / "csv_explore_tasks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Generated {len(tasks)} tasks → {output_path}")
    return tasks, output_path


def add_discount_factor_to_tasks(path_to_json, discount_range):
    """
    Add discount factors to tasks based on different sampling strategies.

    Args:
        path_to_json: Path to the JSON file containing tasks
        discount_range: Can be one of:
            - List of discrete choices: [0.5, 0.7, 0.9, 1.0]
            - Tuple for uniform sampling: (0.5, 0.9)
            - Dict for distribution sampling:
                * Beta distribution: {"type": "beta", "alpha": 5, "beta": 2, "min": 0.5, "max": 1.0}
                * Multi-Gaussian: {"type": "multi", "centers": [0.55, 0.85, 0.95], "std": 0.05, "min": 0.5, "max": 1.0}

    Returns:
        List of tasks with discount factors added
    """
    with open(path_to_json, "r") as f:
        data = json.load(f)

    if discount_range is None:
        raise ValueError("discount_range must be provided.")

    # Print configuration
    if isinstance(discount_range, dict):
        if discount_range.get("type") == "beta":
            print(f"Using Beta distribution: α={discount_range['alpha']}, β={discount_range['beta']}, "
                  f"range=[{discount_range.get('min', 0.0)}, {discount_range.get('max', 1.0)}]")
        elif discount_range.get("type") == "multi":
            print(f"Using Multi-Gaussian distribution: centers={discount_range.get('centers')}, "
                  f"std={discount_range.get('std')}, range=[{discount_range.get('min', 0.0)}, {discount_range.get('max', 1.0)}]")
    elif isinstance(discount_range, list):
        print(f"Using discrete choices: {discount_range}")
    else:
        print(f"Using uniform sampling in range: {discount_range}")

    task_with_discount = []

    for item in data:
        new_item = item.copy()

        if isinstance(discount_range, dict):
            # Distribution-based sampling
            if discount_range.get("type") == "beta":
                # Beta distribution
                alpha = discount_range["alpha"]
                beta_param = discount_range["beta"]
                min_val = discount_range.get("min", 0.0)
                max_val = discount_range.get("max", 1.0)
                # Sample from beta distribution and scale to [min_val, max_val]
                beta_sample = random.betavariate(alpha, beta_param)
                dd = min_val + beta_sample * (max_val - min_val)
                dd = round(dd, 2)

            elif discount_range.get("type") == "multi":
                # Multi-Gaussian (mixture of Gaussians with truncation)
                centers = discount_range.get("centers", [])
                std = discount_range.get("std", 0.0)
                min_val = discount_range.get("min", 0.0)
                max_val = discount_range.get("max", 1.0)

                # Randomly select one of the centers
                chosen_center = random.choice(centers)

                # Calculate standardized bounds for truncnorm
                a = (min_val - chosen_center) / std
                b = (max_val - chosen_center) / std
                sample = truncnorm.rvs(a, b, loc=chosen_center, scale=std)

                # Round to two decimal places
                dd = float(f"{sample:.2f}")

                # Validate bounds
                if dd < min_val or dd > max_val:
                    raise ValueError(f"Sampled discount factor {dd} out of bounds [{min_val}, {max_val}]")
            else:
                raise ValueError(f"Unknown discount_range type: {discount_range.get('type')}")

        elif isinstance(discount_range, list):
            # Discrete choice sampling
            dd = random.choice(discount_range)

        else:
            # Uniform sampling from range (tuple)
            dd = round(random.uniform(discount_range[0], discount_range[1]), 2)

        new_item['discount_factor'] = dd
        task_with_discount.append(new_item)

    print(f"Original tasks: {len(data)}, After adding discount factors: {len(task_with_discount)}")
    return task_with_discount


def follow_instruction(type, instruction, df):
    def extract_backticks(text):
        pattern = r"`([^`]+)`"
        return re.findall(pattern, text)

    # 1. Handle "max" - Returns the id_key value where number_key is highest
    if type == "max":
        id_key, number_key = extract_backticks(instruction)
        
        # Convert the dataframe to a list of dictionaries for easier manual iteration
        records = df.to_dict('records')
        
        # Filter out rows where the number column is empty/NaN
        valid_records = [r for r in records if r[number_key] == r[number_key] and r[number_key] is not None]
        
        if not valid_records:
            return None
            
        # Find the row with the maximum value (converting strings to float)
        best_record = max(valid_records, key=lambda x: float(x[number_key]))
        return best_record[id_key]

    # 2. Handle "mean" - Manual calculation
    elif type == "mean":
        number_key, = extract_backticks(instruction)
        
        # Pull the column values, filtering out NaNs
        # (x == x is a trick to check if a value is NOT NaN in Python)
        try:
            values = [float(x) for x in df[number_key] if x == x and x is not None]
        except Exception as e:
            print(df.head())
            print(f"Error converting values to float for key {number_key}: {e}")
            raise e
            return None
            
        return sum(values) / len(values)

    # 3. Handle "min" - Manual minimum
    elif type == "min":
        number_key, = extract_backticks(instruction)
        try:
            values = [float(x) for x in df[number_key] if x == x and x is not None]
        except Exception as e:
            print(df.head())
            print(f"Error converting values to float for key {number_key}: {e}")
            raise e
            return None
        
        return min(values) if values else None

    else:
        raise ValueError(f"Unknown type: {type}")

def add_instruction_answer_to_tasks(tasks, base_path):
    data_new = []
    for item in tasks:
        task = random.choice([
            ("Find the `{id_key}` of the record with the maximum `{number_key}`.", "max"),
            ("Compute the average of `{number_key}` (excluding None).", "mean"),
            ("Find the minimum of `{number_key}` (excluding None).", "min"),
        ])
        task_desc = task[0].format(
            id_key=item["meta_info"]["key_names"]["id_keys"][0],
            number_key=random.choice(item["meta_info"]["key_names"]["number_keys"]),
        )
        item["task_instruction"] = (task_desc, task[1])

        csv_path = base_path + "/" + item["path"]

        csv_data = pd.read_csv(csv_path,
                        encoding=item['meta_info'].get('encoding', 'utf-8'),
                        sep=item['meta_info']['delimiter'],
                        quotechar=item['meta_info']['quotechar'],
                        skiprows=int(item['meta_info'].get('skiprows', 0)),
                        header=0)
        ans = follow_instruction(item["task_instruction"][1], item["task_instruction"][0], csv_data)
        item["answer"] = ans
        data_new.append(item)
    return data_new
    

def split_tasks(tasks, train_ratio=0.7, val_ratio=0.15, base_path):
    random.shuffle(tasks)
    n_total = len(tasks)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_tasks = tasks[:n_train]
    val_tasks = tasks[n_train:n_train + n_val]
    test_tasks = tasks[n_train + n_val:]

    print(f"Train: {len(train_tasks)}, Val: {len(val_tasks)}, Test: {len(test_tasks)}")

    base_path = Path(base_path)
    test_path = base_path / "csv_explore_tasks_test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_tasks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Generated {len(test_tasks)} tasks → {test_path}")

    val_path = base_path / "csv_explore_tasks_val.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_tasks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Generated {len(val_tasks)} tasks → {val_path}")
    train_path = base_path / "csv_explore_tasks_train.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_tasks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Generated {len(train_tasks)} tasks → {train_path}")
    return train_tasks, val_tasks, test_tasks

if __name__ == "__main__":
    # generate tasks with independent features: delimiter, quotechar, skiprows
    generate_tasks_independent_hard(base_path="./code_explorer_balanced_data", total_tasks=2000)
    tasks = add_discount_factor_to_tasks("./code_explorer_balanced_data/csv_explore_tasks.json", discount_range=(0.35, 0.95))
    tasks_with_answers = add_instruction_answer_to_tasks(tasks, base_path="./code_explorer_balanced_data")
    split_tasks(tasks_with_answers, base_path="./code_explorer_balanced_data")

