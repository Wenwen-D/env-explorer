#!/usr/bin/env python3
"""
Script to augment JSON data with different rho values.

For each item in the input JSON file, creates multiple copies with different
d_unit and d_code values based on the provided rho list.
"""

import json
import argparse
import random
from typing import List
import torch
from src.train.csv_calibrator.light_bert_format_predictor import load_data, FilenameFormatDataset, FormatPredictorTinyBERT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_json(input_file: str, output_file: str, rho_list: List[float], sample: bool = False, calibrator_model_path=None) -> None:
    """
    Process JSON file and create copies with different rho values.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        rho_list: List of rho values to use for creating copies
        sample: If True, sample one rho per item instead of creating copies for all rhos
    """

    if calibrator_model_path is not None:
        calibrator_model = FormatPredictorTinyBERT("prajjwal1/bert-tiny")
        state = torch.load(calibrator_model_path, map_location=DEVICE)
        calibrator_model.load_state_dict(state)
        calibrator_model.to(DEVICE)
        calibrator_model.eval()

    # Read input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each item
    output_data = []
    for item in data:
        # Get original values
        original_task_id = item['task_id']
        discount_factor = item.get('discount_factor', 1.0)

        # Determine which rho values to use
        if sample:
            # Sample one rho value randomly
            rhos_to_use = [random.choice(rho_list)]
        else:
            # Use all rho values
            rhos_to_use = rho_list

        # Create a copy for each rho value
        for rho in rhos_to_use:
            # Deep copy the item
            new_item = json.loads(json.dumps(item))

            # Set new values
            new_item['d_unit'] = discount_factor
            new_item['d_code'] = discount_factor ** rho
            new_item['task_id'] = f"{original_task_id}_{rho}"
            new_item['task_id_original'] = original_task_id
            new_item['rho'] = rho
            if calibrator_model_path is not None:
                pred_prior_info = calibrator_model.predict_one(new_item['filename'])
                new_item['calib_pred_prior'] = (
                    f"delimiter: {pred_prior_info['sep']}, "
                    f"quotechar: {pred_prior_info['quote']}, "
                    f"skiprows: {pred_prior_info['skiprows']}"
                )

            output_data.append(new_item)

    # Write output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(data)} items into {len(output_data)} items")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process JSON file and create copies with different rho values'
    )
    parser.add_argument(
        'input_file',
        help='Path to input JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Path to output JSON file (default: input_file with _processed suffix)'
    )
    parser.add_argument(
        '-r', '--rho',
        nargs='+',
        type=float,
        default=[0.3, 0.8, 1.0, 1.5, 3.0],
        help='List of rho values (default: 0.3 0.8 1.0 1.5 3.0)'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Sample one rho per item instead of creating copies for all rhos'
    )
    parser.add_argument(
        '--calibrator_path',
        default=None,
        help='Path to calibrator model (default: bert_format_model_twotext_1epcs1e4.pt)'
    )

    args = parser.parse_args()

    # Determine output filename
    sample_suffix = '_sampled' if args.sample else ''
    rho_suffix = '_rho_' + '_'.join(str(r) for r in args.rho)
    if args.output is None:
        if args.input_file.endswith('.json'):
            output_file = args.input_file[:-5] + f'_processed{sample_suffix}{rho_suffix}.json'
        else:
            output_file = args.input_file + f'_processed{sample_suffix}{rho_suffix}.json'
    else:
        output_file = args.output

    process_json(args.input_file, output_file, args.rho, args.sample, args.calibrator_path)


if __name__ == '__main__':
    main()



#  python add_du_dc.py ./code_explorer_balanced_data/csv_explore_tasks_test.json --rho 0.5 1 2 4 --calibrator_path ../../src/train/csv_calibrator/bert_format_model.pt
#  python add_du_dc.py ./code_explorer_balanced_data/csv_explore_tasks_train.json --rho 0.5 1 2 4 --calibrator_path ../../src/train/csv_calibrator/bert_format_model.pt
#  python add_du_dc.py ./code_explorer_balanced_data/csv_explore_tasks_val.json --rho 0.5 1 2 4 --sample --calibrator_path ../../src/train/csv_calibrator/bert_format_model.pt