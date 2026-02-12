## 1 - /work/10672/wenxuand/vista/code-explorer/code_explorer/data/csv_explore/csv_sampler_instantiate.py
* build_model()


## run_baseline with discount range
* /work/10672/wenxuand/vista/code-explorer/code_explorer/scripts/run_eval_csv_independent_baseline_experiment.py

## 2. - /work/10672/wenxuand/vista/code-explorer/code_explorer/data/csv_explore/csv_explore_task_generator.py
* generate_task(2000)

## 3. train calibrator
/work/10672/wenxuand/vista/code-explorer/code_explorer/src/train/csv_calibrator/light_bert_format_predictor.py

-> /work/10672/wenxuand/vista/code-explorer/code_explorer/src/train/csv_calibrator/bert_format_model_twotext_hard_2epcs3e4.pt

## 4. /work/10672/wenxuand/vista/code-explorer/code_explorer/data/csv_explore/add_du_dc.py

## 5. /work/10672/wenxuand/vista/code-explorer/code_explorer/output_code_newest/plot_hypothetical_rhos.py
usage: 
  python plot_hypothetical_rhos.py \
    /scratch/10672/wenxuand/csv_explorer_data/csv_explorer_twotext_hard/csv_explore_tasks_test_processed_rho_0.5_1.5_2.5_3.5.json \
        --hypothetical-rhos 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 6.0 \
            -o /work/10672/wenxuand/vista/code-explorer/code_explorer/data/csv_explore/plots/hypothetic_hard.png 


/work/10672/wenxuand/vista/code-explorer/code_explorer/data/csv_explore/plots/hypothetic_easy.png
/work/10672/wenxuand/vista/code-explorer/code_explorer/data/csv_explore/plots/hypothetic_hard.png