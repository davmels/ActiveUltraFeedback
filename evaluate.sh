# launch from evals-post-train repository.

bash launcher_graceful.sh \
  --base_dir /iopsstor/scratch/cscs/dmelikidze/models/dpo_new/parts2704/ \
  --config_file ./configs/apertus/tasks_thesis.txt \
  --table_metrics ./configs/apertus/tasks_thesis_main_table.txt \
  --wandb_entity "ActiveUF_Plus" \
  --wandb_project "evals" \
  --logs_root "$SCRATCH/eval_logs_start/ActiveUF_Plus/evals/" --group_size 6
