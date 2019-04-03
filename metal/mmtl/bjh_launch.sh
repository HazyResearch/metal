python launch.py --tasks RTE \
--seed 123 \
--bert_model bert-base-uncased \
--max_len 200 \
--warmup_steps 0.5 \
--warmup_unit "epochs" \
--lr_scheduler "linear" \
--min_lr 1e-6 \
--optimizer adam \
--lr 5e-5 \
--checkpoint_metric="RTE/RTE_valid/accuracy" \
--checkpoint_metric_mode="max" \
--split_prop 0.8 \
--n_epochs=5 \
--batch_size=32 \
--score_every 0.2 \
--log_every 0.05

# Recreate original RTE seed=123 runs: (new log @ 2019_03_20/recreate2_13_51_14)
# python launch.py --tasks RTE --seed 123 --bert_model bert-base-uncased --max_len 200 --warmup_steps 0.5 --warmup_unit "epochs" --lr_scheduler "linear" --min_lr 1e-6 --optimizer adam --lr 5e-5 --checkpoint_metric="RTE/RTE_valid/accuracy" --checkpoint_metric_mode="max" --split_prop 0.8 --n_epochs=5 --batch_size=32 --score_every 0.2 --log_every 0.05 --run_name recreate2