python launch.py \
--tasks STSB \
--run_name STSB \
--bert_model bert-base-uncased \
--auxiliary_loss_multiplier 0.1 \
--seed 1 \
--lr 1e-5 \
--batch_size 16 \
--n_epochs 10 \
--log_every 0.02 \
--score_every 0.1 \
--optimizer adamax \
--warmup_steps 0.5 \
--warmup_unit epochs \
--max_len 200 \
--checkpoint_metric model/valid/glue \
--checkpoint_metric_mode max

# COLA,SST2,MRPC,STSB,QQP,MNLI,QNLI,RTE