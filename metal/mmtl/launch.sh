python launch.py --device 0 --bert-model bert-base-uncased --bert-output-dim 768 --max-len 200 --batch-size 32 --lr 1e-5 --lr-freeze 1 --l2 0.01 --lr-scheduler exponential --log-every 0.25 --score-every 0.5 --checkpoint-dir test_logs --checkpoint-metric train/loss --checkpoint-metric-mode min --tasks COLA,SST2,MNLI,RTE,WNLI,QQP,MRPC,STSB,QNLI --n-epochs 5
