# launches single task or multitask models of MetalModel with favorable GLUE parameters
# declare -a arr=("COLA" "SST2" "STSB" "MNLI" "MRPC" "QNLI" "QQP" "RTE")

# for task in "${arr[@]}"
# do
python launch.py \
    --device 0 \
    --fp16 1 \
    --bert_model bert-base-uncased \
    --max_len 200 \
    --warmup_unit "epochs" \
    --lr_scheduler "linear" \
    --min_lr 0 \
    --log_every 0.25 --score_every 0.25 \
    --checkpoint_dir checkpoints \
    --checkpoint_metric model/valid/gold/glue_score \
    --checkpoint_metric_mode max \
    --checkpoint_best True \
    --progress_bar 1 \
    --lr 5e-5 \
    --l2 0 \
    --batch_size 32 \
    --tasks STSB \
    --split_prop None \
    --n_epochs 3 \
    --max_datapoints 1000 \
    --model_weights /dfs/scratch0/mccreery/repos/metal/metal/mmtl/aws/output/2019_04_04_15_29_09/1/logdir/2019_04_05/QNLI,STSB,MRPC,QQP,WNLI,RTE,MNLI,SST2,COLA_19_58_22/best_model.pth
# done
