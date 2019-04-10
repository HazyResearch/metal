# launches single task or multitask models of MetalModel with favorable GLUE parameters
declare -a arr=("COLA" "SST2" "RTE" "MRPC" "MNLI" "QNLI" "QQP")

for task in "${arr[@]}"
do
python launch.py \
    --device 0 \
    --fp16 0 \
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
    --attention 1 \
    --lr 5e-5 \
    --l2 0 \
    --batch_size 16 \
    --tasks $task \
    --split_prop None \
    --n_epochs 5 \
    --max_datapoints -1 \
    --model_weights /dfs/scratch0/mccreery/repos/metal/metal/mmtl/aws/output/2019_04_08_17_04_03/0/logdir/2019_04_09/QNLI,STSB,MRPC,QQP,WNLI,RTE,MNLI,SST2,COLA_00_21_20/best_model.pth
done
