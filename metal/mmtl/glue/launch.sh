# launches single task or multitask models of MetalModel
# set -e -x

# TASK=$1

# Default params
#MIN_LR=0
#N_EPOCHS=1
#BATCH_SIZE=16
#SPLIT_PROP=0.8
#MAX_DATAPOINTS=100
#PROGRESS_BAR=1
#CHECKPOINT_METRIC="model/valid/glue"
#CHECKPOINT_METRIC_MODE="min"
#
#if [ $TASK = "COLA" ]; then
#     LR=1e-5
#    L2=0
#    BATCH_SIZE=8
#    CHECKPOINT_METRIC="COLA/COLA_valid/matthews_corr"
#    CHECKPOINT_METRIC_MODE="max"
#
#elif [ $TASK = "SST2" ]; then
#    LR=1e-5
#    L2=0.01
#    CHECKPOINT_METRIC="SST2/SST2_valid/accuracy"
#    CHECKPOINT_METRIC_MODE="max"
#
#elif [ $TASK = "MNLI" ]; then
#    LR=1e-5
#    L2=0
#    CHECKPOINT_METRIC="MNLI/MNLI_valid/accuracy"
#    CHECKPOINT_METRIC_MODE="max"
#
#elif [ $TASK = "RTE" ]; then
#    LR=5e-5
#    L2=0
#    SPLIT_PROP=0.9
#    CHECKPOINT_METRIC="RTE/RTE_valid/accuracy"
#    CHECKPOINT_METRIC_MODE="max"
#
#elif [ $TASK = "WNLI" ]; then
#    LR=1e-4
#    L2=0
#    SPLIT_PROP=0.9
#    BATCH_SIZE=32
#
#elif [ $TASK = "QQP" ]; then
#    LR=1e-5
#    L2=0.01
#    CHECKPOINT_METRIC="QQP/QQP_valid/accuracy"
#    CHECKPOINT_METRIC_MODE="max"
#
#elif [ $TASK = "MRPC" ]; then
#    LR=1e-5
#    L2=0.01
#
#elif [ $TASK = "STSB" ]; then
#    LR=1e-5
#    L2=0
#    BATCH_SIZE=8
#
#elif [ $TASK = "QNLI" ]; then
#    LR=1e-5
#    L2=0.01
#    CHECKPOINT_METRIC="QNLI/QNLI_valid/accuracy"
#    CHECKPOINT_METRIC_MODE="max"
#
#elif [ $TASK = "ALL" ]; then
#    TASK="QNLI,STSB,MRPC,QQP,WNLI,RTE,MNLI,SST2,COLA"
#    LR=1e-5
#    L2=0.01
#
#else
#    echo "Task not found. Exiting."
#    exit
#"""
#
python launch.py \
    --device 0 \
    --bert_model bert-base-uncased \
    --max_len 200 \
    --warmup_steps 0 \
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
    --l2 1e-4 \
    --batch_size 16 \
    --tasks COLA \
    --split_prop None \
    --n_epochs 2 \
    --max_datapoints 1000 \
