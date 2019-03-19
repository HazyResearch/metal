# launches single task or multitask models of MetalModel
set -e -x

TASK=$1

# Default params
MIN_LR=0
N_EPOCHS=5
BATCH_SIZE=32
SPLIT_PROP=0.8
MAX_DATAPOINTS=-1
PROGRESS_BAR=1
CHECKPOINT_METRIC="model/train/loss"
CHECKPOINT_METRIC_MODE="min"

if [ $TASK = "COLA" ]; then
    LR=1e-5
    L2=0
    BATCH_SIZE=8
    CHECKPOINT_METRIC="COLA/COLA_valid/matthews_corr"
    CHECKPOINT_METRIC_MODE="max"

elif [ $TASK = "SST2" ]; then
    LR=1e-5
    L2=0.01
    CHECKPOINT_METRIC="SST2/SST2_valid/accuracy"
    CHECKPOINT_METRIC_MODE="max"

elif [ $TASK = "MNLI" ]; then
    LR=1e-5
    L2=0
    CHECKPOINT_METRIC="MNLI/MNLI_valid/accuracy"
    CHECKPOINT_METRIC_MODE="max"

elif [ $TASK = "RTE" ]; then
    LR=5e-5
    L2=0
    SPLIT_PROP=0.9
    CHECKPOINT_METRIC="RTE/RTE_valid/accuracy"
    CHECKPOINT_METRIC_MODE="max"

elif [ $TASK = "WNLI" ]; then
    LR=1e-4
    L2=0
    SPLIT_PROP=0.9
    BATCH_SIZE=32

elif [ $TASK = "QQP" ]; then
    LR=1e-5
    L2=0.01
    CHECKPOINT_METRIC="QQP/QQP_valid/accuracy"
    CHECKPOINT_METRIC_MODE="max"

elif [ $TASK = "MRPC" ]; then
    LR=1e-5
    L2=0.01

elif [ $TASK = "STSB" ]; then
    LR=1e-5
    L2=0
    BATCH_SIZE=8

elif [ $TASK = "QNLI" ]; then
    LR=1e-5
    L2=0.01
    CHECKPOINT_METRIC="QNLI/QNLI_valid/accuracy"
    CHECKPOINT_METRIC_MODE="max"

elif [ $TASK = "ALL" ]; then
    TASK="QNLI,STSB,MRPC,QQP,WNLI,RTE,MNLI,SST2,COLA"
    LR=1e-5
    L2=0.01

else
    echo "Task not found. Exiting."
    exit
fi

python launch.py \
    --device 0 \
    --bert_model bert-base-uncased \
    --max_len 200 \
    --warmup_steps 0.5 \
    --warmup_unit "epochs" \
    --lr_scheduler "linear" \
    --min_lr $MIN_LR \
    --log_every 0.25 --score_every 0.25 \
    --checkpoint_dir checkpoints \
    --checkpoint_metric $CHECKPOINT_METRIC \
    --checkpoint_metric_mode $CHECKPOINT_METRIC_MODE \
    --checkpoint_best 1 \
    --progress_bar $PROGRESS_BAR \
    --lr $LR \
    --l2 $L2 \
    --batch_size $BATCH_SIZE \
    --tasks $TASK \
    --split_prop $SPLIT_PROP \
    --n_epochs $N_EPOCHS \
    --max_datapoints $MAX_DATAPOINTS \
