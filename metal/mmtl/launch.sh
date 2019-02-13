# launches single task models of MetalModel

TASK=$1

# Default params
N_EPOCHS=3
BATCH_SIZE=32
SPLIT_PROP=0.8

if [ $TASK = "COLA" ]; then
    LR=1e-5
    L2=0
    BATCH_SIZE=8

elif [ $TASK = "SST2" ]; then
    LR=1e-5
    L2=0.01

elif [ $TASK = "MNLI" ]; then
    LR=1e-5
    L2=0
    BATCH_SIZE=2

elif [ $TASK = "RTE" ]; then
    LR=1e-5
    L2=0
    SPLIT_PROP=0.9
    N_EPOCHS=3

elif [ $TASK = "WNLI" ]; then
    LR=1e-4
    L2=0
    SPLIT_PROP=0.99
    BATCH_SIZE=32

elif [ $TASK = "QQP" ]; then
    LR=1e-5
    L2=0.01
    SPLIT_PROP=0.99
    BATCH_SIZE=32

elif [ $TASK = "MRPC" ]; then
    LR=1e-5
    L2=0.01

elif [ $TASK = "STSB" ]; then
    LR=1e-5
    L2=0
    BATCH_SIZE=2

elif [ $TASK = "QNLI" ]; then
    LR=1e-5
    L2=0.01

else
    echo "Task not found. Exiting."
    exit
fi

python launch.py \
    --device 0 \
    --bert_model bert-base-uncased \
    --bert_output_dim 768 \
    --max_len 200 \
    --lr_scheduler exponential \
    --log_every 0.25 --score_every 0.5 \
    --checkpoint_dir test_logs \
    --checkpoint_metric train/loss \
    --checkpoint_metric_mode min \
    --progress_bar 1 \
    --lr $LR \
    --l2 $L2 \
    --batch_size $BATCH_SIZE \
    --tasks $TASK \
    --split_prop $SPLIT_PROP \
    --n_epochs $N_EPOCHS
