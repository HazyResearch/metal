BERT="bert-base-cased"

for TASK in COLA SST2 MNLI SNLI RTE WNLI QQP MRPC STSB QNLI
# for TASK in QQP MRPC STSB QNLI
do
    echo "Making dataset for task $TASK"
    python make_glue_datasets.py --task $TASK --bert_version $BERT
done