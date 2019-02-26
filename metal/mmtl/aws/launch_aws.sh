# NOTE: the following should be included in your path after `source activate_mmtl.sh`
# Uncomment to confirm.

# echo $AWS_ACCESS_KEY_ID
# echo $AWS_SECRET_ACCESS_KEY_ID

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/COLA_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/SST2_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/MNLI_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/RTE_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/WNLI_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/QQP_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/MRPC_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/STSB_dict.py --n_machines 1 --n_trials 1;

python metal/mmtl/aws/mmtl_aws.py --mode run --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY_ID --configpath metal/mmtl/aws/single_task_configs/QNLI_dict.py --n_machines 1 --n_trials 1;
