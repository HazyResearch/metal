MODE=${1:-launch_and_run}

python metal/mmtl/aws/mmtl_aws.py --mode $MODE --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY --configpath metal/mmtl/aws/single_task_configs/MNLI.py --n_machines 4 --commit_hash cc3070b6001841f71fbcc6daf5896642d0280e7d

