MODE=${1:-launch_and_run}

python metal/mmtl/aws/mmtl_aws.py --mode $MODE --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.8xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY --configpath metal/mmtl/aws/single_task_configs/STSB.py --n_machines 1 --commit_hash e950f3b8e263db516e6928748c4c6de29c45a416 --ami ami-01b22f90823e1b2af --run_name test2
