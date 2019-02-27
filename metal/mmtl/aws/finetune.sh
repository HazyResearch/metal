MODE=${1:-launch_and_run}

python metal/mmtl/aws/mmtl_aws.py --mode $MODE --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY --configpath metal/mmtl/aws/single_task_configs/COLA.py --n_machines 1 --commit_hash 901dcfbb716c4111e45198b522d5e3d38a6b1751 
