MODE=${1:-launch_and_run}
TASK=${2:-COLA}

python metal/mmtl/aws/mmtl_aws.py --mode $MODE --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.16xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY --configpath metal/mmtl/aws/NLI.py --n_machines 2 --commit_hash 27982edea447c2878dd8a99fccf090d08ddac9ad --ami ami-0074843706ba1e677 --run_name $TASK
