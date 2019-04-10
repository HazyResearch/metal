MODE=${1:-launch_and_run}
TASK=${2:-COLA}

python $METALHOME/metal/mmtl/aws/mmtl_aws.py --mode $MODE --keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt --instance_type p3.2xlarge  --aws_access_key_id $AWS_ACCESS_KEY_ID --aws_secret_access_key $AWS_SECRET_ACCESS_KEY --configpath $METALHOME/metal/mmtl/aws/single_task_configs/$TASK.py --n_machines 1 --commit_hash 54caf8ed517cb053b4057fa07b079e423c18591b --ami ami-01b22f90823e1b2af --run_name $TASK
