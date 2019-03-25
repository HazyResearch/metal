MODE=${1:-launch_and_run}

echo "Remember to use the correct commit hash."

python $METALHOME/metal/mmtl/aws/mmtl_aws.py \
--mode $MODE \
--keypath /dfs/scratch0/maxlam/mmtl_share/mmtl_shared_east.pem.txt \
--instance_type p3.2xlarge  \
--ami ami-0074843706ba1e677 \
--aws_access_key_id $AWS_ACCESS_KEY_ID \
--aws_secret_access_key $AWS_SECRET_ACCESS_KEY \
--commit_hash ??? \
--n_machines 1 \
--configpath $METALHOME/metal/mmtl/aws/bjh_config.py

# ARE YOU SURE YOU WANT TO USE THAT COMMIT AND AMI?

# A random garbage hash:
# 868fb01a48e4031b8f1ac568e3bd0b413c904541