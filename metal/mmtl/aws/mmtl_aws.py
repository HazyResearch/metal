rements:
# pip install boto3
# pip install paramiko
#
# Sample call:
# python mmtl_aws.py --mode launch_and_run --aws_access_key_id xxx --aws_secret_access_key xxx --keypath ~/.ssh/personalkeyncalifornia.pem
#
# Sample output:
# -------------------------------------------------------------------------
# Waiting for instances: ['i-0d593270be0c148ff', 'i-080af818a58200671']
# i-0563118bddda5433a, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-00602bf3a6e6a5573, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-00f2fc196f2d3d8a9, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0a042e6a6086c6788, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0d593270be0c148ff, state: running
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@13.56.189.108
# i-080af818a58200671, state: running
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@54.183.160.21
# i-09a2fe3e36bd9b94d, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0188aada6bbd5f7b3, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0cc22ecff87ff23c6, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-083d08c9cda3ba259, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0d19e505b62969224, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0e6262257804b27a9, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0947f67da4c3687a5, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-00c6094c5ec85f37a, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0ce6f416cde470dd7, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-05152c755e95630a2, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0ebcf50d148b2c8dc, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-077e7f4b17823ee5b, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0f50e252dc3cc253b, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# i-0657c027832a20f66, state: terminated
#       ssh -i /Users/agnusmaximus/.ssh/personalkeyncalifornia.pem ubuntu@None
# /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:39: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will be removed in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
#   m.add_string(self.Q_C.public_numbers().encode_point())
# /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:39: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will be removed in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
#   m.add_string(self.Q_C.public_numbers().encode_point())
# /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:96: CryptographyDeprecationWarning: Support for unsafe construction of public numbers from encoded data will be removed in a future version. Please use EllipticCurvePublicKey.from_encoded_point
#   self.curve, Q_S_bytes
# /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:111: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will be removed in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
#   hm.add_string(self.Q_C.public_numbers().encode_point())
# /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:96: CryptographyDeprecationWarning: Support for unsafe construction of public numbers from encoded data will be removed in a future version. Please use EllipticCurvePublicKey.from_encoded_point
#   self.curve, Q_S_bytes
# /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:111: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will be removed in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
#   hm.add_string(self.Q_C.public_numbers().encode_point())
# i-080af818a58200671 Added Snorkel MeTaL repository (/home/ubuntu/metal) to $PYTHONPATH.
# /home/ubuntu/metal
# [1.0 epo]: TRAIN:[loss=0.086] VALID:[foo_task/accuracy=0.983, bar_task/accuracy=0.988]
# Saving model at iteration 1.0 with best (max) score 0.983
# [2.0 epo]: TRAIN:[loss=0.111] VALID:[foo_task/accuracy=0.983, bar_task/accuracy=0.996]
# Restoring best model from iteration 1.0 with score 0.983
# Finished Training
# {'bar_task/valid/accuracy': 0.988125, 'foo_task/valid/accuracy': 0.98275}
# {'foo_task': tensor([[ -7.4493,   7.7396],
#         [ 52.5589, -57.5504],
#         [ 36.7535, -40.3470],
#         [  3.2167,  -3.8635]], grad_fn=<AddmmBackward>)}
# {'foo_task': tensor(0.0002, grad_fn=<NllLossBackward>)}
# {'foo_task': tensor([[2.5323e-07, 1.0000e+00],
#         [1.0000e+00, 0.0000e+00],
#         [1.0000e+00, 3.2782e-34],
#         [9.9916e-01, 8.4093e-04]])}
# SUCCESS!
#
# i-0d593270be0c148ff Added Snorkel MeTaL repository (/home/ubuntu/metal) to $PYTHONPATH.
# /home/ubuntu/metal
# [1.0 epo]: TRAIN:[loss=0.100] VALID:[foo_task/accuracy=0.981, bar_task/accuracy=0.984]
# Saving model at iteration 1.0 with best (max) score 0.981
# [2.0 epo]: TRAIN:[loss=0.119] VALID:[foo_task/accuracy=0.997, bar_task/accuracy=0.982]
# Saving model at iteration 2.0 with best (max) score 0.997
# Restoring best model from iteration 2.0 with score 0.997
# Finished Training
# {'bar_task/valid/accuracy': 0.981875, 'foo_task/valid/accuracy': 0.99725}
# {'foo_task': tensor([[-22.6154,  25.5793],
#         [ 25.4288, -27.5316],
#         [ -4.7540,   5.8662],
#         [ 17.6346, -18.9540]], grad_fn=<AddmmBackward>)}
# {'foo_task': tensor(6.0797e-06, grad_fn=<NllLossBackward>)}
# {'foo_task': tensor([[1.1730e-21, 1.0000e+00],
#         [1.0000e+00, 9.9904e-24],
#         [2.4418e-05, 9.9998e-01],
#         [1.0000e+00, 1.2876e-16]])}
# SUCCESS!
# --------------------------------------------------------------------------

import sys
import time
import multiprocessing
import os
import boto3
import argparse
import paramiko

IMAGE_ID = "ami-03da4643b88963008"

# Command prefix is appended to all commands.
# It activates pytorch gpu, clones metal, cds in metal
COMMAND_PREFIX = "source activate pytorch_p36;"\
    "rm -rf metal;"\
    "git clone -b mmtl_aws https://github.com/HazyResearch/metal.git;"\
    "cd metal; source add_to_path.sh;"\
    "pwd;"

COMMAND = "python metal/mmtl/aws_test.py"

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=["list", "launch", "launch_and_run", "run", "shutdown"])
parser.add_argument("--aws_access_key_id", required=True)
parser.add_argument("--aws_secret_access_key", required=True)
parser.add_argument("--region", default="us-west-1")
parser.add_argument("--n_machines", default=2)
parser.add_argument("--keypath", required=True)
parser.add_argument("--outputpath", default="output")

def run_command(args, instance, cmd, output):
    key = paramiko.RSAKey.from_private_key_file(args.keypath)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname=instance.public_ip_address, username="ubuntu", pkey=key)

        # Execute a command(cmd) after connecting/ssh to an instance
        stdin, stdout, stderr = client.exec_command(cmd)
        output[instance.id] = stdout.read().decode("utf-8")

        # close the client connection once the job is done
        client.close()

    except Exception as e:
        print(e)

    return output


def create_ec2_client(args):
    ec2_client = boto3.client(
        'ec2',
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.region
    )
    ec2_resource = boto3.resource(
        'ec2',
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.region
    )
    return ec2_client, ec2_resource

def get_instances(args):
    ec2_client, ec2_resource = create_ec2_client(args)
    return ec2_resource.instances.filter()

def describe_instances(args):
    instances = get_instances(args)
    for instance in instances:
        print("%s, state: %s\n\tssh -i %s ubuntu@%s" %
              (instance.id,
               instance.state["Name"],
               args.keypath,
               instance.public_ip_address
              ))

def launch(args):
    ec2_client, ec2_resource = create_ec2_client(args)
    instances = ec2_resource.create_instances(ImageId=IMAGE_ID,
                                              MinCount=args.n_machines,
                                              MaxCount=args.n_machines,
                                              KeyName=os.path.basename(args.keypath).split(".")[0])

    print("Waiting for instances: %s" % str([x.id for x in instances]))
    for instance in instances:
        instance.wait_until_running()
        instance.reload()

    # This is sometimes necessary to avoid ssh errors
    time.sleep(60)

    describe_instances(args)

    return instances

def shutdown(args):
    instances = get_instances(args)
    for instance in instances:
        instance.terminate()
    describe_instances(args)

def run(args, instances=None):

    # By default run command on all running machines
    if instances is None:
        instances = [x for x in get_instances(args) if x.state["Name"] == "running"]

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    processes = []
    manager = multiprocessing.Manager()
    return_output = manager.dict()
    for instance in instances:
        process = multiprocessing.Process(target=run_command,
                                          args=(args, instance,
                                                COMMAND_PREFIX+COMMAND,
                                                return_output))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    for k, v in return_output.items():
        print(k,v)
        with open(args.outputpath + "/" + k + ".out", "w") as f:
            f.write(v)

def launch_and_run(args):
    instances = launch(args)
    run(args, instances=instances)
    shutdown(args)

if __name__=="__main__":
    args = parser.parse_args()
    if args.mode == "list":
        list_instances(args)
    if args.mode == "launch":
        launch(args)
    if args.mode == "launch_and_run":
        launch_and_run(args)
    if args.mode == "run":
        run(args)
    if args.mode == "shutdown":
        shutdown(args)

