# requirements:
# pip install boto3
# pip install paramiko
#
# Sample call:
# python mmtl_aws.py --mode launch_and_run --aws_access_key_id xxx --aws_secret_access_key xxx --keypath ~/.ssh/personalkeyncalifornia.pem
#
# Sample output:
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
# Putting file secret -> secret
# Putting file secret -> secret
# Getting file secret -> output/i-04c1e57c76aaff218/yoman
# Getting file secret -> output/i-09b38dd9e71d3474a/yoman
# i-04c1e57c76aaff218 testing 42
# i-09b38dd9e71d3474a testing 42
#
# --------------------------------------------------------------------------

import argparse
import inspect
import multiprocessing
import os
import sys
import time

import boto3
import paramiko

from metal.mmtl.aws import grid_search_mmtl

IMAGE_ID = "ami-0c82a5c425d9da154"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["list", "launch", "launch_and_run", "run", "shutdown"]
)
parser.add_argument("--aws_access_key_id", required=True)
parser.add_argument("--aws_secret_access_key", required=True)
parser.add_argument("--region", default="us-west-1")
parser.add_argument("--n_machines", default=2)
parser.add_argument("--keypath", required=True)
parser.add_argument("--outputpath", default="output")


def create_dummy_command_dict2():
    COMMAND_PREFIX = (
        "source activate pytorch_p36;"
        "rm -rf metal;"
        "git clone -b mmtl_aws https://github.com/HazyResearch/metal.git;"
        "cd metal; source add_to_path.sh;"
        "pwd;"
    )
    COMMAND = "python metal/mmtl/aws_test.py"

    return {"cmd": COMMAND_PREFIX + COMMAND, "files_to_put": [], "files_to_get": []}


def create_dummy_command_dict():

    COMMAND_PREFIX = ""
    COMMAND = "cat secret"

    return {
        "cmd": COMMAND_PREFIX + COMMAND,
        "files_to_put": [("secret", "secret")],
        "files_to_get": [("secret", "yoman")],
    }


def run_command(args, instance, cmd_dict, output, run_id):
    key = paramiko.RSAKey.from_private_key_file(args.keypath)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    cmd = cmd_dict["cmd"]
    files_to_put = cmd_dict["files_to_put"]
    files_to_get = cmd_dict["files_to_get"]

    assert os.path.exists(args.outputpath)
    instance_output_path = "%s/%s/" % (args.outputpath, run_id)
    if not os.path.exists(instance_output_path):
        os.makedirs(instance_output_path)

    try:
        client.connect(
            hostname=instance["public_ip_address"], username="ubuntu", pkey=key
        )

        # Put files
        sftp = client.open_sftp()
        for (localpath, remotepath) in files_to_put:
            print("Putting file %s -> %s" % (localpath, remotepath))
            sftp.put(localpath, remotepath)

        # Execute a command(cmd) after connecting/ssh to an instance
        stdin, stdout, stderr = client.exec_command(cmd)
        output[run_id] = stdout.read().decode("utf-8")

        print("ERROR: ", stderr.read())

        # Get files
        for (remotepath, localpath) in files_to_get:
            print(
                "Getting file %s -> %s" % (remotepath, instance_output_path + localpath)
            )
            sftp.get(remotepath, instance_output_path + localpath)

        # close the client connection once the job is done
        sftp.close()
        client.close()

    except Exception as e:
        print(e)

    return output


def create_ec2_client(args):
    ec2_client = boto3.client(
        "ec2",
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.region,
    )
    ec2_resource = boto3.resource(
        "ec2",
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.region,
    )
    return ec2_client, ec2_resource


def get_instances(args):
    ec2_client, ec2_resource = create_ec2_client(args)
    return ec2_resource.instances.filter()


def describe_instances(args):
    instances = get_instances(args)
    for instance in instances:
        print(
            "%s, state: %s\n\tssh -i %s ubuntu@%s"
            % (
                instance.id,
                instance.state["Name"],
                args.keypath,
                instance.public_ip_address,
            )
        )


def launch(args):
    ec2_client, ec2_resource = create_ec2_client(args)
    instances = ec2_resource.create_instances(
        ImageId=IMAGE_ID,
        MinCount=args.n_machines,
        MaxCount=args.n_machines,
        KeyName=os.path.basename(args.keypath).split(".")[0],
    )

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


def worker_job(
    args, instance_dicts, command_dict, return_output, process_id_mapping, run_id
):

    # Get process id and map to instance
    current = str(multiprocessing.current_process())
    wid = process_id_mapping[current]
    instance_dict = instance_dicts[wid]

    # Run command on instance
    run_command(args, instance_dict, command_dict, return_output, run_id)


def initialize_process_ids(wid, process_id_mapping):
    current = str(multiprocessing.current_process())
    process_id_mapping[current] = wid


def run(args, instances=None):

    # By default run command on all running machines
    if instances is None:
        instances = [x for x in get_instances(args) if x.state["Name"] == "running"]

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    manager = multiprocessing.Manager()

    # Generate commands
    return_output = manager.dict()
    process_id_mapping = manager.dict()
    command_dicts = grid_search_mmtl.generate_configs_and_commands(args)
    instance_dicts = [
        {str(k): str(v) for k, v in dict(inspect.getmembers(instance)).items()}
        for instance in instances
    ]
    data = [
        (args, instance_dicts, x, return_output, process_id_mapping, i)
        for i, x in enumerate(command_dicts)
    ]

    p = multiprocessing.Pool(args.n_machines)

    # Map processes to indices in the range 0 to n (todo(maxlam): DANGEROUS)
    initialization_args = [
        (x, process_id_mapping) for x in list(range(args.n_machines))
    ]
    [p.apply(initialize_process_ids, x) for x in initialization_args]

    # Main work
    p.starmap(worker_job, data)

    print("Results")
    print("-" * 100)
    for k, v in return_output.items():
        print(k, v)
        with open(args.outputpath + "/" + str(k) + ".out", "w") as f:
            f.write(str(v))


def launch_and_run(args):
    instances = launch(args)
    run(args, instances=instances)
    shutdown(args)


if __name__ == "__main__":

    args = parser.parse_args()

    if args.mode == "list":
        describe_instances(args)
    if args.mode == "launch":
        launch(args)
    if args.mode == "launch_and_run":
        launch_and_run(args)
    if args.mode == "run":
        run(args)
    if args.mode == "shutdown":
        shutdown(args)
