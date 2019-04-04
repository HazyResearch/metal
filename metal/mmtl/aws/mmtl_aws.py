"""
 requirements:
 pip install boto3
 pip install paramiko

 Sample call:
 python mmtl_aws.py --mode launch_and_run --aws_access_key_id xxx --aws_secret_access_key xxx --keypath ~/.ssh/personalkeyncalifornia.pem --config aws_config

 Sample output:
 /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:39: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will be emoved in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
   m.add_string(self.Q_C.public_numbers().encode_point())
 /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:39: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will be emoved in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
   m.add_string(self.Q_C.public_numbers().encode_point())
 /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:96: CryptographyDeprecationWarning: Support for unsafe construction of public numbers from encoded data will beremoved in a future version. Please use EllipticCurvePublicKey.from_encoded_point
   self.curve, Q_S_bytes
 /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:111: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will beremoved in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
   hm.add_string(self.Q_C.public_numbers().encode_point())
 /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:96: CryptographyDeprecationWarning: Support for unsafe construction of public numbers from encoded data will beremoved in a future version. Please use EllipticCurvePublicKey.from_encoded_point
   self.curve, Q_S_bytes
 /usr/local/lib/python3.6/site-packages/paramiko/kex_ecdh_nist.py:111: CryptographyDeprecationWarning: encode_point has been deprecated on EllipticCurvePublicNumbers and will beremoved in a future version. Please use EllipticCurvePublicKey.public_bytes to obtain both compressed and uncompressed point encoding.
   hm.add_string(self.Q_C.public_numbers().encode_point())
 Putting file secret -> secret
 Putting file secret -> secret
 Getting file secret -> output/i-04c1e57c76aaff218/yoman
 Getting file secret -> output/i-09b38dd9e71d3474a/yoman
 i-04c1e57c76aaff218 testing 42
 i-09b38dd9e71d3474a testing 42
"""

import argparse
import datetime
import importlib
import inspect
import multiprocessing
import os
import sys
import time
from stat import S_ISDIR

import boto3
import paramiko

from metal.mmtl.aws import grid_search_mmtl

# IMAGE_ID = "ami-0c82a5c425d9da154"  # For West
# IMAGE_ID = "ami-04f2030e810b07ced"  # For East
# IMAGE_ID = "ami-0507d23ab4a37c611"  # For East (02-18-2019)
# IMAGE_ID = "ami-01b22f90823e1b2af"  # For East w/ Apex (02-21-2019)
# IMAGE_ID = "ami-0334863e514e3c3df"  # For ensemble GLUE datasets (03-04-2019)
# IMAGE_ID = "ami-0be2f9acd392d909a"  # For ensemble GLUE datasets (03-13-2019) with MNLI fixed.

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=[
        "list",
        "launch",
        "launch_and_run",
        "run",
        "shutdown",
        "shutdown_all",
        "run_and_shutdown",
    ],
)
parser.add_argument("--aws_access_key_id", required=True)
parser.add_argument("--aws_secret_access_key", required=True)
parser.add_argument("--region", default="us-east-1")
parser.add_argument("--n_machines", default=2, type=int)
parser.add_argument("--n_trials", default=None, type=int)
parser.add_argument("--keypath", required=True)
parser.add_argument("--outputpath", default="output")
parser.add_argument("--instance_type", default="t2.medium")
parser.add_argument("--only_print_commands", default=0)
parser.add_argument(
    "--run_name",
    default=datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S"),
)
parser.add_argument(
    "--configpath", required=True, type=str, help="path to config dicts"
)
parser.add_argument(
    "--commit_hash", required=True, type=str, help="git commit hash to run with"
)
parser.add_argument("--ami", required=True, type=str, help="ami id to run with")


def create_dummy_command_dict2():
    COMMAND_PREFIX = (
        "source activate pytorch_p36;"
        "python -m spacy download en;"  # TODO: add to env by default
        "rm -rf metal;"
        "git clone https://github.com/HazyResearch/metal.git;"
        "git checkout 868fb01a48e4031b8f1ac568e3bd0b413c904541;"
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

    # Helper for downloading directory
    def download_dir(sftp, remote_dir, local_dir):
        os.path.exists(local_dir) or os.makedirs(local_dir)
        dir_items = sftp.listdir_attr(remote_dir)
        for item in dir_items:
            # assuming the local system is Windows and the remote system is Linux
            # os.path.join won't help here, so construct remote_path manually
            remote_path = remote_dir + "/" + item.filename
            local_path = os.path.join(local_dir, item.filename)
            if S_ISDIR(item.st_mode):
                download_dir(sftp, remote_path, local_path)
            else:
                sftp.get(remote_path, local_path)

    key = paramiko.RSAKey.from_private_key_file(args.keypath)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    cmd = cmd_dict["cmd"]
    files_to_put, files_to_get, dirs_to_get = [], [], []
    if "files_to_put" in cmd_dict:
        files_to_put = cmd_dict["files_to_put"]
    if "files_to_get" in cmd_dict:
        files_to_get = cmd_dict["files_to_get"]
    if "dirs_to_get" in cmd_dict:
        dirs_to_get = cmd_dict["dirs_to_get"]

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
            print("Putting %s -> %s" % (localpath, remotepath))
            sftp.put(localpath, remotepath)

        # Execute a command(cmd) after connecting/ssh to an instance
        print("Machine: %s Running: %s" % (instance["public_ip_address"], cmd))
        stdin, stdout, stderr = client.exec_command(cmd)
        output[run_id] = stdout.read().decode("utf-8")

        # Save stdout and stderr to instance path
        with open(instance_output_path + "stderr", "w") as f:
            f.write(stderr.read().decode("utf-8"))
        with open(instance_output_path + "stdout", "w") as f:
            f.write(output[run_id])

        # Get files
        for (remotepath, localpath) in files_to_get:
            print(
                "Getting file %s -> %s" % (remotepath, instance_output_path + localpath)
            )
            sftp.get(remotepath, instance_output_path + localpath)

        # Get directories
        for (remotepath, localpath) in dirs_to_get:
            print(
                "Getting dir %s -> %s" % (remotepath, instance_output_path + localpath)
            )
            download_dir(sftp, remotepath, instance_output_path + localpath)

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


def get_user(instance):
    if instance.tags is not None:
        for tag in instance.tags:
            if "Key" in tag and "Value" in tag:
                if tag["Key"] == "user":
                    return tag["Value"]
    return "UNKNOWN"


def get_instances(args, filter_by_user=True):
    ec2_client, ec2_resource = create_ec2_client(args)
    instances = ec2_resource.instances.filter()
    if filter_by_user:
        instances = [
            x
            for x in instances
            if get_user(x) == os.environ["USER"] + "_" + args.run_name
        ]
    return instances


def get_all_instances(args, filter_by_user=True):
    ec2_client, ec2_resource = create_ec2_client(args)
    instances = ec2_resource.instances.filter()
    if filter_by_user:
        instances = [x for x in instances if os.environ["USER"] in get_user(x)]
    return instances


def describe_instances(args):

    instances = get_instances(args, filter_by_user=False)
    for instance in instances:
        print(
            "%s, state: %s, user: %s, type: %s\n\t"
            # "ssh -i %s ubuntu@%s"
            "eval `ssh-agent` && cat %s | ssh-add -k - && ssh -i %s ubuntu@%s"
            % (
                instance.id,
                instance.state["Name"],
                get_user(instance),
                instance.instance_type,
                args.keypath,
                args.keypath,
                instance.public_ip_address,
            )
        )


def launch(args):
    ec2_client, ec2_resource = create_ec2_client(args)
    instances = ec2_resource.create_instances(
        ImageId=args.ami,
        MinCount=args.n_machines,
        MaxCount=args.n_machines,
        KeyName=os.path.basename(args.keypath).split(".")[0],
        InstanceType=args.instance_type,
    )

    # Tag with user
    ec2_client.create_tags(
        Resources=[x.id for x in instances],
        Tags=[{"Key": "user", "Value": os.environ["USER"] + "_" + args.run_name}],
    )

    print("Waiting for instances: %s" % str([x.id for x in instances]))
    for instance in instances:
        instance.wait_until_running()
        instance.reload()

    # This is sometimes necessary to avoid ssh errors
    time.sleep(180)

    describe_instances(args)

    return instances


def shutdown(args):
    instances = get_instances(args)
    for instance in instances:
        instance.terminate()
    describe_instances(args)


def shutdown_all_by_user(args):
    instances = get_all_instances(args)
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
    print("Worker %s -> %s" % (wid, instance_dict["public_ip_address"]))

    # Run command on instance
    run_command(args, instance_dict, command_dict, return_output, run_id)


def initialize_process_ids(wid, process_id_mapping):
    current = str(multiprocessing.current_process())
    process_id_mapping[current] = wid
    print("Process %s => %s" % (current, str(wid)))


def run(args, launch_args, search_space, instances=None):

    # By default run command on all running machines
    if instances is None:
        instances = [x for x in get_instances(args) if x.state["Name"] == "running"]

    # Append the current date to the outputpath
    ts = time.time()
    timestamp_str = datetime.datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H_%M_%S")
    args.outputpath = args.outputpath + "/" + timestamp_str + "/"
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    manager = multiprocessing.Manager()

    # Generate commands
    return_output = manager.dict()
    process_id_mapping = manager.dict()
    command_dicts = grid_search_mmtl.generate_configs_and_commands(
        args, launch_args, search_space, args.n_trials
    )
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
    p.starmap(initialize_process_ids, initialization_args)

    # Main work
    if args.only_print_commands:
        for cmd_dict in command_dicts:
            print("-" * 50)
            for x in cmd_dict["files_to_put"]:
                print("cp %s %s" % x)
            print(cmd_dict["cmd"])
            print("-" * 50)
    else:
        p.starmap(worker_job, data)

    print("Results")
    print("-" * 100)
    for k, v in return_output.items():
        with open(args.outputpath + "/" + str(k) + ".out", "w") as f:
            f.write(str(v))


def launch_and_run(args, launch_args, search_space):
    instances = launch(args)
    run(args, launch_args, search_space, instances=instances)
    shutdown(args)


if __name__ == "__main__":

    args = parser.parse_args()
    # importing from config
    from importlib.machinery import SourceFileLoader

    config_dicts = SourceFileLoader("config_dicts", args.configpath).load_module()
    # config_dicts = importlib.import_module(args.configpath)
    search_space = config_dicts.search_space
    launch_args = config_dicts.launch_args
    if args.mode == "list":
        describe_instances(args)
    if args.mode == "launch":
        launch(args)
    if args.mode == "launch_and_run":
        launch_and_run(args, launch_args, search_space)
    if args.mode == "run":
        run(args, launch_args, search_space)
    if args.mode == "shutdown":
        shutdown(args)
    if args.mode == "run_and_shutdown":
        run(args, launch_args, search_space)
        shutdown(args)
    if args.mode == "shutdown_all":
        shutdown_all_by_user(args)
