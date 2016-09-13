# QANTA

NOTE: This project is actively maintained, but is going through changes rapidly since it is
research code. We do our best to make sure the code works after cloning and running installation
steps, but greatly appreciate any bug reports and encourage you to open a pull request to fix the
bug or add documentation. We will make a note here when we create a stable `2.0` tag.

## Setup
The primary way to run Qanta is using our [Packer](https://www.packer.io/) and
[Terraform](https://www.terraform.io) scripts to run it on
[Elastic Cloud Compute (EC2)](https://aws.amazon.com/ec2/) which is part of
[Amazon Web Services (AWS)](https://aws.amazon.com). The alternative is to inspect the bash scripts
associated with our Packer/Terraform scripts to infer the setup procedure.

Packer installs dependencies that don't need to know about runtime information (eg, it
installs `apt-get` software, download software distributions, etc). Terraform takes care of
creating AWS EC2 machines and provisioning them correctly (networking, secrets, dns, SSD drives,
etc).

### AWS Setup
**WARNING: Running Qanta scripts will create EC2 instances which you will be billed for**

Qanta scripts by default use [Spot Instances](https://aws.amazon.com/ec2/spot/) to get machines
at the lowest price at the expense that they may be terminated at any time if demand increases.
We find in practice that using the region `us-west-1` makes such terminations rare. Qanta primarily
uses `r3.8xlarge` machines which have 32 CPU cores, 244GB of RAM, and 640GB of SSD storage, but
other [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) are available.

#### Install and Configure Local Software

To execute the AWS scripts you will need to follow these steps (`brew` options are for Macs):

1. [Install Packer Binaries](https://www.packer.io/downloads.html) or run `brew install packer`
2. [Install Terraform 0.7.x](https://www.terraform.io/downloads.html) or run `brew install terraform`
3. Python 3.5+: If you don't have a preferred distribution,
[Anaconda Python](https://www.continuum.io/downloads) is a good choice
4. Install the AWS command line tools via `pip3 install awscli`
5. Run `aws configure` to setup your AWS credentials, set default region to `us-west-1`
6. Create an [EC2 key pair](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
7. Set the environment variable `TF_VAR_key_pair` to the key pair name from the prior step
8. Set the environment variables `TF_VAR_access_key` and `TF_VAR_secret_key` to match your AWS
credentials.
9. **WARNING**: These are copied by Terraform to the cluster so that the cluster has S3 access. See AWS the
configuration section for a summary of how the Terraform install scripts treat these keys.
10. Run `bin/generate-ssh-keys.sh n` where n equals the number of workers. You should start with zero
and scale up as necessary. This will generate SSH keys that are copied to the Spark cluster so that
nodes can communicate via SSH
11. [Install sshuttle](https://github.com/apenwarr/sshuttle/) which is used to create an SSH VPN

#### What do the Packer/Terraform scripts install and configure?
This section is purely informative, you can skip to [Run AWS Scripts](#run-aws-scripts)

##### Installed Software
* Python 3.5
* Apache Spark 1.6.1
* Vowpal Wabbit 8.1.1
* Docker 1.11.1
* Postgres
* KenLM
* All python packages in `packer/requirements.txt`

##### AWS Configuration
* Creates and configures an AWS virtual private cloud, internet gateway, route table, subnet on
us-west-1b, and security groups that optimize between security and convenience
* Security Groups: SSH access is enabled to the master, all other master node ports are closed to
the internet, all other instances can communicate with each other but are not reachable by the
internet.
* Spot instance requests for requested number of workers and a master node

#### Configuration
* SSH keys generated from `bin/generate-ssh-keys.sh` are copied to each instance. Each instance
receives its own ssh key and all other instances have SSH access to every other instance.
* AWS keys are copied to `/home/ubuntu/.bashrc`,
`/home/ubuntu/dependencies/spark-1.6.1-bin-hadoop2.6/conf/spark-env.sh`, and
`/home/ubuntu/.aws/credentials`.
* **Warning**: AWS keys are printed during `terraform apply`, we plan
on fixing this, but haven't yet.
* Configure the 2 SSD drives attached to `r3.8xlarge` instances for use
* Clone the `Pinafore/qb` to `/ssd-c/qanta/qb` and set it as the quiz bowl root
* Download bootstrap AWS files to get the system running faster
* TODO: based on variables download data from latest run to continue work on.

#### Run AWS/Terraform/Packer Scripts

The AWS scripts are split between Packer and Terraform. Packer should be run from `packer/` and
Terraform from the root directory. Running Packer is optional because we publish public AMIs which Terraform uses by default.
If you are developing new pieces of qanta that require new software it might be helpful to build your own AMIs

1. (Optional) Packer: `packer build packer.json`
2. Terraform: `terraform apply` and note the `master_ip` output
3. SSH into the `master_ip` with `ssh -i mykey.pem ubuntu@ipaddr`

The packer step is optional because we publish the two most recent Qanta AMIs on AWS which Terraform
uses automatically.

Additionally, the output from `terraform apply` is documented below and can be shown again with
`terraform show`

* `master_private_dns`: Address to access when using `sshuttle`
* `master_private_ip`: Internal AWS ip address
* `master_public_dns` and `master_public_ip`: Use for access from open web (eg ssh)
* `vpc_id`: Useful when adding custom security group

##### Terraform Environment Variables
Below is a list of variables that can change the behavior of Terraform. These can also be
passed into the CLI via `-var name=value` and dropping the `TF_VAR` portion.

* `TF_VAR_key_pair`: Which EC2 key pair to use
* `TF_VAR_access_key`: AWS access key
* `TF_VAR_secret_key`: AWS Secret key
* `TF_VAR_spot_price`: Max EC2 spot price
* `TF_VAR_master_instance_type`: Which EC2 instance type to use for master
* `TF_VAR_worker_instance_type`: Which EC2 instance type to use for workers (TODO: no-op ATM)
* `TF_VAR_num_workers`: How many workers to use (TODO: no-op ATM)
* `TF_VAR_cluster_id`: On multi-user accounts allows separate users to run simultaneous machines

#### Shutting Down EC2 Instances

To teardown the cluster, you have two options.

1. `terraform destroy` will destroy all infrastructure created including the VPC/subnets/etc. If you
want to completely reset the AWS infrastructure this does the job
2. `terraform destroy -target=aws_spot_instance_request.master` will only destroy the EC2 instance.
This is the only part of the insfrastructure aside from S3 that AWS charges you for.

### Non-AWS Setup
Since we do not primarily develop qanta outside of AWS and setups vary widely we don't maintain a
formal set of procedures to get qanta running not using AWS. Below are a listing of the important
scripts that Packer and Terraform run to install and configure a running qanta system.

1. `packer/packer.json`: Inventory of commands to setup pre-runtime image
2. `packer/setup.sh`: Install dependencies which don't require runtime information
3. `aws.tf`: Terraform configuration
4. `terraform/`: Bash scripts and configuration scripts

## Environment Variables
The majority of QANTA configuration is done through environment variables. These are set
appropriately for AWS by Packer/Terraform, but are otherwise set to sensible defaults.

Reference `conf/qb-env.sh.template` for a list of available configuration variables

## Run QANTA
### Pre-requisites

#### Accessing Resources on EC2

For security reasons, the AWS machines qanta creates are only accessible to the internet via SSH
to the master node. To gain access to the various web UIs (Spark, Luigi, Ganglia) and other services
running on the cluster there are three options:

* Create an SSH VPN with `sshuttle`. This forwards all traffic from your machine through the master
node, and gives access to AWS internal routing
* Create an SSH tunnel to forward specific ports on the master to localhost
* In the EC2 Console create a security group which whitelists your IP address and add it to the
instance

`sshuttle` gets you working off the ground the fastest but routes all your traffic to AWS. SSH
tunneling doesn't give access to everything. EC2 Security Groups is the overall most convenient
solution since it requires only adding your custom group to the instance after it starts

The reason for these security precautions is that allowing access to the Spark application master
or the spark master web UI would in principle expose a way for an attacker to gain access to your
AWS credentials.

##### SSH VPN

**Warning:** The following steps guide you through creating an SSH VPN to the master node. This
means **all** traffic will be redirected through the master node while the SSH VPN is running. AWS
charges for bandwidth which is outgoing from AWS so avoid downloading large files/videos to your
machine while the VPN is active (AWS does not charge for incoming bandwidth).

We recommend that before you work with Qanta that you run:

`sshuttle -N -H --dns -r ubuntu@public-ip 0/0`

##### SSH Tunnel

The following SSH command will forward all the important UIs running on the master node to
`localhost`:

`ssh -L 8080:localhost:8080 -L 4040:localhost:4040 -L 8082:localhost:8082 ubuntu@instance-ip`

This can be made easier by adding an entry like below in `~/.ssh/config`. Note that the example
domain `example.com` is mapped to the master ip address outputed by terraform. This can be
accomplished by modifying `/etc/hosts` or creating a new DNS entry for the domain.

```
Host qanta
  HostName example.com
  StrictHostKeyChecking no
  UserKnownHostsFile=/dev/null
  User ubuntu
  LocalForward 8082 127.0.0.1:8082
  LocalForward 8080 127.0.0.1:8080
```

Now you can simply do `ssh qanta` and navigating to `localhost:8082` will access the EC2 instance.

##### Custom Security Group
1. Go to [console.aws.amazon.com](console.aws.amazon.com)
2. Under "Network & Security" click "Security Groups"
3. Click "Create Security Group"
4. Configure with a name, any relevant inbound rules (eg from a whitelist IP), and be sure to choose
the VPC created by Terraform. This can be retrieved by using `terraform show` and using the variable
output from `vpc_id`.
5. Under "Instance" click "Instances"
6. Select your instance, click the "Actions" drop down, click "Networking" then
"Change Security Groups", and finally add your security group


#### Non-AWS dependency download
If you are running on AWS, these files are already downloaded. Otherwise you will need to run either
`terraform/aws-downloads.sh` to get dependencies from Amazon S3 or run the bash commands below.

```bash
# Download Wikifier (S3 Download in script mentioned above is much faster, this is 8GB file compressed)
wget -O /tmp/Wikifier2013.zip http://cogcomp.cs.illinois.edu/software/Wikifier2013.zip
unzip /tmp/Wikifier2013.zip -d data/external
rm /tmp/Wikifier2013.zip

# Download nltk data
$ python3 setup.py download
```

#### Standard Pre-requisites
Before running qanta software you will need to compile the C language model and download the nltk
datasets used by running:

```bash
# Run pre-requisites
$ python3 setup.py download
$ make clm
```

### Qanta Running Summary
QANTA can be run in two modes: batch or streaming. Batch mode is used for training and evaluating
large batches of questions at a time. Running the batch pipeline is managed by
[Spotify Luigi](https://github.com/spotify/luigi). Luigi is a pure python make-like framework for
running data pipelines. The QANTA pipeline is specified in `qanta/pipeline.py`. Below are the
pre-requisites that need to be met before running the pipeline and how to run the pipeline itself.

### Running Batch Mode

These steps will guide you through starting Apache Spark, Luigi, and running the pipeline.
Where marked steps are marked"(Non-AWS)" indicates a step which is unnecessary to do if running
qanta from the AWS instance started by Terraform.

1. Start the spark cluster by navigating into `$SPARK_HOME` and running `sbin/start-all.sh`
2. Start the Luigi daemon: `luigid --background` from `/ssd-c/qanta`
3. Before you can run any of the features, you need to build the guess database:
`luigi --module qanta.pipeline CreateGuesses --workers 1`.  You can skip ahead to next step if you
want (it will also create the guesses, but seeing the guesses will ensure that the deep guesser
worked as expected).
4. Run the full pipeline: `luigi --module qanta.pipeline AllSummaries --workers 30`
5. Observe pipeline progress at [http://hostname:8082](http://hostname:8082)

To rerun any part of the pipeline it is sufficient to delete the target file generated by the task
you wish to rerun.

### Running Streaming Mode

**Warning: This mode is highly experimental**

Again, Apache Spark needs to be running at the url specified in the environment variable
`QB_SPARK_MASTER`.

Streaming mode works by coordinating several processes to predict whether or not to buzz given line
of text (a sentence, partial sentence, paragraph, or anything not containing a new line). The Qanta
server is responsible for:
* Creating a socket then waiting until a connection is established
* The connection is established by starting an Apache Spark streaming job that binds its input to
that socket
* Once the connection is established the Qanta server will start streaming questions to Spark until
its queue is empty.
* Each question is also stored in a PostgreSQL database with a column reserved for Spark's response
* The Qanta server will start to poll the database every 100ms to see if Spark completed all the
questions that were queued.

Spark Streaming will then do the following per input line:
* Read the input from the socket
* Extract all features
* Collect the features, form a Vowpal Wabbit input line, and have VW create predictions. This
requires that VW is running in daemon mode
* Save the output to a PostgreSQL database

Once the outputs are saved in the PostgreSQL database
* The Qanta server reads the results and outputs the desired quantities

With that high-level overview in place, here is how you start the whole system.

1. Start the PostgreSQL database (Docker must be running first): `bin/start-postgres.sh`
2. Start the Vowpal Wabbit daemon: `bin/start-vw-daemon.sh model-file.vw`
(eg `data/models/sentence.16.vw`)
3. Start Qanta server: `python3 cli.py qanta_stream`
4. Start Spark Streaming job: `python3 cli.py spark_stream`

### AWS S3 Checkpoint/Restore

To provide and easy way to version, checkpoint, and restore runs of qanta we provide a script to
manage that at `aws_checkpoint.py`. We assume that you set an environment variable
`QB_AWS_S3_BUCKET` to where you want to checkpoint to and restore from. We assume that we have full
access to all the contents of the bucket so we suggest creating a dedicated bucket.

### Problems you may encounter

> pg_config executable not found

Install postgres (required for python package psycopg2, used by streaming)

> pyspark uses the wrong version of python

Set PYSPARK_PYTHON to be python3

> ImportError: No module named 'pyspark'

export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH

> ValueError: unknown locale: UTF-8

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

> No module named 'pyspark'
