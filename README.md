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
creating AWS EC2 machines and provisioning them correctly (networking, secrets, dns, SSD drivers,
etc).

### AWS Setup
**WARNING: Running Qanta scripts will create EC2 instances which you will be billed for**

Qanta scripts by default use [Spot Instances](https://aws.amazon.com/ec2/spot/) to get machines
at the lowest price at the expense that they may be terminated at any time if demand increases.
We find in practice that when using the region `us-west-1` makes terminations rare. Qanta primarily
uses `r3.8xlarge` machines which have 32 CPU cores, 244GB of RAM, and 640GB of SSD storage, but
other [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) are available.

#### Install and Configure Local Software

To execute the AWS scripts you will need to follow these steps:

1. [Packer Binaries](https://www.packer.io/downloads.html) or via `brew install packer`
2. [Terraform Binaries](https://www.terraform.io/downloads.html) or via `brew install terraform`
3. Python 3.5+: If you don't have a preferred distribution,
[Anaconda Python](https://www.continuum.io/downloads) is a good choice
4. Install the AWS command line tools via `pip3 install awscli`
5. Run `aws configure` to setup your AWS credentials, set default region to `us-west-1`
6. Create an [EC2 key pair](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
7. Set the environment variable `TF_VAR_key_pair` to the key pair name from the prior step
8. Set the environment variables `TF_VAR_access_key` and `TF_VAR_secret_key` to match your AWS
credentials. **WARNING**: These are used to running Terraform and are subsequently copied to the
instance to provide access to AWS resources such as S3
9. Run `bin/generate-ssh-keys.sh n` where n equals the number of workers. You should start with zero
and scale up as necessary. This will generate SSH keys that are copied to the Spark cluster so that
nodes can communicate via SSH

#### Execute AWS Scripts

The AWS scripts are split between Packer and Terraform. All these commands are run from the root
directory.

1. (Optional) Packer (from `packer/` directory): `packer build packer.json`
2. Terraform: `terraform apply`

The packer step is optional because we publish the most two recent Qanta AMIs on AWS and keep the
default Terraform base AMI in sync with this. If you update from our repository frequently this
should not cause issues. Otherwise, we suggest you run the packer script and set the environment
variable `TF_VAR_qanta_ami` to the AMI id.

### Non-AWS Setup
Since we do not primarily develop qanta outside of AWS and setups vary widely we don't maintain a
formal set of procedures to get qanta running not using AWS. Below are a listing of the important
scripts that Packer and Terraform run to install and configure a running qanta system.

1. `packer/packer.json`: Inventory of commands to setup pre-runtime image
2. `packer/setup.sh`: Install dependencies which don't require runtime information
3. `aws.tf`: Terraform configuration
4. `terraform/`: Bash scripts and configuration scripts

#### Dependencies

* Python 3.5
* Apache Spark 1.6.1
* Vowpal Wabbit 8.1.1
* Docker 1.11.1
* Postgres 
* kenlm 
* All python packages in `packer/requirements.txt`

#### Installation
1. Download the Illinois Wikifier code (VERSION 2).  Place the data directory in data/wikifier/data and put the wikifier-3.0-jar-with-dependencies.jar in the lib directory http://cogcomp.cs.illinois.edu/page/software_view/Wikifier and put the config directory in data/wikifier/config

## Environment Variables
The majority of QANTA configuration is done through environment variables. Where possible, these
have been set to sensible defaults.

The simplest way to set this up is to copy the contents of `conf/qb-env.sh.template` and make sure
that the script is executed. For example, in your `~/.bashrc` inserting a line `source qb-env.sh`.

Documentation for what each of these does is in the configuration template

## Running QANTA
QANTA can be run in two modes: batch or streaming. Batch mode is used for training and evaluating
large batches of questions at a time. Running the batch pipeline is managed by
[Spotify Luigi](https://github.com/spotify/luigi). Luigi is a pure python make-like framework for
running data pipelines. The QANTA pipeline is specified in `qanta/pipeline.py`. Below are the
pre-requisites that need to be met before running the pipeline and how to run the pipeline itself.
Eventually any data related targets will be moved to Luigi and leave only compile-like targets in
the makefile

### Prerequisites
Before running the system, there are a number of compile-like dependencies and data dependencies to
download. If you don't mind waiting a while, executing the commands below will get everything you
need.

However, some of the data dependencies take a while to download. To speed things along, we also
provide a script to download the files from our Amazon S3 bucket and place them into the correct
location. The script is in `bin/bootstrap.sh`, needs to be executed from the root of the QB
repository, and requires you to have already run `aws configure` to setup your AWS credentials.
You also may need to run `pip install awscli`.

```bash
# Download Wikifier (S3 Download in script mentioned above is much faster, this is 8GB file compressed)
wget -O /tmp/Wikifier2013.zip http://cogcomp.cs.illinois.edu/software/Wikifier2013.zip
unzip /tmp/Wikifier2013.zip -d data/external
rm /tmp/Wikifier2013.zip

# Run pre-requisites
$ make prereqs

# Download nltk data
$ python3 setup.py download
```

Additionally, you must have Apache Spark running at the url specified in the environment variable
`QB_SPARK_MASTER`

### Running Batch Mode

These instructions are a work in progress. Generally speaking, you need to start the spark cluster,
start luigi, then submit the `AllSummaries` task. If it is your first time running it, we suggest
you use more intermediate targets from `qanta/pipline.py`

0. Start the spark cluster by navigating into `$SPARK_HOME` and running `sbin/start-all.sh`
1. Start the Luigi daemon: `luigid --background --address 0.0.0.0`
2. Before you can run any of the features, you need to build the guess database: `luigi --module qanta.pipeline CreateGuesses --workers 30` (change 30 to number of concurrent tasks to run at a time).  You can skip ahead to next step if you want (it will also create the guesses, but seeing the guesses will ensure that the deep guesser worked as expected).
3. Run the pipeline: `luigi --module qanta.pipeline AllSummaries --workers 30`
4. Observe pipeline progress at [http://hostname:8082](http://hostname:8082)

To rerun any part of the pipeline it is sufficient to delete the target file generated by the task
you wish to rerun.

### Running Streaming Mode
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
