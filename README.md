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

However, we also run this software setup outside of AWS; you can skip to
non-AWS setup for those instructions, which require a little more manual
effort.

### AWS Setup
**WARNING: Running Qanta scripts will create EC2 instances which you will be billed for**

Qanta scripts by default use [Spot Instances](https://aws.amazon.com/ec2/spot/) to get machines
at the lowest price at the expense that they may be terminated at any time if demand increases.
We find in practice that using the region `us-west-2` makes such terminations rare. Qanta primarily
uses `r3.8xlarge` machines which have 32 CPU cores, 244GB of RAM, and 640GB of SSD storage, but
other [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) are available.

#### Install and Configure Local Software

To execute the AWS scripts you will need to follow these steps:

1. [Install Packer Binaries](https://www.packer.io/downloads.html)
2. [Install Terraform 0.7.x](https://www.terraform.io/downloads.html)
3. Python 3.5+: If you don't have a preferred distribution,
[Anaconda Python](https://www.continuum.io/downloads) is a good choice
4. Install the AWS command line tools via `pip3 install awscli`. Run `pip3 install pyhcl`
5. Run `aws configure` to setup your AWS credentials, set default region to `us-west-2`
6. Create an [EC2 key pair](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
7. Set the environment variable `TF_VAR_key_pair` to the key pair name from the prior step
8. Set the environment variables `TF_VAR_access_key` and `TF_VAR_secret_key` to match your AWS
credentials.

#### What do the Packer/Terraform scripts install and configure?
This section is purely informative, you can skip to [Run AWS Scripts](#run-aws-scripts)

##### Installed Software
* Python 3.6
* Apache Spark 2.2.0
* Vowpal Wabbit 8.1.1
* Elastic Search 5.6.X (Not 6.X)
* CUDA and Nvidia drivers if using a GPU instance
* lz4
* All python packages in `packer/requirements.txt`

##### AWS Configuration
* Creates and configures an AWS virtual private cloud, internet gateway, route table, subnet on
us-west-1b, and security groups that optimize between security and convenience
* Security Groups: SSH access is enabled to the master, all other master node ports are closed to
the internet, all other instances can communicate with each other but are not reachable by the
internet.
* Spot instance requests for requested number of workers and a master node

#### Configuration
* **Warning**: AWS keys are printed during `terraform apply`
* Configure the 2 SSD drives attached to `r3.8xlarge` instances for use
* Clone the `Pinafore/qb` to `/ssd-c/qanta/qb` and set it as the quiz bowl root
* Download bootstrap AWS files to get the system running faster

#### Run AWS/Terraform/Packer Scripts

The AWS scripts are split between Packer and Terraform. Packer should be run from `packer/` and
Terraform from the root directory. Running Packer is optional because we publish public AMIs which Terraform uses by default.
If you are developing new pieces of qanta that require new software it might be helpful to build your own AMIs

1. (Optional) Packer: `packer build packer.json`
2. Terraform: `terraform apply` and note the `master_ip` output
3. SSH into the `master_ip` with `ssh -i mykey.pem ubuntu@ipaddr`

Additionally, the output from `terraform apply` is documented below and can be shown again with
`terraform show`

* `master_public_dns` and `master_public_ip`: Use for access from open web (eg ssh)
* `vpc_id`: Useful when adding custom security group

##### Terraform Environment Variables
Below is a list of variables that can change the behavior of Terraform. These can also be
passed into the CLI via `-var name=value` and dropping the `TF_VAR` portion.

* `TF_VAR_key_pair`: Which EC2 key pair to use
* `TF_VAR_access_key`: AWS access key
* `TF_VAR_secret_key`: AWS Secret key
* `TF_VAR_spot_price`: Max EC2 spot price
* `TF_VAR_master_instance_type`: Which EC2 instance type to use
* `TF_VAR_instance_count`: How many instances to start
* `TF_VAR_cluster_id`: On multi-user accounts allows separate users to run simultaneous machines
* `TF_VAR_qb_aws_s3_bucket`: Used to set `QB_AWS_S3_BUCKET` for checkpoint script
* `TF_VAR_qb_aws_s3_namespace`: Used to set `QB_AWS_S3_NAMESPACE` for checkpoint script

#### Shutting Down EC2 Instances

To teardown the cluster, you have two options.

1. `terraform destroy` will destroy all infrastructure created including the VPC/subnets/etc. If you
want to completely reset the AWS infrastructure this does the job
2. `terraform destroy -target=aws_spot_instance_request.master` will only destroy the EC2 instance.
This is the only part of the insfrastructure aside from S3 that AWS charges you for.


#### Accessing Resources on EC2

For security reasons, the AWS machines qanta creates are only accessible to the internet via SSH
to the master node. To gain access to the various web UIs (Spark, Luigi, Tensorboard) and other services
running on the cluster there are two options:

* Create an SSH tunnel to forward specific ports on the master to localhost
* In the EC2 Console create a security group which whitelists your IP address and add it to the
instance

##### SSH Tunnel

The following SSH command will forward all the important UIs running on the master node to
`localhost`:

`ssh -L 8080:localhost:8080 -L 4040:localhost:4040 -L 8082:localhost:8082 -L 6006:localhost:6006 ubuntu@instance-ip`

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
  LocalForward 6006 127.0.0.1:6006
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

### Non-AWS Setup

Since we do not primarily develop qanta outside of AWS and setups vary widely we don't maintain a
formal set of procedures to get qanta running not using AWS. Below are a listing of the important
scripts that Packer and Terraform run to install and configure a running qanta system.

#### Install Programs

1. Install Scala/Spark

http://www.scala-lang.org/download/
http://spark.apache.org/downloads.html

You will also need to setup spark to make sure it runs correctly.  A simple
way of making sure that QANTA can access Spark correctly is to run:

```
> python
Python 3.6.1 |Anaconda 4.4.0 (64-bit)| (default, May 11 2017, 13:09:58) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from qanta.spark import create_spark_context
>>> sc = create_spark_context()
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
17/07/25 10:04:01 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
17/07/25 10:04:01 WARN Utils: Your hostname, hongwu resolves to a loopback address: 127.0.0.2; using 192.168.2.2 instead (on interface eth0)
17/07/25 10:04:01 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
```

2. Install ElasticSearch 5.6

Install version 5.6.X, do not use 6.X. Also be sure that the directory `bin/` within the extracted files is in your
`$PATH` as it contains the necessary binary `elasticsearch`.

https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html

https://www.elastic.co/downloads/past-releases/elasticsearch-5-6-8

3. Install Python packages

```
pip install -r packer/requirements.txt
```

#### Non-AWS dependency download
If you are running on AWS, these files are already downloaded. Otherwise you will need to run either
`terraform/aws-downloads.sh` to get dependencies from Amazon S3 or run the bash commands below.

```bash
# Download nltk data
$ python3 setup.py download
```

### Qanta on Path

In addition to these steps you need to either run `python setup.py develop` or include the qanta directory in your
`PYTHONPATH` environment variable. We intend to fix path issues in the future by fixing absolute/relative paths.


## Configuration

QANTA configuration is done through a combination of environment variables and the `qanta-defaults.hcl` file. These are set
appropriately for AWS by Packer/Terraform, but are otherwise set to sensible defaults. QANTA will read a `qanta.hcl`
first if it exists, otherwise it will fall back to reading `qanta-defaults.hcl`. This is meant to allow for custom
configuration of `qanta.hcl` after copying it via `cp qanta-defaults.hcl qanta.hcl` without having a chance for configs
to accidentally become defaults unless that is on purpose.

Reference `conf/qb-env.sh.template` for a list of available configuration
variables  Copy this to something that you scource in your .bashrc.

## Run QANTA
### Pre-requisites

Complete either the non-AWS or AWS setup as above.

### Qanta Running Summary
Running the training batch pipeline is managed by
[Spotify Luigi](https://github.com/spotify/luigi). Luigi is a pure python make-like framework for
running data pipelines. The QANTA pipeline is specified in `qanta/pipeline.py`. Below are the
pre-requisites that need to be met before running the pipeline and how to run the pipeline itself.

### Running

These steps will guide you through starting Apache Spark, Luigi, and running the pipeline.
Where marked steps are marked"(Non-AWS)" indicates a step which is unnecessary to do if running
qanta from the AWS instance started by Terraform.

1. Start the Luigi daemon: `luigid --background` from `/ssd-c/qanta`
2. Start elastic search: `elasticsearch -d`

The above steps can also be done by using the `bin/init-nonaws.sh` script
which uses environment variables to move into the correct directory.

3. Run the full pipeline: `luigi --module qanta.pipeline All`
4. Observe pipeline progress at [http://hostname:8082](http://hostname:8082)

To rerun any part of the pipeline it is sufficient to delete the target file generated by the task
you wish to rerun.

## AWS S3 Checkpoint/Restore

To provide and easy way to version, checkpoint, and restore runs of qanta we provide a script to
manage that at `aws_checkpoint.py`. We assume that you set an environment variable
`QB_AWS_S3_BUCKET` to where you want to checkpoint to and restore from. We assume that we have full
access to all the contents of the bucket so we suggest creating a dedicated bucket.

## Problems you may encounter

> pg_config executable not found

Install postgres (required for python package `psycopg2`)

> pyspark uses the wrong version of python

Set PYSPARK_PYTHON to be python3

> ImportError: No module named 'pyspark'

export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH

> ValueError: unknown locale: UTF-8

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

> TypeError: namedtuple() missing 3 required keyword-only arguments: 'verbose', 'rename', and 'module'

Python 3.6 needs Spark 2.1.1

## Expo Instructions

The expo files can be generated from a completed qanta run by calling

```bash
luigi --module qanta.expo.pipeline --workers 2 AllExpo
```

If that has already been done you can restore the expo files from a backup instead of running the
pipeline

```bash
./checkpoint restore expo
```

Then to finally run the expo

```bash
python3 qanta/expo/buzzer.py --questions=output/expo/test.questions.csv --buzzes=output/expo/test.16.buzz --output=output/expo/competition.csv --finals=output/expo/test.16.final
```

## Utility Templates

Terraform works by reading all files ending in `.tf` within the directory that it is run. Unless the
filename ends with `_override` it will concatenate all these files together. In the case of
`_override` it will use the contents to override the current configuration. The combination of these
allows for keeping the root `aws.tf` clean while adding the possibility of customizing the build.

In the repository there are a number of `.tf.tftemplate` files. These are not read by terraform but
are intended to be copied to the same filename without the `.tftemplate` extension. The extension
merely serves to make it so that terraform by default does not read it, but to keep it in source
control (the files ending in `.tf` are in `.gitignore`). Below is a description of these

* `aws_gpu_override.tf.tftemplate`: This configures terraform to start a GPU instance instead of a
normal instance. This instance uses a different AMI that has GPU enabled Tensorflow/CUDA/etc.
* `aws_small_override.tf.tftemplate`: This configures terraform to use a smaller CPU instance than the
default r3.8xlarge
* `naqt_db.tf.tftemplate`: Configure qanta to use the private NAQT dataset
* `eip.tf.template`: Configure terraform to add a pre-made elastic IP to the instance

# Page Assignment and Data Ingestion

We use Wikipedia as our inventory of possible answers.  Because we
also use questions for training data, we need to map individual
questions to Wikipedia pages.  We have three systems for doing this
(the code that does the mapping lives in ingestion/create_db.py, which
produces a database of questions based on protobowl and NAQT input).

We provide a database with non-naqt questions from Protobowl, so if you're
content using that source of data, you can ignore this portion of the README.
However, this may be useful if you want to improve our question ingestion code
or to provide additional questions.

As per our agreement with NAQT, we cannot distribute the NAQT data,
but we include the ingestion code in the interest of scientific transparency.

## Unambiguous Page Assignments

These are the easiest pages to handle.  Given an answer string to a quiz bowl question, we directly map it to a Wikipedia page.

Unambiguous pages are unambiguous on the Wikipedia side.  There can be multiple answer lines associated with an answer:
 * adlai e stevenson ii    Adlai Stevenson II
 * adlai e stevenson jr    Adlai Stevenson II
 * adlai ewingstevensonii  Adlai Stevenson II
 * adlai stevenson ii      Adlai Stevenson II
 * buddha	Gautama Buddha
 * buddha or siddhartha gautama	Gautama Buddha
 * buddhism	Buddhism

However, some answers should not be in this list
 * byte	Byte
 * buffer	Buffer solution
 * britain	Battle of Britain

## Easy Ambiguous Page Assignments

Often, the same answer string can refer to multiple Wikipedia
entities.  If we can use words in the question to easily differentiate
them, then the page assignment can be done automatically.

For instance "Java" can refer to an island in Indonesia or a
programming language.
* java	Java	island
* java	Java (programming language)	language

Unlike above, where there were only two fields in our tab delimited
file, there are now three fields.  The first two fields are the same;
the last is a word that, if it appears in the question, says that the
question should be assigned to the page.


Sometimes there's a reasonable default answer.  For
example, most questions with the answer "Paris" will be about the city
in France.  However, there are also many questions about "Paris
(mythology)".  In this case, we create a rule
* paris	Paris (mythology)	aphrodite
* paris	Paris

If it finds a question with "Paris" as the answer line and the workd
"aphrodite" in the question, it will assign the question to "Paris
(mythology)".  Every other question, however, will be assigned to
"Paris" (the city).

We do not use ambiguous page assignments for closely related concepts
for example, "Orion (mythology)" and "Orion (constellation)" are so
tightly coupled that individual words cannot separate the concepts.
These cases have to be resolved individually for questions.

## Specific Question Assignments

If the above approaches cannot solve page assignments, then the last
resort is to explicitly assign questions to pages based on either
Protobowl or NAQT id.  These files have four fields but only use the
first three.

## Running the Assignment Process

> python3 ingestion/create_db.py

Needs Protobowl files at https://s3.amazonaws.com/protobowl/questions-05-05-2017.json.xz

## Last Mile Answer Matching

In addition to the ingestion script above, we have mapped additional answers with a new script. Eventually we intend
to merge this code into the main process/script, but for now it takes as input the database from the process above and
outputs a new database. This new database is in the repository, but we document how to generate it below.

```bash
python cli.py generate_additional_answer_mappings ~/Downloads/
python cli.py db_merge_answers data/internal/non_naqt.db /tmp/non_naqt.db ~/Downloads/answer_map.json /tmp/page_assignments.json
mv /tmp/non_naqt.db data/internal/non_naqt.db
```

# Wikipedia Dumps

As part of our ingestion pipeline we access raw wikipedia dumps. The current code is based on the english wikipedia
dumps created on 2017/04/01 available at https://dumps.wikimedia.org/enwiki/20170401/

Of these we use the following

* [Wikipedia page text](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-pages-articles-multistream.xml.bz2): This is used to get the text, title, and id of wikipedia pages
* [Wikipedia titles](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-all-titles.gz): This is used for more convenient access to wikipedia page titles
* [Wikipedia redirects](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-redirect.sql.gz): DB dump for wikipedia redirects, used for resolving different ways of referencing the same wikipedia entity
* [Wikipedia page to ids](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-page.sql.gz): Contains a mapping of wikipedia page and ids, necessary for making the redirect table useful

NOTE: If you are a Pinafore lab member with access to our S3 buckets on AWS this data is available at 

# Wikipedia Redirect Mapping Creation

All the wikipedia database dumps are provided in MySQL sql files. This guide has a good explanation of how to install MySQL which is necessary to use SQL dumps. For this task we will need these tables:

* Redirect table: https://www.mediawiki.org/wiki/Manual:Redirect_table
* Page table: https://www.mediawiki.org/wiki/Manual:Page_table
* The namespace page is also helpful: https://www.mediawiki.org/wiki/Manual:Namespace

To install, prepare MySQL, and read in the Wikipedia SQL dumps execute the following:

1. Install MySQL `sudo apt-get install mysql-server` and `sudo mysql_secure_installation`
2. Login with something like `mysql --user=root --password=something`
3. Create a database and use it with `create database wikipedia;` and `use wikipedia;`
4. `source enwiki-20170401-redirect.sql;` (in MySQL session)
5. `source enwiki-20170401-page.sql;` (in MySQL session)
6. This will take quite a long time, so wait it out...
7. Finally run the query to fetch the redirect mapping and write it to a CSV by executing `bin/redirect.sql` with `source bin/redirect.sql`. The file will be located in `/var/lib/mysql/redirect.csv` which requires `sudo` access to copy
8. The result of that query is CSV file containing a source page id, source page title, and target page title. This can be
interpretted as the source page redirecting to the target page. We filter namespace=0 to keep only redirects/pages that are main pages and trash things like list/category pages

These references may be useful and are the source for these instructions:

* https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-16-04
* https://dev.mysql.com/doc/refman/5.7/en/mysql-batch-commands.html
* http://stackoverflow.com/questions/356578/how-to-output-mysql-query-results-in-csv-format
