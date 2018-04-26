This document contains our old instructions for running qanta on aws. There are no guarantee this works out of the
box. These instructions are here primarily for archival reasons and for reference.

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