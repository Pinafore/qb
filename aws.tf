# __     __         _       _     _
# \ \   / /_ _ _ __(_) __ _| |__ | | ___  ___
#  \ \ / / _` | '__| |/ _` | '_ \| |/ _ \/ __|
#   \ V / (_| | |  | | (_| | |_) | |  __/\__ \
#    \_/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/

variable "key_pair" {}
variable "access_key" {}
variable "secret_key" {}

variable "spot_price" {
  default = "1.0"
}

variable "master_instance_type" {
  default = "p2.xlarge"
  description = "EC2 Instance type to use for the master node"
}

variable "cluster_id" {
  default = "default"
  description = "Cluster identifier to prevent collissions for users on the same AWS account"
}

variable "qb_aws_s3_bucket" {
  default = ""
  description = "Stores variable for QB_AWS_S3_BUCKET used in checkpoint script"
}

variable "qb_aws_s3_namespace" {
  default = ""
  description = "Stores variable for QB_AWS_S3_NAMESPACE used in checkpoint script"
}

variable "qb_branch" {
  default = "master"
  description = "Which git branch to checkout when cloning Pinafore/qb"
}

variable "num_experiments" {
  default = "1"
  description = "Number of masters to bring up"
}

provider "aws" {
  region = "us-west-2"
}

data "aws_ami" "qanta_ami" {
  most_recent = true
  filter {
    name = "tag-key"
    values = ["Image"]
  }
  filter {
    name = "tag-value"
<<<<<<< HEAD
    values = ["davis-qanta"]
=======
    values = ["qanta-cpu"]
>>>>>>> master
  }
}

#  _   _      _                      _    _
# | \ | | ___| |___      _____  _ __| | _(_)_ __   __ _
# |  \| |/ _ \ __\ \ /\ / / _ \| '__| |/ / | '_ \ / _` |
# | |\  |  __/ |_ \ V  V / (_) | |  |   <| | | | | (_| |
# |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\_|_| |_|\__, |
#                                                 |___/

# Create a VPC to launch our instances into
resource "aws_vpc" "qanta" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support = true
  enable_dns_hostnames = true
}

# Create an internet gateway to give our subnet access to the outside world
resource "aws_internet_gateway" "qanta" {
  vpc_id = "${aws_vpc.qanta.id}"
}

# Grant the VPC internet access on its main route table
resource "aws_route" "internet_access" {
  route_table_id         = "${aws_vpc.qanta.main_route_table_id}"
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = "${aws_internet_gateway.qanta.id}"
}

# We use us-west-2a since r3.8xlarge are cheaper in this availability zone
# p2 Instances are cheaper on us-west-2c however
resource "aws_subnet" "qanta_zone_2a" {
  vpc_id                  = "${aws_vpc.qanta.id}"
  cidr_block              = "10.0.2.0/24"
  map_public_ip_on_launch = true
<<<<<<< HEAD
  availability_zone = "us-west-2b"
=======
  availability_zone = "us-west-2a"
>>>>>>> master
}

# A security group for SSH access from anywhere
resource "aws_security_group" "qanta_ssh" {
  name        = "qanta_ssh"
  description = "Enable SSH access from anywhere"
  vpc_id      = "${aws_vpc.qanta.id}"

  # SSH access from anywhere
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # MOSH access from anywhere
  ingress {
    from_port = 60000
    to_port = 61000
    protocol = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # outbound internet access
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "qanta_internal" {
  name = "qanta_internal"
  description = "Full access to machines in group"
  vpc_id = "${aws_vpc.qanta.id}"

  ingress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    self = true
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    self = true
  }
}

#  _____ ____ ____    ___           _
# | ____/ ___|___ \  |_ _|_ __  ___| |_ __ _ _ __   ___ ___  ___
# |  _|| |     __) |  | || '_ \/ __| __/ _` | '_ \ / __/ _ \/ __|
# | |__| |___ / __/   | || | | \__ \ || (_| | | | | (_|  __/\__ \
# |_____\____|_____| |___|_| |_|___/\__\__,_|_| |_|\___\___||___/

resource "aws_spot_instance_request" "master" {
  count = "${var.num_experiments}"
  # ami           = "${data.aws_ami.qanta_ami.id}"
  ami           = "ami-2b7ea04b"
  instance_type = "${var.master_instance_type}"
  key_name = "${var.key_pair}"
  spot_price = "${var.spot_price}"
  spot_type = "one-time"
  wait_for_fulfillment = true

  vpc_security_group_ids = [
    "${aws_security_group.qanta_internal.id}",
    "${aws_security_group.qanta_ssh.id}"
  ]
  subnet_id = "${aws_subnet.qanta_zone_2a.id}"

  root_block_device {
    volume_type = "gp2"
    volume_size = "80"
    delete_on_termination = true
  }

  # ephemeral_block_device {
  #   device_name = "/dev/sdb"
  #   virtual_name = "ephemeral0"
  # }

  # ephemeral_block_device {
  #   device_name = "/dev/sdc"
  #   virtual_name = "ephemeral1"
  # }

  ebs_block_device {
    device_name = "/dev/sdb"
    volume_type = "gp2"
    volume_size = 80
  }

  ebs_block_device {
    device_name = "/dev/sdc"
    volume_type = "gp2"
    volume_size = 80
  }
  connection {
    user = "ubuntu"
  }

  # Copy SSH keys for Spark to use
  provisioner "file" {
    source = "terraform/ssh-keys"
    destination = "/home/ubuntu"
  }

  provisioner "remote-exec" {
    inline = [
      "mkdir -p /home/ubuntu/.ssh",
      "cat /home/ubuntu/ssh-keys/*.pub >> /home/ubuntu/.ssh/authorized_keys",
      "mv /home/ubuntu/ssh-keys/spark.master /home/ubuntu/.ssh/id_rsa",
      "mv /home/ubuntu/ssh-keys/spark.master.pub /home/ubuntu/.ssh/id_rsa.pub",
      "chmod 600 /home/ubuntu/.ssh/id_rsa"
    ]
  }

  provisioner "remote-exec" {
    script = "terraform/configure-drives.sh"
  }

  provisioner "remote-exec" {
    script = "terraform/configure-swap.sh"
  }

  # Add a line to bashrc to source all our environment
  provisioner "remote-exec" {
    inline = [
        "echo source /home/ubuntu/qbenv >> /home/ubuntu/.bashrc"
    ]
  }

  # Configure AWS credentials
  provisioner "remote-exec" {
    inline = [
      "echo \"export AWS_ACCESS_KEY_ID=${var.access_key}\" >> /home/ubuntu/dependencies/spark-2.0.0-bin-hadoop2.7/conf/spark-env.sh",
      "echo \"export AWS_ACCESS_KEY_ID=${var.access_key}\" >> /home/ubuntu/qbenv",
      "echo \"export AWS_SECRET_ACCESS_KEY=${var.secret_key}\" >> /home/ubuntu/dependencies/spark-2.0.0-bin-hadoop2.7/conf/spark-env.sh",
      "echo \"export AWS_SECRET_ACCESS_KEY=${var.secret_key}\" >> /home/ubuntu/qbenv",
      "mkdir -p /home/ubuntu/.aws",
      "echo \"[default]\" >> /home/ubuntu/.aws/credentials",
      "echo \"aws_access_key_id = ${var.access_key}\" >> /home/ubuntu/.aws/credentials",
      "echo \"aws_secret_access_key = ${var.secret_key}\" >> /home/ubuntu/.aws/credentials",
    ]
  }

  # Configure qanta environment variables
  provisioner "remote-exec" {
    inline = [
      "echo 'export PATH=/home/ubuntu/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games' >> /home/ubuntu/qbenv",
      "echo 'export QB_ROOT=/ssd-c/qanta/qb' >> /home/ubuntu/qbenv",
      "echo 'export QB_QUESTION_DB=$${QB_ROOT}/data/internal/non_naqt.db' >> /home/ubuntu/qbenv",
      "echo 'export QB_GUESS_DB=$${QB_ROOT}/output/guesses.db' >> /home/ubuntu/qbenv",
      "echo 'export PYTHONPATH=:/home/ubuntu/dependencies/spark-2.0.0-bin-hadoop2.7/python' >> /home/ubuntu/qbenv",
      "echo 'export QB_SPARK_MASTER=spark://${self.private_dns}:7077' >> /home/ubuntu/qbenv",
      "echo 'export PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python' >> /home/ubuntu/qbenv",
      "echo 'export PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python' >> /home/ubuntu/qbenv",
      "echo 'export QB_AWS_S3_BUCKET=${var.qb_aws_s3_bucket}' >> /home/ubuntu/qbenv",
      "echo 'export QB_AWS_S3_NAMESPACE=${var.qb_aws_s3_namespace}' >> /home/ubuntu/qbenv",
      "echo 'export QB_TF_EXPERIMENT_ID=${count.index}' >> /home/ubuntu/qbenv",
      "echo 'export CUDA_HOME=/usr/local/cuda' >> /home/ubuntu/qbenv",
      "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> /home/ubuntu/qbenv",
      "echo 'export PATH=/home/ubuntu/anaconda3/bin:$PATH' >> /home/ubuntu/qbenv",
      "echo 'export SPARK_HOME=/home/ubuntu/dependencies/spark-2.0.0-bin-hadoop2.7' >> /home/ubuntu/qbenv",
      "echo 'export PYTHONPATH=$PYTHONPATH:/ssd-c/qanta/qb' >> /home/ubuntu/qbenv"
    ]
  }

  provisioner "remote-exec" {
    inline = [
      "sudo mkdir /ssd-c/qanta",
      "sudo chown ubuntu /ssd-c/qanta",
      "git clone https://github.com/Pinafore/qb /ssd-c/qanta/qb",
      "(cd /ssd-c/qanta/qb && git checkout ${var.qb_branch} && /home/ubuntu/anaconda3/bin/python setup.py develop)"
    ]
  }

  provisioner "file" {
    source = "terraform/luigi.cfg"
    destination = "/ssd-c/qanta/luigi.cfg"
  }

  provisioner "remote-exec" {
    script = "terraform/aws-downloads.sh"
  }

  provisioner "file" {
    source = "terraform/setup-experiment.sh"
    destination = "/tmp/setup-experiment.sh"
  }

  provisioner "remote-exec" {
    inline = [
        "bash /tmp/setup-experiment.sh"
    ]
  }

  provisioner "file" {
      source = "qanta/guesser/tf/experiments.py"
      destination = "/ssd-c/qanta/qb/qanta/guesser/tf/experiments.py"
  }

  provisioner "file" {
      source = "terraform/launch-experiment.sh"
      destination = "/tmp/launch-experiment.sh"
  }
  provisioner "remote-exec" {
        inline = [
          "bash /tmp/launch-experiment.sh"
        ]
  }
}

output "master_public_ip" {
  value = "${join(",",aws_spot_instance_request.master.*.public_ip)}"
}

output "master_public_dns" {
  value = "${join(",",aws_spot_instance_request.master.*.public_dns)}"
}

output "master_private_ip" {
  value = "${join(",",aws_spot_instance_request.master.*.private_ip)}"
}

output "master_private_dns" {
  value = "${join(",",aws_spot_instance_request.master.*.private_dns)}"
}

output "master_instance_id" {
  value = "${join(",",aws_spot_instance_request.master.*.id)}"
}

output "vpc_id" {
  value = "${aws_vpc.qanta.id}"
}

# ascii art from http://patorjk.com/software/taag/#p=display&f=Standard&t=EC2%20Instances
