# __     __         _       _     _
# \ \   / /_ _ _ __(_) __ _| |__ | | ___  ___
#  \ \ / / _` | '__| |/ _` | '_ \| |/ _ \/ __|
#   \ V / (_| | |  | | (_| | |_) | |  __/\__ \
#    \_/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/

variable "key_pair" {}
variable "qanta_ami" {
  default = "ami-cf0445af"
}
variable "access_key" {}
variable "secret_key" {}
variable "spot_price" {
  default = "1.0"
}
variable "master_instance_type" {
  default = "r3.8xlarge"
}

variable "worker_instance_type" {
  default = "r3.8xlarge"
}

variable "num_workers" {
  default = 0
}

provider "aws" {
  region = "us-west-1"
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

# We use us-west-1b since r3.8xlarge are cheaper in this availability zone
resource "aws_subnet" "qanta_zone_1b" {
  vpc_id                  = "${aws_vpc.qanta.id}"
  cidr_block              = "10.0.2.0/24"
  map_public_ip_on_launch = true
  availability_zone = "us-west-1b"
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

  ingress {
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

# Create Spark workers
resource "aws_spot_instance_request" "workers" {
  ami           = "${var.qanta_ami}"
  instance_type = "${var.worker_instance_type}"
  count         = "${var.num_workers}"
  key_name = "${var.key_pair}"
  spot_price = "${var.spot_price}"

  vpc_security_group_ids = [
    "${aws_security_group.qanta_internal.id}",
    "${aws_security_group.qanta_ssh.id}"
  ]
  subnet_id = "${aws_subnet.qanta_zone_1b.id}"

  tags {
    SparkRole = "worker"
  }
  wait_for_fulfillment = true
}

# Create Spark master node
resource "aws_spot_instance_request" "master" {
  ami           = "${var.qanta_ami}"
  instance_type = "${var.master_instance_type}"
  key_name = "${var.key_pair}"
  spot_price = "${var.spot_price}"

  vpc_security_group_ids = [
    "${aws_security_group.qanta_internal.id}",
    "${aws_security_group.qanta_ssh.id}"
  ]
  subnet_id = "${aws_subnet.qanta_zone_1b.id}"

  tags {
    IsSparkMaster = "true"
    SparkRole = "worker"
  }

  ephemeral_block_device {
    device_name = "/dev/sdb"
    virtual_name = "ephemeral0"
  }

  ephemeral_block_device {
    device_name = "/dev/sdc"
    virtual_name = "ephemeral1"
  }
  wait_for_fulfillment = true

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

  # Configure AWS credentials
  provisioner "remote-exec" {
    inline = [
      "echo \"export AWS_ACCESS_KEY_ID=${var.access_key} >> /home/ubuntu/dependencies/spark-1.6.1-bin-hadoop2.6/conf/spark-env.sh\"",
      "echo \"export AWS_ACCESS_KEY_ID=${var.access_key} >> /home/ubuntu/.bashrc\"",
      "echo \"export AWS_SECRET_ACCESS=${var.secret_key} >> /home/ubuntu/dependencies/spark-1.6.1-bin-hadoop2.6/conf/spark-env.sh\"",
      "echo \"export AWS_SECRET_ACCESS=${var.secret_key} >> /home/ubuntu/.bashrc\"",
      "mkdir -p /home/ubuntu/.aws",
      "echo \"[default]\" >> /home/ubuntu/.aws/credentials",
      "echo \"aws_access_key_id = ${var.access_key}\" >> /home/ubuntu/.aws/credentials",
      "echo \"aws_secret_access_key = ${var.secret_key}\" >> /home/ubuntu/.aws/credentials",
    ]
  }

  provisioner "remote-exec" {
    inline = [
      "sudo mkdir /ssd-c/qanta",
      "sudo chown ubuntu /ssd-c/qanta",
      "git clone https://github.com/Pinafore/qb /ssd-c/qanta/qb"
    ]
  }

  provisioner "remote-exec" {
    script = "terraform/aws-downloads.sh"
  }
}

output "master_ip" {
  value = "${aws_spot_instance_request.master.public_ip}"
}

# ascii art from http://patorjk.com/software/taag/#p=display&f=Standard&t=EC2%20Instances
