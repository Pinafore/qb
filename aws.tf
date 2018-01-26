# __     __         _       _     _
# \ \   / /_ _ _ __(_) __ _| |__ | | ___  ___
#  \ \ / / _` | '__| |/ _` | '_ \| |/ _ \/ __|
#   \ V / (_| | |  | | (_| | |_) | |  __/\__ \
#    \_/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/

variable "key_pair" {}
variable "access_key" {}
variable "secret_key" {}

variable "kuro_server" {
  default = ""
}

variable "spot_price" {
  default = "2.70"
}

variable "master_instance_type" {
  default = "r3.8xlarge"
  description = "EC2 Instance type to use for the master node"
}

variable "instance_count" {
  default = 1
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
    values = ["qanta-cpu"]
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
resource "aws_subnet" "qanta_zone_2c" {
  vpc_id                  = "${aws_vpc.qanta.id}"
  cidr_block              = "10.0.2.0/24"
  map_public_ip_on_launch = true
  availability_zone = "us-west-2c"
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

data "aws_iam_instance_profile" "s3_instance_profile" {
  name = "s3-full-access"
}


#  _____ ____ ____    ___           _
# | ____/ ___|___ \  |_ _|_ __  ___| |_ __ _ _ __   ___ ___  ___
# |  _|| |     __) |  | || '_ \/ __| __/ _` | '_ \ / __/ _ \/ __|
# | |__| |___ / __/   | || | | \__ \ || (_| | | | | (_|  __/\__ \
# |_____\____|_____| |___|_| |_|___/\__\__,_|_| |_|\___\___||___/

resource "aws_spot_instance_request" "qanta" {
  ami           = "${data.aws_ami.qanta_ami.id}"
  instance_type = "${var.master_instance_type}"
  key_name = "${var.key_pair}"
  spot_price = "${var.spot_price}"
  spot_type = "one-time"
  wait_for_fulfillment = true
  count = "${var.instance_count}"
  iam_instance_profile = "${data.aws_iam_instance_profile.s3_instance_profile.name}"

  vpc_security_group_ids = [
    "${aws_security_group.qanta_internal.id}",
    "${aws_security_group.qanta_ssh.id}"
  ]
  subnet_id = "${aws_subnet.qanta_zone_2c.id}"

  root_block_device {
    volume_type = "gp2"
    volume_size = "80"
    delete_on_termination = true
  }

  ephemeral_block_device {
    device_name = "/dev/sdb"
    virtual_name = "ephemeral0"
  }

  ephemeral_block_device {
    device_name = "/dev/sdc"
    virtual_name = "ephemeral1"
  }

  connection {
    user = "ubuntu"
  }

  provisioner "remote-exec" {
    script = "terraform/configure-drives.sh"
  }

  provisioner "remote-exec" {
    script = "terraform/configure-swap.sh"
  }

  # Configure qanta environment variables
  provisioner "remote-exec" {
    inline = [
      "echo \"export PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python\" >> /home/ubuntu/.bashrc",
      "echo \"export QB_AWS_S3_BUCKET=${var.qb_aws_s3_bucket}\" >> /home/ubuntu/.bashrc",
      "echo \"export QB_AWS_S3_NAMESPACE=${var.qb_aws_s3_namespace}\" >> /home/ubuntu/.bashrc",
      "echo \"export KURO_SERVER=${var.kuro_server}\" >> /home/ubuntu/.bashrc",
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

  provisioner "remote-exec" {
    script = "terraform/terraform-aws-downloads.sh"
  }

  provisioner "remote-exec" {
    inline = [
      "export PATH=$PATH:/home/ubuntu/anaconda3/bin",
      "cd /ssd-c/qanta/qb && bash bin/init.sh"
    ]
  }
}

output "qanta_public_ip" {
  value = "${join(",", aws_spot_instance_request.qanta.*.public_ip)}"
}

output "qanta_public_dns" {
  value = "${join(",", aws_spot_instance_request.qanta.*.public_dns)}"
}

output "qanta_private_ip" {
  value = "${join(",", aws_spot_instance_request.qanta.*.private_ip)}"
}

output "qanta_private_dns" {
  value = "${join(",", aws_spot_instance_request.qanta.*.private_dns)}"
}

output "qanta_instance_id" {
  value = "${join(",", aws_spot_instance_request.qanta.*.id)}"
}

output "vpc_id" {
  value = "${aws_vpc.qanta.id}"
}
