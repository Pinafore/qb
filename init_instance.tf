resource "null_resource" "init_instance" {
  # Changes to any instance of the cluster requires re-provisioning
  triggers {
    cluster_instance_ids = "${aws_spot_instance_request.master.spot_instance_id}"
  }

  # Bootstrap script can run on any instance of the cluster
  # So we just choose the first in this case
  connection {
    host = "${aws_spot_instance_request.master.public_ip}"
  }

  provisioner "remote-exec" {
    script = "bin/init.sh"
  }
}
