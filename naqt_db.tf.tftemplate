resource "null_resource" "naqt_db" {
  # Changes to any instance of the cluster requires re-provisioning
  triggers {
    cluster_instance_ids = "${aws_spot_instance_request.master.spot_instance_id}"
  }

  # Bootstrap script can run on any instance of the cluster
  # So we just choose the first in this case
  connection {
    host = "${aws_spot_instance_request.master.public_ip}"
    user = "ubuntu"
  }

  provisioner "remote-exec" {
    inline = [
      "/home/ubuntu/anaconda3/bin/aws s3 cp s3://entilzha-us-west-2/questions/naqt.db /ssd-c/qanta/qb/data/internal/naqt.db",
      "echo \"export QB_QUESTION_DB=\\$${QB_ROOT}/data/internal/naqt.db\" >> /home/ubuntu/.bashrc"
    ]
  }
}
