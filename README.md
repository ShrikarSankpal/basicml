# basicml# basic2
To see what flavour and version of linux I am running.
cat /etc/os-release
I am using Centos7


Installing Docker:


Checking Docker Version:
docker --version
I am using Docker version 26.1.1, build 4cf5afa

Starting docker:
systemctl start docker

Steps for Docker:
Be careful while writing your working dir in Dockerfile.

building the image
docker build -t my_app .

Just running the image, interactively, provided you have CMD ["/bin/bash"] in your Dockerfile at the end.
docker run -it my_app

Another method to run cmd in container:
docker exec -it container_name /bin/bash

Mounting a local file inside a container:
docker run -v path/in/host/filename.txt:path/in/container/filename.txt my_app
