#!/bin/bash

source variables.sh

docker image build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t $IMAGE_NAME .
