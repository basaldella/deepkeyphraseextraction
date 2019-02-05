#!/bin/bash

source variables.sh

docker image build -t $IMAGE_NAME .
