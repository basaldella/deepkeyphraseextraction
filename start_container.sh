#!/bin/bash

source variables.sh

if [ $# -eq 0 ]; then
    $DOCKER_BIN run -d -v "$(pwd)/src":/opt/proj/src -v "$(pwd)/data":/opt/proj/data --user $(id -u):$(id -g) --name $CONTAINER_NAME $IMAGE_NAME
    exit 0
fi

if [ $# -eq 1 ]; then
    if [ "$1" == "-it" ]; then
        $DOCKER_BIN run -it -v "$(pwd)/src":/opt/proj/src -v "$(pwd)/data":/opt/proj/data --user $(id -u):$(id -g) --name $CONTAINER_NAME $IMAGE_NAME bash
	read -p "Do you want to remove the container? [y/n] " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
	    docker container rm $CONTAINER_NAME
	fi
        exit 0
    fi
fi

echo "run_docker.sh : starts the container in detached mode"
echo
echo "Options:"
echo "  -it:	Starts the container in interactive mode with a bash shell instad"
exit 1


