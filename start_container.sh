#!/bin/bash

source variables.sh

if [ $# -eq 0 ]; then

    ${DOCKER_BIN} run -it --rm -v "$(pwd)/src":/opt/proj/src -v "$(pwd)/data":/opt/proj/data -v "$(pwd)/models":/opt/proj/models --user $(id -u):$(id -g) --name ${CONTAINER_NAME} ${IMAGE_NAME}
    exit 0
fi

if [ $# -eq 1 ]; then
    if [ "$1" == "-bg" ]; then
        ${DOCKER_BIN} run -d -v "$(pwd)/src":/opt/proj/src -v "$(pwd)/data":/opt/proj/data -v "$(pwd)/models":/opt/proj/models --user $(id -u):$(id -g) --name ${CONTAINER_NAME} ${IMAGE_NAME}
        exit 0
    fi
fi

echo "run_docker.sh : starts the container in detached mode"
echo
echo "Options:"
echo "  -bg:	Start the container in background"
exit 1


