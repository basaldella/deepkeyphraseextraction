#!/bin/bash

source variables.sh

mkdir -p /tmp/$IMAGE_NAME

WORKDIR="/home/$DOCKER_USER"
VOLUMES="-v $(pwd)/src:${WORKDIR}/src -v $(pwd)/data:${WORKDIR}/data -v $(pwd)/models:${WORKDIR}/models -v /tmp/${IMAGE_NAME}/:${WORKDIR}/tmp"

echo ${VOLUMES}

if [ $# -eq 0 ]; then

    ${DOCKER_BIN} run -it --rm --user $(id -u):$(id -g) ${VOLUMES} --name ${CONTAINER_NAME} ${IMAGE_NAME}
    exit 0
fi

if [ $# -eq 1 ]; then
    if [ "$1" == "-bg" ]; then
        ${DOCKER_BIN} run -d ${VOLUMES} --user $(id -u):$(id -g) --name ${CONTAINER_NAME} ${IMAGE_NAME}
        exit 0
    fi
fi

echo "run_docker.sh : starts the container in detached mode"
echo
echo "Options:"
echo "  -bg:	Start the container in background"
exit 1


