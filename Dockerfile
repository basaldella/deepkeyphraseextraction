# Sets up a Keras GPU-based container starting from the official tensorflow
# Docker image. Requires a Nvidia video card to run.

# Requires nvidia-docker
# https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

FROM tensorflow/tensorflow:1.12.0-gpu

WORKDIR "/"

# Prepare the environment
COPY requirements.txt /tmp/requirements.txt
RUN bash -c "pip install -r /tmp/requirements.txt && \
            rm /tmp/requirements.txt"

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# Remove tensorflow default notebooks
RUN bash -c "rm -rf /notebooks"

CMD ["/bin/bash"]