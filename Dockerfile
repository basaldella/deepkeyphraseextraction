# Sets up a Keras GPU-based container starting from the official tensorflow
# Docker image. Requires a Nvidia video card to run.

# Requires nvidia-docker
# https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

FROM tensorflow/tensorflow:1.12.0-gpu-py3

WORKDIR "/"

# Set the locale
# See https://stackoverflow.com/questions/28405902/how-to-set-the-locale-inside-a-ubuntu-docker-container
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG UNAME=docker-user
ARG GNAME=docker-grp
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# Prepare the environment
COPY requirements.txt /tmp/requirements.txt
RUN bash -c "pip install -r /tmp/requirements.txt && \
            rm /tmp/requirements.txt"

RUN bash -c "python -m nltk.downloader -d /usr/local/share/nltk_data popular"

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# Remove tensorflow default notebooks
RUN bash -c "rm -rf /notebooks"

USER $UNAME
CMD ["/bin/bash"]