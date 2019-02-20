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

RUN apt-get update && \
    apt-get install -y git

# Prepare the environment
COPY requirements.txt /tmp/requirements.txt
# NOTE: we remove eventual conflicting versions of tensorflow-gpu specified in
# the requirements file. If you want to update tensorflow please specify a new
# version in the 'FROM tensorflow ... ' line.
RUN awk '!/tensorflow-gpu==.*/' /tmp/requirements.txt > /tmp/requirements.clean.txt
RUN bash -c "pip install -r /tmp/requirements.clean.txt && \
            rm /tmp/requirements.txt"

RUN bash -c "python -m nltk.downloader -d /usr/local/share/nltk_data popular"

RUN echo '#From https://www.tensorflow.org/guide/using_gpu\
\nimport tensorflow as tf\
\n# Creates a graph.\
\na = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")\
\nb = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")\
\nc = tf.matmul(a, b)\
\n# Creates a session with log_device_placement set to True.\
\nsess = tf.Session(config=tf.ConfigProto(log_device_placement=True))\
\n# Runs the op.\
\nprint(sess.run(c))' > test_gpu.py


ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]

ARG UNAME=docker-user
ARG GNAME=docker-grp
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

# Remove tensorflow default notebooks
RUN bash -c "rm -rf /notebooks"

USER $UNAME
WORKDIR /home/$UNAME
CMD ["/bin/bash"]