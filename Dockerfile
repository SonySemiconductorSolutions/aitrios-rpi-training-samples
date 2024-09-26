FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install sudo --no-install-recommends \
    apt-utils \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install sudo --no-install-recommends \
    build-essential \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-setuptools \
    git \
    gnupg \
&& update-alternatives --install /usr/bin/python python /usr/bin/python3.10 100 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /work
COPY ./ /work
WORKDIR /work
RUN pip install --upgrade pip \
&& pip install --upgrade setuptools \
&& pip install .

RUN mkdir -p /home/$USER
