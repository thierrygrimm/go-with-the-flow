ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --upgrade --no-cache-dir

COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV PYTHONPATH="/workspace/src"
