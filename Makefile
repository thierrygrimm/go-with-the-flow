.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := go-with-the-flow
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD | tr / _)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)

COMMA := ,
DASH := -
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)

ifeq ($(origin PORT), undefined)
  PORT = 8888
endif

# docker
ifeq ($(origin CONTAINER_NAME), undefined)
  CONTAINER_NAME := default
endif

ifeq ($(origin GPU_ID), undefined)
  GPU_ID := all
  GPU_NAME := $(GPU_ID)
else
  GPU_NAME = $(subst $(COMMA),$(DASH),$(GPU_ID))
endif

ifeq ("$(GPU)", "false")
  GPU_ARGS := --shm-size 64G
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_$(CONTAINER_NAME)
else
  GPU_ARGS := --gpus '"device=$(GPU_ID)"' --shm-size 64G --ipc=host
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_gpu_$(GPU_NAME)_$(CONTAINER_NAME)
endif

DOCKER_ARGS := -v $$PWD:/workspace/ -v $$HOME/.cache/torch:/root/.cache/torch -e WANDB_API_KEY -e WANDB_MODE -p $(PORT):8888 --rm
DOCKER_CMD := docker run $(DOCKER_ARGS) $(GPU_ARGS) $(DOCKER_CONTAINER_NAME) -it $(PROJECTNAME):$(GIT_BRANCH)

PYTHONPATH_LOCAL := PYTHONPATH="$(PROJECT_DIR)/src:$$PYTHONPATH"

###########################
# PROJECT UTILS
###########################
.PHONY: install
install:  ##@Utils install dependencies locally
	@python3 -m pip install -r requirements.txt

.PHONY: clean
clean:  ##@Utils clean caches
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store

###########################
# DOCKER
###########################
LOCAL_BASE := pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
SERVER_BASE := pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
BASE_IMAGE ?= $(SERVER_BASE)

_build:
	@echo "Build image $(GIT_BRANCH) (base=$(BASE_IMAGE))..."
	@docker build --build-arg BASE_IMAGE=$(BASE_IMAGE) -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash_local: BASE_IMAGE=$(LOCAL_BASE)
run_bash_local: _build  ##@Docker bash (local, CUDA 12.1)
	$(DOCKER_CMD) /bin/bash

run_bash_server: BASE_IMAGE=$(SERVER_BASE)
run_bash_server: _build  ##@Docker bash (server, CUDA 12.8)
	$(DOCKER_CMD) /bin/bash

###########################
# EXPERIMENTS
###########################
METHOD ?= ddpm
SIZE ?= small

.PHONY: smoke
smoke:  ##@Experiments smoke (local python, METHOD=, SIZE=)
	$(PYTHONPATH_LOCAL) python -m smoke --method $(METHOD) --size $(SIZE)

smoke_local: BASE_IMAGE=$(LOCAL_BASE)
smoke_local: _build  ##@Experiments smoke (docker, CUDA 12.1)
	$(DOCKER_CMD) python -m smoke --method $(METHOD) --size $(SIZE)

smoke_server: BASE_IMAGE=$(SERVER_BASE)
smoke_server: _build  ##@Experiments smoke (docker, CUDA 12.8)
	$(DOCKER_CMD) python -m smoke --method $(METHOD) --size $(SIZE)

###########################
# TRAINING
###########################
N_STEPS ?= 100000
RESUME ?=
TRAIN_ARGS = --method $(METHOD) --size $(SIZE) --n-steps $(N_STEPS) $(RESUME)

.PHONY: train
train:  ##@Training real training (local python, METHOD=, SIZE=, N_STEPS=, RESUME=)
	$(PYTHONPATH_LOCAL) python -m run $(TRAIN_ARGS)

train_local: BASE_IMAGE=$(LOCAL_BASE)
train_local: _build  ##@Training real training (docker, CUDA 12.1)
	$(DOCKER_CMD) python -m run $(TRAIN_ARGS)

train_server: BASE_IMAGE=$(SERVER_BASE)
train_server: _build  ##@Training real training (docker, CUDA 12.8)
	$(DOCKER_CMD) python -m run $(TRAIN_ARGS)

###########################
# EVALUATION (NLL + sample grid + FID-vs-NFE sweep with wall-clock)
###########################
N_SAMPLES_FID ?= 10000
N_SAMPLES_NLL ?= 1024
SAMPLE_BATCH ?= 512
EVAL_ARGS = --method $(METHOD) --size $(SIZE) \
            --n-samples-fid $(N_SAMPLES_FID) --n-samples-nll $(N_SAMPLES_NLL) \
            --sample-batch $(SAMPLE_BATCH)

.PHONY: eval
eval:  ##@Eval NLL + samples + FID-vs-NFE sweep (local python)
	$(PYTHONPATH_LOCAL) python -m eval $(EVAL_ARGS)

eval_local: BASE_IMAGE=$(LOCAL_BASE)
eval_local: _build  ##@Eval NLL + samples + FID-vs-NFE sweep (docker, CUDA 12.1)
	$(DOCKER_CMD) python -m eval $(EVAL_ARGS)

eval_server: BASE_IMAGE=$(SERVER_BASE)
eval_server: _build  ##@Eval NLL + samples + FID-vs-NFE sweep (docker, CUDA 12.8)
	$(DOCKER_CMD) python -m eval $(EVAL_ARGS)

plot_local: BASE_IMAGE=$(LOCAL_BASE)
plot_local: _build  ##@Eval FID-vs-NFE + FID-vs-wallclock plots (docker, CUDA 12.1)
	$(DOCKER_CMD) python -m plot --size $(SIZE)

plot_server: BASE_IMAGE=$(SERVER_BASE)
plot_server: _build  ##@Eval FID-vs-NFE + FID-vs-wallclock plots (docker, CUDA 12.8)
	$(DOCKER_CMD) python -m plot --size $(SIZE)

plot_training_local: BASE_IMAGE=$(LOCAL_BASE)
plot_training_local: _build  ##@Eval training-loss + eval-FID vs step from W&B (docker, CUDA 12.1)
	$(DOCKER_CMD) python -m plot_training --size $(SIZE)

plot_training_server: BASE_IMAGE=$(SERVER_BASE)
plot_training_server: _build  ##@Eval training-loss + eval-FID vs step from W&B (docker, CUDA 12.8)
	$(DOCKER_CMD) python -m plot_training --size $(SIZE)

TRAJ_ARGS = --method $(METHOD) --size $(SIZE)

trajectory_local: BASE_IMAGE=$(LOCAL_BASE)
trajectory_local: _build  ##@Eval noise->image trajectory grid at low/mid/high NFE (docker, CUDA 12.1)
	$(DOCKER_CMD) python -m trajectory $(TRAJ_ARGS)

trajectory_server: BASE_IMAGE=$(SERVER_BASE)
trajectory_server: _build  ##@Eval noise->image trajectory grid at low/mid/high NFE (docker, CUDA 12.8)
	$(DOCKER_CMD) python -m trajectory $(TRAJ_ARGS)
