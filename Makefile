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

DOCKER_ARGS := -v $$PWD:/workspace/ -p $(PORT):8888 --rm
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
BASE_IMAGE ?= pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

_build:
	@echo "Build image $(GIT_BRANCH) (base=$(BASE_IMAGE))..."
	@docker build --build-arg BASE_IMAGE=$(BASE_IMAGE) -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker run interactive bash
	@echo "Running bash with GPU=$(GPU) GPU_ID=$(GPU_ID)"
	$(DOCKER_CMD) /bin/bash

###########################
# EXPERIMENTS
###########################
.PHONY: smoke
smoke:  ##@Experiments run end-to-end smoke test (local, requires `make install`)
	$(PYTHONPATH_LOCAL) python -m smoke

.PHONY: smoke_docker
smoke_docker: _build  ##@Experiments run end-to-end smoke test inside docker
	$(DOCKER_CMD) python -m smoke
