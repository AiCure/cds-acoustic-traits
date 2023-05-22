# README
# This Dockerfile is configured to create a docker container that can be run
# on AWS Batch as described at:
# https://aicure.atlassian.net/wiki/spaces/CDS/pages/2703622145/DBM+-+Batch+job+development
# IMPORTANT: It must be run from a root directory with the following structure:
# |- root
#     |- Dockerfile    <- Copy of this docker file up to the root directory
#     |-- aicurelib    <- Copy of aicurelib repo
#     |-- batch_base   <- Copy of batch_base repo
#     |-- cds-acoustic-traits <- this repo
# 

ARG AWS_ACCOUNT_ID=272510231547
ARG AWS_REGION=us-west-2
FROM python:3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Above line may be a requirement for installing opencv-python
RUN apt-get install -y cmake --upgrade
RUN apt-get update && apt-get install -y ffmpeg python-tk

RUN pip install --upgrade pip
RUN pip install awscli

RUN aws configure set default.s3.signature_version s3v4
RUN aws configure set default.region us-west-2

RUN pip install pyarrow
# Needed for logging I think. Def need this one

RUN mkdir /acoustics
WORKDIR /acoustics
#COPY requirements.txt .

COPY ./aicurelib ./aicurelib
COPY ./batch_base ./batch_base
COPY ./cds-acoustic-traits ./cds-acoustic-traits

RUN pip install -e ./aicurelib
RUN pip install -e ./batch_base

RUN pip install -r ./cds-acoustic-traits/requirements.txt

RUN pip install -e ./cds-acoustic-traits

ENTRYPOINT ["python", "./cds-acoustic-traits/acoustics/batch_run_praat.py"]
