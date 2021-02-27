# FROM nvidia/cuda:10.2-base 
FROM tensorflow/tensorflow:1.15.5-gpu 
# -ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive




RUN apt update
RUN apt install -y python3 python3-pip
RUN pip3 install --upgrade pip

# RUN pip3 install tensorflow-gpu==1.15 scipy scikit-learn numpy
RUN pip3 install scipy scikit-learn numpy

WORKDIR /electra

