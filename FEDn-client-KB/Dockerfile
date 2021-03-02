# FROM python:3.7
FROM nvcr.io/nvidia/tensorflow:21.02-tf1-py3
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git python3 python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install scipy scikit-learn numpy

RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn

COPY fedn-network.yaml /app/
# COPY requirements.txt /app/
WORKDIR /app
# RUN pip install -r requirements.txt
