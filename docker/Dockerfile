# FROM tensorflow/tensorflow:latest-gpu

# RUN apt install -y libgl1-mesa-glx
 
# # Install needed libraries
# COPY ./requirements.txt /tmp
# RUN python -m pip install -r /tmp/requirements.txt



# BASE CONTAINER
#FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
FROM pytorch/pytorch:latest

# APT INSTALLS
RUN apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev git
RUN apt -y install python3-pip

# PIP
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /tmp
RUN python3 -m pip install -r /tmp/requirements.txt

# APP DIRECTORY PATH
WORKDIR /app
