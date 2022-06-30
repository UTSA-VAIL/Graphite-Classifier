# FROM python:3.8.10

# ADD main.py .

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# CMD [ "python3", "./main.py" ]

##### fixes?

# base container
FROM pytorch/pytorch:latest

# apt installs
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev git
RUN apt -y install python3-pip

# pip
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /tmp
RUN python3 -m pip install -r /tmp/requirements.txt

# app directory path
WORKDIR /app

#docker run -it -v "$(pwd)":/app resnet