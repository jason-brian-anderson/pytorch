#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM pytorch/pytorch

LABEL version="0.0" maintainer="Jason Anderson" description="This is a functional pytorch dev platform"

RUN apt-get update && apt-get install -y xz-utils vim tcpdump git


COPY ../requirements.txt /requirements.txt  
#WORKDIR /

RUN conda install -y pip
RUN pip install  -r /requirements.txt
EXPOSE 80