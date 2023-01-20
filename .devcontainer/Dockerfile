FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

#FROM pytorch/pytorch

LABEL version="0.0" maintainer="Jason Anderson" description="This is a functional pytorch dev platform"

RUN apt-get update && apt-get install -y xz-utils vim tcpdump git



COPY requirements.txt /requirements.txt
WORKDIR /


#RUN pip install --user -r ./requirements.txt
RUN conda install -y jupyter matplotlib pandas scikit-learn pytest sphinx

EXPOSE 80

RUN echo "================================================================================================================================================"
RUN echo "================================================================================================================================================"
RUN echo "================================================================================================================================================"

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=80", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

#docker run --gpus all -it --rm -p 80:80 sexy

RUN echo "================================================================================================================================================"
RUN echo "================================================================================================================================================"
RUN echo "================================================================================================================================================"


#https://www.youtube.com/watch?v=jtBVppyfDbE


