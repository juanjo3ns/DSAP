FROM python:3.7.4
ENV PATH /usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"
WORKDIR /home
ADD requirements.txt .
RUN apt update -y; apt install -y \
	python-pip
RUN pip install -r requirements.txt
