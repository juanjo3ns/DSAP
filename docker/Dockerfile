FROM python:3.7.4
WORKDIR /home
ADD requirements.txt .
RUN apt update -y; apt install -y \
	python-pip
RUN pip install -r requirements.txt
