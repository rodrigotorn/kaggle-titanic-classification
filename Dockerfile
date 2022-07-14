FROM ubuntu:20.04

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip

# Copy requirements to home
COPY requirements.txt /home/requirements.txt 

# Install requirements and delete the text file
RUN pip install -r /home/requirements.txt && \
  rm /home/requirements.txt

# During runtime, execute the pipeline
CMD cd /home/kaggle-titanic-classification && \
  python3 pipeline.py
