FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-ec2

WORKDIR /workspace

COPY src /workspace
COPY xlam /workspace

RUN pip install -r /workspace/requirements.txt


