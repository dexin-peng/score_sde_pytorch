FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install git git-lfs openssh-server -y
RUN apt clean && apt purge -y

RUN pip install --upgrade pip
RUN pip install tqdm torch-tb-profiler torch_ema ml_collections pytorch_fid

RUN ln -s /opt/conda/bin/python3 /usr/local/bin/python3

RUN git lfs install

RUN mkdir -p /root/.ssh
RUN echo "your public key" >> /root/.ssh/authorized_keys

RUN mkdir /score_sde_pytorch
WORKDIR /score_sde_pytorch
CMD ["/bin/bash", "-c", "service ssh start && sleep infinity"]
