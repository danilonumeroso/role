FROM nvidia/cuda:10.1-devel-ubuntu18.04

COPY . /home/chess-rl
WORKDIR /home/chess-rl

RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install unzip
RUN apt-get install make

RUN curl -so /home/stockfish https://stockfishchess.org/files/stockfish_13_linux_x64.zip \
 && unzip /home/stockfish

RUN curl -so /home/installer.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x\
86_64.sh \
 && chmod +x /home/installer.sh \
 && /home/installer.sh -b -p /home/miniconda \
 && /bin/rm /home/installer.sh

# RUN /home/miniconda/bin/conda env create -f /home/chess-rl/env.yaml