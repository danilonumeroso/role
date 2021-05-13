FROM nvidia/cuda:10.1-devel-ubuntu18.04

COPY . /home/chess-rl
WORKDIR /home/chess-rl
ENV HOME=/home/chess-rl

RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install unzip

RUN curl -so /home/stockfish.zip https://stockfishchess.org/files/stockfish_13_linux_x64.zip \
 && unzip /home/stockfish.zip \
 && cd /home/chess-rl/stockfish_13_linux_x64/sf_13/src \
 && make net \
 && make build ARCH=x86-64-modern \
 && cd \
 && ln -s stockfish_13_linux_x64/sf_13/src/stockfish stockfish

RUN curl -so /home/installer.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x\
86_64.sh \
 && chmod +x /home/installer.sh \
 && /home/installer.sh -b -p /home/miniconda \
 && /bin/rm /home/installer.sh

RUN /home/miniconda/bin/conda env create -f /home/chess-rl/env.yaml