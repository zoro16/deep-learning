FROM debian:9.3

RUN apt-get update && apt-get install wget curl -y
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev  libncursesw5-dev xz-utils tk-dev

# install python 3.6
RUN wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tgz
RUN tar xvf Python-3.6.4.tgz
RUN cd Python-3.6.4 && ./configure && make && make install && cd ..


RUN pip3 install numpy pandas matplotlib tensorflow h5py keras pydot ipython

ADD . /usr/models

WORKDIR /usr/models


CMD ["/bin/bash"]
