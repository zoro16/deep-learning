FROM debian:9.3

RUN apt-get update && apt-get install wget curl -y
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev llvm libncurses5-dev  libncursesw5-dev xz-utils tk-dev \
libcupti-dev python3-numpy python3-dev python3-pip python3-wheel

# # install python 3.6
RUN wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tgz
RUN tar xvf Python-3.6.4.tgz
RUN cd Python-3.6.4 && ./configure && make && make install && cd ..

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jupyter
RUN mkdir /root/.jupyter
RUN echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
RUN python3 -m pip install numpy pandas matplotlib \
tensorflow h5py keras pydot ipython scipy pyconfig wheel six

EXPOSE 8888
EXPOSE 7777

ADD . /models

## Cleanup ##
RUN apt-get clean && apt-get autoremove

WORKDIR /models

CMD ["/bin/bash"]


#  docker run -p 8888:8888 --name deep01 --rm -it dscience:latest
# we need to run "jupyter-notebook --allow-root" from inside the container
