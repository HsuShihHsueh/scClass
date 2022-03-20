FROM python:3.6

RUN apt-get update
RUN apt-get install nano
RUN pip3 install numpy
RUN pip3 install sklearn
RUN pip3 install keras==2.2.4
RUN pip3 install tensorflow==1.11.0
RUN pip3 install scanpy
RUN pip install 'h5py==2.10.0' --force-reinstall
COPY ./ /scCaps/
COPY ./autorun.sh /



##install nano
# apt-get update
# apt-get install nano