FROM ubuntu:20.04
RUN apt-get update \
    && apt-get install python3 -y \
    python3-pip \
    && apt install build-essential wget -y \
    && apt-get clean \
    && apt-get autoremove

RUN apt-get install -y wget
RUN wget https://artiya4u.keybase.pub/TA-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr && make && make install

ADD . /home/app
WORKDIR /home/app

RUN  pip3 install --upgrade pip
RUN  pip3 install -r requirements.txt
CMD ["python3","server.py"]
