# Predictive-Crypto-Trading-Algorithm
A project to build and train model to predict labels on crypto data.

## Set up and how to use it?

1. Run in local machine or in any VM. 
    1. Install `python 3.8.5`. 
    2. Install `pip`. 
    3. Install `build-essentials` in the linux using `apt install build-essential wget -y` along with `wget`.
    3. Install TA-Lib using command 
    ```
    wget https://artiya4u.keybase.pub/TA-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr && make && make install
    ```
    Please make sure that you install this otherwise the project will not run.

    3. Install `requirments.txt` using command `pip install -r requirements.txt`
    4. Run the main server file using command `python3 server.py`. This will start the production server on port `5000` of local machine. In order to change the port number open the file `server.py` and change the port number.

2. Run using docker.
    1. Install `docker` on your machine.
    2. Run command `docker build -t ml-algo .` in terminal once docker is running. This will create docker image for this repository.
    3. Run the image using command `docker run -p 5000:5000 ml-algo`. Here port is `5000` of docker is bined to `5000` port of host machine. It needs to be changed depending on the port you want to use.
