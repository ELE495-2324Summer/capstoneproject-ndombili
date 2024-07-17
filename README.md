[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5mCoF9-h)
# TOBB ETÜ ELE495 - Grup Ndombili

![img1](https://github.com/user-attachments/assets/d2fb57a9-a0bb-4b38-b2c6-56392afc974b)
![img2](https://github.com/user-attachments/assets/18d783f6-a590-420c-afda-bc81bcac068e)



# Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Video](#video)
- [Acknowledgements](#acknowledgements)

## Introduction
This is a Jetbot Robot variation. It aims to navigate inside a parking course using artificial intelligence, and park into the desired parking spot.

## Features

- Hardware Components: \
2x TC3272 RGB Color sensor \
2x HC-SR04 Ultrasonic distance sensor \
2x Arduino Nano \
Jetbot Kit (wheels, csicamera, batterypack etc.)
- Operating system: \
Ubuntu 18.04.6 LTS Linux 4.9.201-tegra arm64


The Jetbot Robot is controlled and navigated using a modern web-application based phone app. This app establishes a two-way communication channel via a configured TCP socket connection between the robot and the app, ensuring minimal input lag. \

Improved Robot design uses precise distance sensors to calibrate and ensure smooth operation. \

Used Arduino microcontrollers provide more room for the heavy AI models by externally computing the sensor outputs.


## Installation
To install required packages, follow the instructions below;

```bash
git clone http://github.com/NVIDIA-AI-IOT/jetbot.git
cd jetbot
./scripts/configure_jetson.sh
./scripts/enable_swap.sh
cd docker
./enable.sh
#  The steps above can be skipped if jetson nano has been set up using jetbot image
```
Now we need to install Jetson-Inference which provides many neural network applications.
The installation has to be inside the docker container previously installed.

```bash
git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
# To see the active docker containers, sudo docker ps
sudo docker exec -it <container_id> /bin/bash
apt-get update
apt-get install git cmake libpython3-dev python3-numpy
cd jetson-inference
mkdir build
cd build
cmake ../
make -j$(nproc)
make install
ldconfig
```

Download or Clone main.py ultrasonic.py socket_server.py and models folder from this repo into the docker environement.

Models used for Road following and Collision avoidence needs to be downloaded manually from the link below (because of the large file size) \

[Download Models](https://drive.google.com/drive/folders/1K9h_5UuOM2FzX6DYbM1fD55CvJnAI5uD?usp=sharing)

The file structure should look like this,
```bash
/ / code
-main.py
-socket_server.py
-ultrasonic.py
/ models
  -best_model.pth
  -best_steering_model_xy_trt.pth
```
## Usage
To use the Robot, simplest way is hardcoding the plate number into the main.py and running the main.py using python >= 3.6.9 

To use the Robot wirelessly via the Provided Phone app.
Flet needs to be downloaded to a Computer which will then be used to host the website, and any phone sharing the same network will be able to connect 
```bash
pip install flet
```
Hosting the webapp is simple as typing to the terminal,
```bash
flet run app.py --port 5050
```
Inside the phone browser, type the local ipv4 address of the computer hosting the app. Then add the port to the end of the address \
example "ipv4ofthecomputer":5050

Inside the app, enter the ip address displayed on the Robot into the ip field and press connect.
You should see a green message indicating succesful connection. \
Enter any number between 1-10 to start the navigation process.\
If robot goes over a red line, this information is sent to the app. \
If the robot parks into the provided spot or cannot find any plates, a message is sent to the app to notify. 

## Video



https://github.com/user-attachments/assets/888a150e-0563-4d39-9787-a765b1f66573



## Acknowledgements

Jetson-Inference provided by dusty-nv on github
[Dusty](https://github.com/dusty-nv)
[Jetson-Inference](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md)
