

# -------------------------------------------------------------------------------------------------------------
# MAIN PROGRAM LOOP RUNS THE PROGRAM (AT THE END OF THE PAGE)
# CODE SECTION HAS THE FUNCTION DEFINITIONS THAT IS BEING CALLED INSIDE THE MAIN LOOP
# PROGRAM USES A BASIC STATE LOGIC TO NAVIGATE BETWEEN THESE FUNCTIONS
# SOCKET_SERVER & ULTRASONIC MODULES ARE PART OF THE PROGRAM AND SOME VARIABLES AND FUNCTIONS ARE DIRECTLY USED
# -------------------------------------------------------------------------------------------------------------


import jetson_inference
import jetson_utils
import torch
import torchvision
from torch2trt import TRTModule
from jetbot import Robot, Camera
import PIL.Image
import numpy as np
import cv2
import os
import time
import socket_server
import ultrasonic
import Jetson.GPIO as GPIO
import threading

#--------------------------------
# Model Initialization
#--------------------------------
print('creating Colavoidance Model...')
modelcol = torchvision.models.alexnet(pretrained=False)
modelcol.classifier[6] = torch.nn.Linear(modelcol.classifier[6].in_features, 2)
modelcol.load_state_dict(torch.load('models/best_model.pth'))
print('creating Roadfollowing Model...')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('models/best_steering_model_xy_trt.pth'))
device = torch.device('cuda')
modelcol = modelcol.to(device)
net = jetson_inference.detectNet(model="models/ssd-mobilenet.onnx",labels="models/labels.txt", threshold=0.5, input_blob="input_0", output_cvg="scores", output_bbox="boxes")
print('Done creating the models')

#--------------------------------
# Image preprocessing Functions
#--------------------------------
def col_preprocess(img):
    mean = 255.0 * np.array([0.485, 0.456, 0.406])
    stdev = 255.0 * np.array([0.229, 0.224, 0.225])
    normalize = torchvision.transforms.Normalize(mean, stdev)
    def preprocess(camera_value):
        global device
        nonlocal normalize
        x = camera_value
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = normalize(x)
        x = x.to(device)
        x = x[None, ...]
        return x
    return preprocess(img)
def road_preprocess(img):
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
    def preprocess(image):
        global device
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device).half()
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]
    return preprocess(img)

# Module initializations
robot = Robot()
robot.left(0.08)
time.sleep(0.5)
robot.stop()
time.sleep(1)
camera = Camera(width=300, height=300, frame_rate=30)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(31, GPIO.IN)
GPIO.setup(32, GPIO.IN)

#--------------#
# Code section #
#--------------#

# Global Variables
angle = 0.0
angle_last = 0.0
turn_enable = True
turn_direction = True
turn_flag = 0
roadState = 0
mainState = 3
plakaNo = 7
framecount = 0
evenodd = 0
flagturnfirst = True
turn90speed = 0.16
turn90time = 0.66
ortaplaka = 0
endcount = 0
endflag = False
giriscount = 0
flagparketti = False
oldmsg = 'empty'
distr = 0
distl = 0
rdtime = 0
redline = 0
redcount = 0

# RED LINE DETECTION LOGIC
# -------------------------------------------------------
# Runs on a different thread to constantly check for red lines
def redlinedet():
    global redline, redcount
    while True:
        redline = (GPIO.input(31) | GPIO.input(32))
        if redline:
            redcount += 1
            socket_server.sndtx = 'KirmiziCizgiIhlali'
            socket_server.send_message()
        time.sleep(0.1)


# ROAD FOLLOWING & COLLISION AVOIDANCE FUNCTION
# -------------------------------------------------------
# Uses logic from Jetbot library road following & collision avoidance
# Road folllowing runs within the collision avoidance when orob_blocked is less than 0.95
def rfca(img):
    global angle, angle_last, turn_enable, turn_direction, robot, giriscount
    global turn_flag
    global roadState, mainState
    global turn90speed, turn90time
    global distl, distr
    global rdtime
    imageres = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
    x = imageres
    x = col_preprocess(x)
    y = modelcol(x)
    y = torch.nn.functional.softmax(y, dim=1)
    prob_blocked = float(y.flatten()[0])
    if prob_blocked < 0.95:
        turn_enable = True
        xy = model_trt(road_preprocess(imageres)).detach().float().cpu().numpy().flatten()
        x1 = xy[0]
        y1 = (0.5 - xy[1]) / 2.0
        angle = np.arctan2(x1, y1)
        pid = angle * 0.009 + (angle - angle_last) * 0.002
        angle_last = angle
        robot.left_motor.value = max(min(0.1 + pid, 1.0), 0.0)
        robot.right_motor.value = max(min(0.1 - pid, 1.0), 0.0)
        distl, distr = ultrasonic.calc()
        if roadState == 0:
            if turn_flag > 1 and (((distl+distr)//2) in [96,97]):
                robot.stop()
                time.sleep(0.5)
                robot.right(turn90speed)
                time.sleep(turn90time)
                robot.stop()
                time.sleep(0.5)
                distl, distr = ultrasonic.calc()
                while abs(distl-distr) > 1:
                    distl, distr = ultrasonic.calc()
                    if distl-distr > 0:
                        robot.right(0.08)
                        time.sleep(0.08)
                        robot.stop()
                    else:
                        robot.left(0.08)
                        time.sleep(0.08)
                        robot.stop()
                    time.sleep(0.1)
                time.sleep(0.5)  
                robot.forward(turn90speed)
                time.sleep(1)
                robot.stop()
                roadState = 1
                mainState = 1
                return
        elif roadState == 1:
            if time.time()-rdtime > 1.5:
                if (((distl+distr)//2) in [26,27,45,46,61,62,78,79]): #2.giris fazla don
                    giriscount += 1
                    robot.stop()
                    time.sleep(0.5)
                    if giriscount == 2:
                        robot.right(turn90speed)
                        time.sleep(turn90time)
                        robot.stop()
                    else:
                        robot.right(turn90speed)
                        time.sleep(turn90time)
                        robot.stop()
                    time.sleep(0.5)
                    distl, distr = ultrasonic.calc()
                    while abs(distl-distr) > 1:
                        distl, distr = ultrasonic.calc()
                        if distl-distr > 0:
                            robot.right(0.08)
                            time.sleep(0.08)
                            robot.stop()
                        else:
                            robot.left(0.08)
                            time.sleep(0.08)
                            robot.stop()
                        time.sleep(0.1)
                    time.sleep(0.5)
                    robot.forward(turn90speed)
                    time.sleep(1)
                    robot.stop()
                    mainState = 1
                    return
    else:
        time.sleep(0.3)
        turn_flag += 1
        if turn_enable:
            turn_direction = angle >= 0
            turn_enable = False
        if turn_direction:
            robot.right(0.18)
        else:
            robot.left(0.18)
        time.sleep(0.85)

# FIND PARKING SPOT FUNCTION
# -------------------------------------------------------
# Checks for each parking spot one by one
# Gets close to the parking spot each time
# alligns itself with the wall when needed
# If number of parking spots checked is > 10 returns with no parking spot found

def findPlate(frame):
    global evenodd, robot_speed, robot, giriscount, flagparketti, framecount
    global mainState
    global turn90speed, turn90time
    global distl, distr
    global rdtime
    cuda_image = jetson_utils.cudaFromNumpy(frame)
    detections = net.Detect(cuda_image)
    distl, distr = ultrasonic.calc()
    while abs(distl-distr) > 1:
        distl, distr = ultrasonic.calc()
        if distl-distr > 0:
            robot.right(0.08)
            time.sleep(0.08)
            robot.stop()
        else:
            robot.left(0.08)
            time.sleep(0.08)
            robot.stop()
        time.sleep(0.1)
    for detection in detections:
        if (detection.ClassID == plakaNo) and detection.Center[0] < 185 and detection.Center[0] > 115:
            mainState = 2
            time.sleep(1)
            return
        else:
            framecount += 1
            if framecount > 100:
                framecount = 0
                evenodd += 1
                if evenodd > 9:
                    flagparketti = False
                    socket_server.sndtx = 'plakabulunamadi'
                    print("Plaka Bulunamadi")
                    time.sleep(0.5)
                    socket_server.send_message()
                    resetProgram()
                    return
                elif evenodd % 2:                # tek girislerde
                    robot.backward(turn90speed)
                    time.sleep(1)
                    robot.stop()
                    time.sleep(0.5)
                    if giriscount == 2:
                        robot.right(turn90speed)
                        time.sleep(turn90time)
                        robot.stop()
                        time.sleep(0.5)
                        robot.right(turn90speed)
                        time.sleep(turn90time)
                        robot.stop()
                    else:
                        robot.right(turn90speed)
                        time.sleep(turn90time)
                        robot.stop()
                        time.sleep(0.5)
                        robot.right(turn90speed)
                        time.sleep(turn90time)
                        robot.stop()
                    distl, distr = ultrasonic.calc()
                    while abs(distl-distr) > 1:
                        distl, distr = ultrasonic.calc()
                        if distl-distr > 0:
                            robot.right(0.1)
                            time.sleep(0.08)
                            robot.stop()
                        else:
                            robot.left(0.1)
                            time.sleep(0.08)
                            robot.stop()
                        time.sleep(0.1)
                    time.sleep(0.5)
                    robot.forward(turn90speed)
                    time.sleep(1)
                    robot.stop()
                    time.sleep(0.5)
                else:                            # cift girislerde
                    robot.backward(turn90speed)
                    time.sleep(1)
                    robot.stop()
                    time.sleep(0.5)
                    robot.right(turn90speed)
                    time.sleep(turn90time)
                    robot.stop()
                    time.sleep(0.5)
                    mainState = 0
                    rdtime = time.time()
                    return
                
# FINAL PARK FUNCTION
# -------------------------------------------------------
# When the correct parking spot is identified
# This function handles the parking process 
# uses a simple on off controller to slowly approach the parking spot
# When it is too close to the plate and no detection is seen,
# function returns and resetProgram is called to start over


def park(frame):
    global mainState, robot, flagturnfirst, endcount, endflag, flagparketti
    cuda_image = jetson_utils.cudaFromNumpy(frame)
    detections = net.Detect(cuda_image)
    if not detections:
        if endcount > 3:
            endflag = True
            endcount = 0
        else:
            endcount += 1
    if not detections and endflag:
        robot.stop()
        socket_server.sndtx = 'parktamamlandi'
        print("Park islemi tamamlandi")
        time.sleep(0.5)
        socket_server.send_message()
        flagparketti = True
        resetProgram()
        return
    else:
        for detection in detections:
            if detection.ClassID == plakaNo:
                if detection.Center[0] > 170 and flagturnfirst:
                    robot.right(0.12)
                    time.sleep(0.4)
                    robot.stop()
                    time.sleep(0.5)
                    robot.forward(0.12)
                    time.sleep(0.6)
                    robot.stop()
                    time.sleep(0.5)
                    robot.left(0.12)
                    time.sleep(0.4)
                    robot.stop()
                    time.sleep(0.5)
                    flagturnfirst = False
                elif detection.Center[0] < 130 and flagturnfirst:
                    robot.left(0.12)
                    time.sleep(0.4)
                    robot.stop()
                    time.sleep(0.5)
                    robot.forward(0.12)
                    time.sleep(0.6)
                    robot.stop()
                    time.sleep(0.5)
                    robot.right(0.12)
                    time.sleep(0.4)
                    robot.stop()
                    time.sleep(0.5)
                    flagturnfirst = False
                else:
                    flagturnfirst = False
                    if detection.Center[0] > 153:
                        pid = 0.022
                    elif detection.Center[0] < 147:
                        pid = -0.022
                    else:
                        pid = 0.00
                    robot.left_motor.value = max(min(0.10 + pid, 1.0), 0.0)
                    robot.right_motor.value = max(min(0.10 - pid, 1.0), 0.0)

# RESET FUNCTION
# -------------------------------------------------------
# To start over, sets the variables back to their original position

def resetProgram():
    global turn_direction, turn_enable, turn_flag, roadState, mainState, evenodd, parketmeaktif, endflag, endcount, giriscount, framecount, redcount
    turn_enable = True
    turn_direction = True
    turn_flag = 0
    roadState = 0
    mainState = 3
    evenodd = 0
    parketmeaktif = False
    endflag = False
    endcount = 0
    framecount = 0
    giriscount = 0
    socket_server.sndtx = 'ToplamKirmiziIhlal: '+ str(redcount)
    socket_server.send_message()
    redcount = 0
    time.sleep(1)

# SETUP CODE WAITS FOR PLATE NUMBER
# -------------------------------------------------------
# This function runs first, or after the resetProgram is called and another parking process is starting
# The function checks if the robot is already in a parking spot
# If necessary backs up from the parking spot before restarting the parking process
# Waits for the requested plate number

def setup():
    global turn90speed, turn90time, plakaNo
    global mainState
    global oldmsg
    if flagparketti:
        oldmsg = socket_server.msgrx
        while socket_server.msgrx == oldmsg:
            time.sleep(0.1)
        plakaNo = int(socket_server.msgrx)
        print(plakaNo)
        oldmsg = socket_server.msgrx
        time.sleep(0.5)
        robot.backward(0.12)
        time.sleep(3)
        robot.stop()
        time.sleep(0.5)
        robot.left(turn90speed)
        time.sleep(turn90time)
        robot.stop()
        time.sleep(0.5)
        mainState = 0
        return
    else:
        oldmsg = socket_server.msgrx
        while socket_server.msgrx == oldmsg:
            time.sleep(0.1)
        if socket_server.msgrx.isdigit():
            plakaNo = int(socket_server.msgrx)
        print(plakaNo)
        oldmsg = socket_server.msgrx
        time.sleep(0.5)
        mainState = 0
        return

# Configure Threads
socketthread = threading.Thread(target=socket_server.start_server, daemon=True)
redlinethread = threading.Thread(target=redlinedet, daemon=True)


#---------------------------#
#     MAIN PROGRAM LOOP     #
#---------------------------#
# -------------------------------------------------------
# Program starts with getting the new frame from the camera each loop
# Then it starts the threads if alrady not running
# And it chooses one of the 4 States each time
# The states are functions from above which handles one specific job of parking at a time
# States are defined globally and changed when needed within the functions 

while True:
    frame = camera.value
    if frame is None:
        continue
    if socketthread.is_alive() != True:
        socketthread.start()
    if redlinethread.is_alive() != True:
        redlinethread.start()
    if mainState == 3:
        setup()
    elif mainState == 0:
        rfca(frame)
    elif mainState == 1:
        findPlate(frame)
    elif mainState == 2:
        park(frame)

