# Ultrasonic sensor calculation
# using Jetson GPIO Library
# The setup has 2 ultrasonic sensors
# calc function returns the distances
# The code automatically runs with the main.py

import Jetson.GPIO as GPIO
import time

us1_trig = 11
us1_echo = 12
us2_trig = 23
us2_echo = 24

GPIO.setmode(GPIO.BOARD)
GPIO.setup(us1_trig, GPIO.OUT)
GPIO.setup(us1_echo, GPIO.IN)
GPIO.setup(us2_trig, GPIO.OUT)
GPIO.setup(us2_echo, GPIO.IN)

def get_distance(trig, echo):
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig, GPIO.LOW)
    starttime = time.time()
    endtime = time.time()
    while GPIO.input(echo) == 0:
        starttime = time.time()
    while GPIO.input(echo) == 1:
        endtime = time.time()
    timedif = endtime - starttime
    return (timedif*34300)/2

def calc():
    distl = get_distance(us1_trig, us1_echo)
    distr = get_distance(us2_trig, us2_echo)
    return distl, distr

        