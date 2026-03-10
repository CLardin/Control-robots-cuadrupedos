#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import numpy as np
import cv2 as cv
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

# HSV Thresholds for Red color (handling the wrap-around at 180 degrees)
redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)
redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)

navegacion_habilitada = None
imagenRGB = None
pub_cmd = None

def visualizar(elem1, elem2):
    cv.imshow('frame', elem1)
    cv.imshow('maskRed', elem2)
    cv.waitKey(1)

def detect_color(cv_image):
    try:
        # Image dimensions and splitting into thirds for lateral detection
        alto, ancho, _ = cv_image.shape
        tercio = ancho // 3

        frameHSV = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
        maskRed1 = cv.inRange(frameHSV, redBajo1, redAlto1) 
        maskRed2 = cv.inRange(frameHSV, redBajo2, redAlto2)
        maskRed = cv.add(maskRed1, maskRed2)
        
        _, contornos, _ = cv.findContours(maskRed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	
	    # Optional: Visualize received image (higher computational cost)
        #visualizar(cv_image, maskRed)
        
        # Default position if nothing is found
        posicion = "ninguna"

        if len(contornos) > 0:
            # Find the largest contour
            c = max(contornos, key=cv.contourArea)
            area = cv.contourArea(c)
            
            if area > 1000:
                # Calculate the X coordinate of the object's center (Centroid)
                M = cv.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) # Sum of X positions divided by total pixels
                    
                    # Determine which zone the center 'cx' falls into
                    if cx < tercio:
                        posicion = "izquierda"
                    elif cx > (2 * tercio):
                        posicion = "derecha"
                    else:
                        posicion = "centro"
                        
                    rospy.loginfo_throttle(1, "[CAMERA] Object detected")

        return posicion

    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))
        return "error"
    
def navegar():
    global navegacion_habilitada, imagenRGB
    rate = rospy.Rate(10) # 10 Hz
    rospy.loginfo_throttle(1, "Blind navigation active...")

    while not rospy.is_shutdown():
        #if navegacion_habilitada:
        if imagenRGB is not None:
            rospy.loginfo("[NAVEGACION] Buscando color...")
            if detect_color(imagenRGB) == "centro":
                rospy.loginfo("[NAVIGATION] Object CENTERED.")
                mover(0.2,0.0)
            elif detect_color(imagenRGB) == "izquierda":
                rospy.loginfo("[NAVIGATION] Object LEFT.")
                mover(0.0,0.3)
            elif detect_color(imagenRGB) == "derecha":
                rospy.loginfo("[NAVIGATION] Object RIGHT.")
                mover(0.0,-0.3)
            else:
                rospy.loginfo("[NAVIGATION] No object found.")
                mover(0.0,0.0)
        else:	
            rospy.loginfo("[NAVIGATION] No data available yet.")
        #else:
        #    rospy.logdebug_throttle(2, "Pause navigation. BT in control.")
        
        rate.sleep()

def cb_camara(msg):
        global imagenRGB
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            imagenRGB = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

def cb_habilitar(msg):
    global navegacion_habilitada
    navegacion_habilitada = msg.data

def mover(x, z):
    global pub_cmd
    vel = Twist()
    vel.linear.x = x
    vel.angular.z = z
    pub_cmd.publish(vel)

def main():
    global pub_cmd
    rospy.init_node('robot_main_navigation')

    pub_cmd = rospy.Publisher('/cmd_vel_nav', Twist, queue_size=1)
    sub_cam = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, cb_camara)
    sub_nav = rospy.Subscriber('/habilitar_navegacion', Bool, cb_habilitar)

    try:
        navegar()
    except rospy.ROSInterruptException:
        pass
    finally:
        mover(0.0,0.0)
        print("Safely shutting down program...")

if __name__ == '__main__':
    main()
