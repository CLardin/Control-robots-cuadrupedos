#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import cv2 as cv
import time
import numpy as np
import py_trees
import py_trees_ros
import py_trees.display

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)
redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)

navegacion_habilitada = None
imagenRGB = None
navegacion_habilitada = False
pub_cmd = None

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
    
class NavegacionActiva(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(NavegacionActiva, self).__init__(name)


    def update(self):
        global navegacion_habilitada
        # If navigation is enable, SUCCESS
        if navegacion_habilitada:
            rospy.loginfo("[BT_COLOR] Navigation active. {%s}", navegacion_habilitada)
            return py_trees.common.Status.SUCCESS
        else:
            rospy.loginfo("[BT_COLOR] Navigation NOT active. {%s}", navegacion_habilitada)
            return py_trees.common.Status.FAILURE
        
class AccionNavegar(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(AccionNavegar, self).__init__(name)

    def update(self):
        global imagenRGB

        #rospy.loginfo_throttle(1.0, "[BT_COLOR] Navigating...")

        if imagenRGB is None:
            rospy.loginfo("[NAVEGACION] Aun no hay datos.")
            return py_trees.common.Status.RUNNING

        # Call detection ONCE per tick
        resultado = detect_color(imagenRGB)
        
        if resultado == "centro":
            rospy.loginfo_throttle(1, "[BT_COLOR] Advancing to center")
            mover(0.2, 0.0)
        elif resultado == "izquierda":
            mover(0.0, 0.3)
        elif resultado == "derecha":
            mover(0.0, -0.3)
        else:
            # If there is no object, stay still waiting
            mover(0.0, 0.0)

        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """
        This method runs when the node is interrupted
        or the tree decides it should no longer execute.
        """
        rospy.loginfo("[BT_COLOR] Stopping motors due to node interruption.")
        mover(0.0, 0.0)
    
def construir_arbol_color():
    # Sequence so AccionNavegar only occurs if NavegacionActiva is SUCCESS
    root = py_trees.composites.Sequence(name="Color Tracking", memory=False)
    
    check_habilitado = NavegacionActiva("Is navigation allowed?")
    accion_seguir = AccionNavegar("Follow Object")
    
    root.add_children([check_habilitado, accion_seguir])
    return root

def stop_robot():
    """Safety function that runs when the node is shut down"""
    rospy.loginfo("Shutdown detected: Stopping motors...")
    # Create a temporary publisher in case the global one is already closed
    p = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    vel = Twist()
    # Send the stop command several times to ensure it enters the buffer
    for _ in range(5):
        p.publish(vel)
        rospy.sleep(0.05)

def main():

    global pub_cmd

    rospy.init_node('bt_seguimiento_color')

    # Register the stop function
    rospy.on_shutdown(stop_robot)

    pub_cmd = rospy.Publisher('/cmd_vel_nav', Twist, queue_size=1)
    sub_cam = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, cb_camara)
    sub_nav = rospy.Subscriber('/habilitar_navegacion', Bool, cb_habilitar)
    
    try: 
        arbol = construir_arbol_color()
        
        
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            arbol.tick_once()

            if arbol.status == py_trees.common.Status.FAILURE:
                mover(0.0, 0.0)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        mover(0.0,0.0)
        print("Closing program safely...")

if __name__ == '__main__':
    main()