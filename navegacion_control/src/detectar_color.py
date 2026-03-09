#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Autor/a: Carmen Lardin Sanchez

import rospy
import numpy as np
import cv2 as cv
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

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
        # 1. Corregimos el desempaquetado de la imagen
        alto, ancho, _ = cv_image.shape
        tercio = ancho // 3

        frameHSV = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
        maskRed1 = cv.inRange(frameHSV, redBajo1, redAlto1) 
        maskRed2 = cv.inRange(frameHSV, redBajo2, redAlto2)
        maskRed = cv.add(maskRed1, maskRed2)
        
        _, contornos, _ = cv.findContours(maskRed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	
	    # Visualizar la imagen recivida, mayor coste
        #visualizar(cv_image, maskRed)
        
        # Variable por defecto si no encontramos nada
        posicion = "ninguna"

        if len(contornos) > 0:
            # Encontramos el contorno más grande de todos
            c = max(contornos, key=cv.contourArea)
            area = cv.contourArea(c)
            
            if area > 1000:
                # Encontramos la coordenada X del centro del objeto
                M = cv.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) # Suma de todas las posiciones en X entre el numero total de pixeles (Da el centro)
                    
                    # Comprobamos en qué zona cae el centro 'cx'
                    if cx < tercio:
                        posicion = "izquierda"
                    elif cx > (2 * tercio):
                        posicion = "derecha"
                    else:
                        posicion = "centro"
                        
                    rospy.loginfo_throttle(1, "[CAMARA] Objeto visto")

        return posicion

    except Exception as e:
        rospy.logerr("Error procesando imagen: %s", str(e))
        return "error"
    
def navegar():
    global navegacion_habilitada, imagenRGB
    rate = rospy.Rate(10) # 10 Hz
    rospy.loginfo_throttle(1, "Navegación ciega activa...")

    while not rospy.is_shutdown():
        #if navegacion_habilitada:
        if imagenRGB is not None:
            rospy.loginfo("[NAVEGACION] Buscando color...")
            if detect_color(imagenRGB) == "centro":
                rospy.loginfo("[NAVEGACION] Objeto CENTRO.")
                mover(0.2,0.0)
            elif detect_color(imagenRGB) == "izquierda":
                rospy.loginfo("[NAVEGACION] Objeto IZQUIERDA.")
                mover(0.0,0.3)
            elif detect_color(imagenRGB) == "derecha":
                rospy.loginfo("[NAVEGACION] Objeto DERECHA.")
                mover(0.0,-0.3)
            else:
                rospy.loginfo("[NAVEGACION] NO hay objeto.")
                mover(0.0,0.0)
        else:	
            rospy.loginfo("[NAVEGACION] Aun no hay datos.")
        #else:
        #    rospy.logdebug_throttle(2, "Navegación pausada. BT al mando.")
        
        rate.sleep()

def cb_camara(msg):
        global imagenRGB
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            imagenRGB = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr("Error procesando imagen: %s", str(e))

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

    # 1. ESTE es ahora el nodo principal
    rospy.init_node('robot_navegacion_principal')

    pub_cmd = rospy.Publisher('/cmd_vel_nav', Twist, queue_size=1)
    sub_cam = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, cb_camara)
    sub_nav = rospy.Subscriber('/habilitar_navegacion', Bool, cb_habilitar)
    
    # 4. Ejecutamos (esto mantendrá el programa vivo controlando el flujo)
    try:
        navegar()
    except rospy.ROSInterruptException:
        pass
    finally:
        mover(0.0,0.0)
        print("Cerrando programa con seguridad...")

if __name__ == '__main__':
    main()
