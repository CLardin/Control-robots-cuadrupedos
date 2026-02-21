#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Autor/a: Carmen Lardin Sanchez

import rospy
import numpy as np
import cv2 as cv
from geometry_msgs.msg import Twist

from control.BT_Evasion import RobotContext
from control.subprograma import Control

redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)

redBajo2=np.array([175, 100, 20], np.uint8)
redAlto2=np.array([179, 255, 255], np.uint8)

def visualizar(elem1, elem2):
    # 3. Visualización
    cv.imshow('frame', elem1)
    cv.imshow('maskRed', elem2)
    
    # En ROS/Callbacks usa un tiempo mínimo
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

def navegar(robot):
    rospy.loginfo_throttle(1, "Navegación ciega activa...")
    if robot.imagenRGB is not None:
        rospy.loginfo("[NAVEGACION] Buscando color...")
        if detect_color(robot.imagenRGB) == "centro":
            rospy.loginfo("[NAVEGACION] Objeto CENTRO.")
            robot.mover(0.2,0.0)
        elif detect_color(robot.imagenRGB) == "izquierda":
            rospy.loginfo("[NAVEGACION] Objeto IZQUIERDA.")
            robot.mover(0.0,0.3)
        elif detect_color(robot.imagenRGB) == "derecha":
            rospy.loginfo("[NAVEGACION] Objeto DERECHA.")
            robot.mover(0.0,-0.3)
        else:
            rospy.loginfo("[NAVEGACION] NO hay objeto.")
            robot.mover(0.0,0.0)
    else:	
	    rospy.loginfo("[NAVEGACION] Aun no hay datos.")

def main():
    # 1. ESTE es ahora el nodo principal
    rospy.init_node('robot_navegacion_principal')
    
    # 2. Instanciamos los recursos del robot
    robot = RobotContext()
    
    # 3. Instanciamos la máquina de estados pasándole la función de navegación
    rospy.loginfo("Iniciando el Cerebro Híbrido desde Navegación...")
    cerebro = Control(robot, navegar)
    
    # 4. Ejecutamos (esto mantendrá el programa vivo controlando el flujo)
    try:
        cerebro.execute()
    except rospy.ROSInterruptException:
        pass
    finally:
        robot.detener()
        print("Cerrando programa con seguridad...")

if __name__ == '__main__':
    main()
