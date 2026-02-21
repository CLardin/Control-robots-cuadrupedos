#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import sys
import subprocess # Nueva libreria para lanzar comandos

distancia = []
d_izq = 10.0
d_cen = 10.0
d_der = 10.0

# Mueve los rangos de color fuera de la función para ahorrar memoria
redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)
redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)

bridge = CvBridge()

def cb_camarad(msg):
    global distancia, d_izq, d_cen, d_der
    
    # 1. CHIVATO: Si esto se imprime, la red y el topic funcionan
    rospy.loginfo_throttle(1.0, "[DEBUG] He recibido un frame de la camara.")
    
    try:
        # Decodificacion manual para PNGs de 16 bits
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv.imdecode(np_arr, cv.IMREAD_ANYDEPTH)
        
        alto, ancho = cv_image.shape
        tercio = ancho // 3
        
        y_inicio = alto // 4
        alto_roi = alto // 2
        y_fin = y_inicio + alto_roi
        
        roi_izq = cv_image[y_inicio:y_fin, 0:tercio]
        roi_cen = cv_image[y_inicio:y_fin, tercio:2*tercio]
        roi_der = cv_image[y_inicio:y_fin, 2*tercio:ancho]
        
        def get_min_dist(sub_img):
            mask = (sub_img > 100) & (sub_img < 10000)
            validos = sub_img[mask]
            if validos.size > 0:
                return float(np.min(validos)) / 1000.0
            return 10.0
                
        d_cen = get_min_dist(roi_cen)
        d_izq = get_min_dist(roi_izq)
        d_der = get_min_dist(roi_der)
        
        rospy.loginfo_throttle(1.0, "DISTANCIAS [m] -> I: {:.2f} | C: {:.2f} | D: {:.2f}".format(d_izq, d_cen, d_der))
        
    except Exception as e:
        # 2. CHIVATO: Captura cualquier tipo de error (no solo de cv_bridge)
        rospy.logerr("ERROR OCULTO EN PYTHON: %s", str(e))

def cb_camarargb(msg):
    try:
        # 1. Decodificación
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        
        if cv_image is None:
            return

        # 2. Procesamiento (Optimizado)
        frameHSV = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
        maskRed1 = cv.inRange(frameHSV, redBajo1, redAlto1)
        maskRed2 = cv.inRange(frameHSV, redBajo2, redAlto2)
        maskRed = cv.add(maskRed1, maskRed2)
        
        # bitwise_and es costoso en VM, úsalo solo si es necesario mostrarlo
        maskRedvis = cv.bitwise_and(cv_image, cv_image, mask=maskRed) 
        
        # 3. Visualización
        cv.imshow('frame', cv_image)
        cv.imshow('maskRedvis', maskRedvis)
        
        # IMPORTANTE: En ROS/Callbacks usa un tiempo mínimo
        cv.waitKey(1) 

    except Exception as e:
        # 2. CHIVATO: Captura cualquier tipo de error (no solo de cv_bridge)
        rospy.logerr("ERROR OCULTO EN PYTHON: %s", str(e))


def main():
    print("Forzando formato PNG en el robot de manera remota...")
    # Ejecutamos el comando de consola desde Python
    comando = "rosrun dynamic_reconfigure dynparam set /camera/depth/image_rect_raw/compressed format png"
    subprocess.call(comando, shell=True)
    
    print("Configuracion enviada. Iniciando nodo...")
    rospy.init_node('cerebro_navegacion')
    
    rospy.Subscriber('/camera/depth/image_rect_raw/compressed', CompressedImage, cb_camarad)
    rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, cb_camarargb)
    
    rospy.loginfo("Suscrito correctamente. Esperando la primera imagen...")
    rospy.spin()

if __name__ == '__main__':
    main()
