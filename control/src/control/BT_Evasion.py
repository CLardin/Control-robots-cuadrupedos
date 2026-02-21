#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2 as cv
import time
import numpy as np
import py_trees
import py_trees_ros
import py_trees.display

import smach
import smach_ros

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# --- Importaciones para lectura de teclado no bloqueante ---
import sys
import select
import termios
import tty
import threading

robot = None
cv_image = None

class RobotContext:
    def __init__(self):
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.cb_lidar)
        self.sub_cam_rgb = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.cb_camaraRGB)
        #self.sub_cam_depth = rospy.Subscriber('/camera/depth/image_rect_raw/compressed', CompressedImage, self.cb_camaraD)
        
        # [Frente, Izquierda, Derecha, Atras]
        self.distancias_lidar = [10.0, 10.0, 10.0, 10.0] 
        self.umbral_obstaculo = 0.6  # Metros para detectar colision
        self.umbral_libre = 0.8      # Metros para considerar camino libre
        self.parada_emergencia = False # Flag de seguridad
        self.imagenRGB = None

    def mover(self, x, z):
        vel = Twist()
        vel.linear.x = x
        vel.angular.z = z
        self.pub_cmd.publish(vel)

    def detener(self):
        self.mover(0.0, 0.0)

    def cb_lidar(self, msg):
        # --- Tu leogica de LIDAR portada y corregida ---
        ranges = np.array(msg.ranges)
        dist_lidar_temp = []

        def get_min_range_sector(centro_idx, window_size):
            # Calcular indices manejando limites
            start = max(0, centro_idx - window_size)
            end = min(len(ranges), centro_idx + window_size + 1)
            sector = ranges[start:end]
            # Filtrar
            validos = sector[(np.isfinite(sector)) & (sector > 0.1)]
            return float(np.min(validos)) if validos.size > 0 else 10.0

        # NOTA: Ajusta estos indices segun tu Lidar especifico (Unitree A1 suele tener Lidar 2D o 3D)
        # Aqui asumo indices simples para el ejemplo:
        # FRENTE
        # FRENTE (Centro 180)
        dist_lidar_temp.append(get_min_range_sector(180, 45))
        
        # IZQUIERDA (Centro 90)
        dist_lidar_temp.append(get_min_range_sector(90, 45))
        
        # DERECHA (Centro 270)
        dist_lidar_temp.append(get_min_range_sector(270, 45))

        self.distancias_lidar = dist_lidar_temp
        # Debug throttle para no saturar consola
        rospy.loginfo_throttle(2.0, "LIDAR: F={:.2f} I={:.2f} D={:.2f}".format(
            self.distancias_lidar[0], self.distancias_lidar[1], self.distancias_lidar[2]))
        
    def cb_camaraRGB(self, msg):
        try:
            # Decodificación
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv.imdecode(np_arr, cv.IMREAD_COLOR)
            self.imagenRGB = cv_image
            
            if cv_image is None:
                return
          
            # rospy.loginfo("[BT] Tipo de dato: %s", type(cv_image))

        except Exception as e:
            rospy.logerr("Error procesando imagen: %s", str(e))

    def cb_camaraD(self, msg):
        try:
            # Decodificacion manual para PNGs de 16 bits
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv.imdecode(np_arr, cv.IMREAD_ANYDEPTH)

            if cv_image is None:
                return
            
        except Exception as e:
            rospy.logerr("Error procesando imagen: %s", str(e))

# ==========================================
# Behavior Tree (Para el estado de Evitacieon)
# ==========================================

class VerificarCaminoLibre(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(VerificarCaminoLibre, self).__init__(name)

    def update(self):
        global robot
        # Si hay espacio suficiente enfrente, eXITO (termina el BT)
        if robot.distancias_lidar[0] > robot.umbral_libre:
            rospy.loginfo("[BT] Camino despejado. Terminando evasieon.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class AccionRotar(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(AccionRotar, self).__init__(name)

    def update(self):
        global robot
        # Girar hasta encontrar hueco
        izq_lidar = robot.distancias_lidar[1]
        der_lidar = robot.distancias_lidar[2]

        rospy.loginfo_throttle(1.0, "[BT] Obstaculo cerca. Rotando...")

        if der_lidar > izq_lidar:
            # Derecha esta mas libre -> Girar derecha (vel angular negativa)
            rospy.loginfo("Gira derecha der > izq {%.2f}", robot.distancias_lidar[0])
            #robot.mover(0.0, -0.3) 
        elif izq_lidar > der_lidar:
            # Izquierda esta mas libre -> Girar izquierda (vel angular positiva)
            rospy.loginfo("Gira izquierda izq > derecha {%.2f}", robot.distancias_lidar[0])
            #robot.mover(0.0, 0.3)
        else:
            # Son iguales -> Girar izquierda por defecto
            rospy.loginfo("Gira izquierda izq = derecha {%.2f}", robot.distancias_lidar[0])
            #robot.mover(0.0, 0.3)
        return py_trees.common.Status.RUNNING

def construir_arbol_evas():
    # Estructura: Fallback (Selector)
    # 1. Esta libre.. -> Si, SUCCESS (SMACH sale de evasieon)
    # 2. Si no, Rotar -> RUNNING (SMACH se queda en evasieon)
    root = py_trees.composites.Selector(name="Evasion", memory=False)
    check = VerificarCaminoLibre("Camino Libre..")
    rotate = AccionRotar("Rotar")
    root.add_children([check, rotate]) # Si check es SUCCESS returnea, si es FAILURE hace rotate
    return root

class EvasionState(smach.State):
    def __init__(self, robot_instance):
        smach.State.__init__(self, outcomes=['despejado', 'emergencia'])
        global robot

        robot = robot_instance
        self.arbol = construir_arbol_evas()

    def execute(self, userdata):
        rospy.loginfo("--- Iniciando Control por Behavior Tree ---")
        
        rate = rospy.Rate(10) # 10Hz para el tick del árbol
        while not rospy.is_shutdown():
            # 1. Verificar si el hilo de teclado activó emergencia
            if robot.parada_emergencia:
                return 'emergencia'

            # 2. Hacer "Tick" al árbol
            self.arbol.tick_once()
            
            # 3. Analizar el estado del árbol
            # Si el árbol dice SUCCESS, es que VerificarCaminoLibre pasó
            if self.arbol.status == py_trees.common.Status.SUCCESS:
                robot.detener()
                return 'despejado'
            
            rate.sleep()

# ==========================================
# Hilo de Monitorizacion de Teclado
# ==========================================

def getKey(timeout=0.1):
    settings = termios.tcgetattr(sys.stdin)
    key = None
    try:
        # Poner la terminal en modo 'raw' (lectura directa)
        tty.setcbreak(sys.stdin.fileno())

        # Desactivar señales (ISIG)
        # Esto evita que Ctrl+C mate el proceso y permite leerlo como '\x03'
        mode = termios.tcgetattr(sys.stdin.fileno())
        mode[3] = mode[3] & ~termios.ISIG
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, mode)

        # Verificar si hay algo en el buffer (select)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
    except Exception as e:
        print(e)
    finally:
        # Restaurar la configuracion original de la terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def monitor_teclado():
    """
    Este hilo corre en paralelo y verifica si se pulsa 's' o 'Espacio'.
    """
    rospy.loginfo("--- MONITOR DE TECLADO ACTIVO: Pulsa 's' para PARADA DE EMERGENCIA ---")
    
    while not rospy.is_shutdown():
        # getKey tiene un pequeno timeout, asi que el bucle no gira a lo loco
        key = getKey(timeout=0.2) 
        
        if key == 's'or key == ' ' or key == '\x03':
            rospy.loginfo("\n\n PARADA DE EMERGENCIA DETECTADA (Tecla pulsada) \n")
            
            # 1. Bloquear comandos en el robot context
            if robot:
                robot.parada_emergencia = True
                
                # 2. Mandar comandos de parada agresivamente (como en tu ejemplo C++)
                for _ in range(10):
                    robot.detener()
                    time.sleep(0.05)
            
            # 3. Matar ROS
            rospy.signal_shutdown("Usuario presiono parada de emergencia")
            # 4. Salir del hilo
            sys.exit(0)
