#!/usr/bin/env python
import rospy
import math
import time
import numpy as np
import smach
import smach_ros
import py_trees
import py_trees_ros
import py_trees.display
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# --- Importaciones para lectura de teclado no bloqueante ---
import sys
import select
import termios
import tty
import threading

# ==========================================
# UTILIDAD: Lectura de Teclado No Bloqueante
# ==========================================
def getKey(timeout=0.1):
    settings = termios.tcgetattr(sys.stdin)
    key = None
    try:
        # Poner la terminal en modo 'raw' (lectura directa)
        tty.setcbreak(sys.stdin.fileno())
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

# ==========================================
# 1. Contexto del Robot (Hardware Abstraction)
# ==========================================
class RobotContext:
    def __init__(self):
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.cb_lidar)
        self.sub_scan = rospy.Subscriber('/camera/depth/image_rect_raw/compressed', CompressedImage, self.cb_camara)

        # Inicializar el puente de OpenCV
        self.bridge = CvBridge()
        
        # [Frente, Izquierda, Derecha, Atras]
        self.distancias_lidar = [10.0, 10.0, 10.0, 10.0] 
        self.distancias_cam = [10.0, 10.0, 10.0, 10.0] 
        self.umbral_obstaculo = 0.8  # Metros para detectar colision
        self.umbral_libre = 1.0      # Metros para considerar camino libre
        self.parada_emergencia = False # Flag de seguridad

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
        
    def cb_camara(self, img_msg):
        dist_cam_temp = []

        # COnvertir el mensaje de ROS a una matriz de OpenCV
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        cv_image = cv.imdecode(np_arr, cv.IMREAD_UNCHANGED)

        if cv_image is None:
                return

        # DImensiones y ROI
        alto, ancho = cv_image.shape
        tercio = ancho // 3
        y_inicio = alto // 4
        alto_roi = alto // 2
        y_fin = y_inicio + alto_roi

        roi_izq = cv_image[y_inicio:y_fin, 0:tercio]
        roi_cen = cv_image[y_inicio:y_fin, tercio:2*tercio]
        roi_der = cv_image[y_inicio:y_fin, 2*tercio:ancho]

        def get_min_dist(sub_img):
            # Filtro: valores entre 100mm (0.1m) y 10000mm (10m)
            # Creamos una mascara para ignorar los ceros (sin datos) y valores fuera de rango
            mask = (sub_img > 100) & (sub_img < 10000)
            
            validos = sub_img[mask]
            
            if validos.size > 0:
                # El valor minimo en mm convertido a metros
                return float(np.min(validos)) / 1000.0
            else:
                return 10.0
        
        # Orden: [0] Frente, [1] Izquierda, [2] Derecha, [3] Trasera
        dist_cam_temp.append(get_min_dist(roi_cen))
        dist_cam_temp.append(get_min_dist(roi_izq))
        dist_cam_temp.append(get_min_dist(roi_der))
        dist_cam_temp.append(10.0)

        self.distancias_cam = dist_cam_temp

        rospy.loginfo_throttle(0.5, "CAMARA [m] -> I: {:.2f} | C: {:.2f} | D: {:.2f}".format(self.distancias_cam[1], self.distancias_cam[0], self.distancias_cam[2]))


# Instancia global para que BT y SMACH accedan
robot = None 

# ==========================================
# 2. Behavior Tree (Para el estado de Evitacieon)
# ==========================================

class VerificarCaminoLibre(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(VerificarCaminoLibre, self).__init__(name)

    def update(self):
        # Si hay espacio suficiente enfrente, eXITO (termina el BT)
        if robot.distancias_lidar[0] > robot.umbral_libre and robot.distancias_cam[0] > robot.umbral_libre:
            rospy.loginfo("[BT] Camino despejado. Terminando evasieon.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class AccionRotar(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(AccionRotar, self).__init__(name)

    def update(self):
        # Girar hasta encontrar hueco
        izq_lidar = robot.distancias_lidar[1]
        der_lidar = robot.distancias_lidar[2]

        izq_cam = robot.distancias_cam[1]
        der_cam = robot.distancias_cam[2]

        rospy.loginfo_throttle(1.0, "[BT] Obstaculo cerca. Rotando...")

        if der_lidar > izq_lidar and der_cam > izq_cam:
            # Derecha esta mas libre -> Girar derecha (vel angular negativa)
            rospy.loginfo("Gira derecha der > izq {%.2f}, {%.2f}", robot.distancias_lidar[0], robot.distancias_cam[0])
            robot.mover(0.0, -0.3) 
        elif izq_lidar > der_lidar and izq_cam > der_cam:
            # Izquierda esta mas libre -> Girar izquierda (vel angular positiva)
            rospy.loginfo("Gira izquierda izq > derecha {%.2f}, {%.2f}", robot.distancias_lidar[0], robot.distancias_cam[0])
            robot.mover(0.0, 0.3)
        else:
            # Son iguales -> Girar izquierda por defecto
            rospy.loginfo("Gira izquierda izq = derecha {%.2f}, {%.2f}", robot.distancias_lidar[0], robot.distancias_cam[0])
            robot.mover(0.0, 0.3)
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

# ==========================================
# 3. Estados SMACH
# ==========================================

class EstadoNavegacion(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['obstaculo_detectado', 'preempted'])

    def execute(self, userdata):
        rospy.loginfo('>>> SMACH: NAVEGANDO (Caminando recto)')
        r = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            if robot.parada_emergencia: return 'preempted'

            # 1. Comportamiento: Caminar
            robot.mover(0.2, 0.0) # 0.2 m/s adelante

            # 2. Chequeo de Transicieon
            if robot.distancias_lidar[0] < robot.umbral_obstaculo and robot.distancias_cam[0] < robot.umbral_libre:
                rospy.logwarn("OBSTaCULO A {:.2f}m Cambiando a Evasieon.".format(robot.distancias_lidar[0]))
                robot.detener()
                return 'obstaculo_detectado'
            
            r.sleep()
        return 'preempted'

class EstadoEvitacion(smach.State):
    """
    Este estado ejecuta el Behavior Tree hasta que devuelve SUCCESS.
    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['camino_libre', 'fallo', 'preempted'])
        self.tree = py_trees_ros.trees.BehaviourTree(construir_arbol_evas())

        # --- NUEVO: Renderizar a imagen ---
        py_trees.display.render_dot_tree(self.tree.root, name="arbol_evasion")
        rospy.loginfo("Arbol renderizado y guardado como arbol_evasion.svg/png")
        # ----------------------------------
        
        self.snapshot_visitor = py_trees.visitors.SnapshotVisitor()
        self.tree.setup(timeout=15)

    def execute(self, userdata):
        rospy.loginfo('>>> SMACH: EVITACION (Ejecutando Behavior Tree)')
        r = rospy.Rate(10)

        while not rospy.is_shutdown():
            if robot.parada_emergencia: return 'preempted'

            # Tick al arbol
            self.tree.tick()
            estado_bt = self.tree.root.status

            # Si el arbol dice SUCCESS, significa que el camino esta libre
            if estado_bt == py_trees.common.Status.SUCCESS:
                robot.detener()
                return 'camino_libre'
            
            # Si dice FAILURE o RUNNING, seguimos en este estado
            if estado_bt == py_trees.common.Status.FAILURE:
                return 'fallo'

            r.sleep()
        return 'fallo'


# ==========================================
# 4. Hilo de Monitorizacion de Teclado
# ==========================================
def monitor_teclado():
    """
    Este hilo corre en paralelo y verifica si se pulsa 's' o 'Espacio'.
    """
    rospy.loginfo("--- MONITOR DE TECLADO ACTIVO: Pulsa 's' para PARADA DE EMERGENCIA ---")
    
    while not rospy.is_shutdown():
        # getKey tiene un pequeno timeout, asi que el bucle no gira a lo loco
        key = getKey(timeout=0.2) 
        
        if key == 's' or key == ' ':
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

# ==========================================
# 5. Main
# ==========================================

def main():
    global robot
    rospy.init_node('unitree_smach_bt')
    robot = RobotContext()
    
    # Esperar un poco a que lleguen datos del lidar
    rospy.sleep(1.0) 

    # --- INICIAR HILO DE TECLADO ---
    # Creamos un hilo demonio (daemon) que se cerrara si el programa principal muere
    input_thread = threading.Thread(target=monitor_teclado)
    input_thread.daemon = True 
    input_thread.start()

    # Crear Maquina de Estados
    sm = smach.StateMachine(outcomes=['MISION_TERMINADA'])

    with sm:
        # Estado 1: Navegar
        smach.StateMachine.add('NAVEGAR', 
                               EstadoNavegacion(), 
                               transitions={'obstaculo_detectado':'EVADIR',
                                            'preempted': 'MISION_TERMINADA'})
        
        # Estado 2: Evadir (BT)
        smach.StateMachine.add('EVADIR', 
                               EstadoEvitacion(), 
                               transitions={'camino_libre':'NAVEGAR',
                                            'fallo': 'MISION_TERMINADA',
                                            'preempted': 'MISION_TERMINADA'})
        
    sis = smach_ros.IntrospectionServer('visor_smach', sm, '/MAQUINA_ESTADOS')
    sis.start()

    # Ejecutar la maquina de estados (Esto bloquea el hilo principal)
    try:
        sm.execute()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        sis.stop()
        # Asegurarse de parar al salir si fue por CTRL+C normal
        if robot:
            robot.detener()
        print("Programa terminado.")

if __name__ == '__main__':
    main()
