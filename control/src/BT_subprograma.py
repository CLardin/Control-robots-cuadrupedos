#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Autor/a: Carmen Lardin Sanchez

import rospy
import py_trees
import py_trees_ros
import threading
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

import BT_Evasion
from BT_Evasion import RobotContext, monitor_teclado

# ==========================================
# Nodos de Comportamiento Personalizados
# ==========================================

class CondicionPeligro(py_trees.behaviour.Behaviour):
    """Falla si el camino está libre, Éxito si hay peligro."""
    def __init__(self, name, robot_context):
        super(CondicionPeligro, self).__init__(name)
        self.robot = robot_context
        self.evadiendo = False 

    def update(self):
        dist_frente = self.robot.distancias_lidar[0]
        if not self.evadiendo:
            if dist_frente < self.robot.umbral_libre:
                rospy.logwarn("--- MONITOR: Peligro de choque detectado ---")
                self.evadiendo = True
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
        else:
            if dist_frente > self.robot.umbral_libre:
                self.evadiendo = False
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.SUCCESS

class GestionNavegacion(py_trees.behaviour.Behaviour):
    """
    Doble función: 
    1. Publica si la navegación está permitida en /habilitar_navegacion.
    2. Si está permitida, actúa como puente para /cmd_vel_nav.
    """
    def __init__(self, name, robot_context, permiso):
        super(GestionNavegacion, self).__init__(name)
        self.robot = robot_context
        self.permiso = permiso # True o False
        self.pub_hab = rospy.Publisher('/habilitar_navegacion', Bool, queue_size=1)
        self.sub_ext = rospy.Subscriber('/cmd_vel_nav', Twist, self._cb_vel)
        self.vel_externa = Twist()

    def _cb_vel(self, msg):
        self.vel_externa = msg

    def update(self):
        # Publicamos el estado para los otros programas
        self.pub_hab.publish(self.permiso) 

        if self.permiso:
            # Si hay permiso, el BT deja pasar la velocidad al robot real
            self.robot.pub_cmd.publish(self.vel_externa)
            return py_trees.common.Status.RUNNING
        else:
            # Si no hay permiso (estamos en rama evasión), este nodo solo avisa y termina
            return py_trees.common.Status.SUCCESS

# ==========================================
# Construcción del Árbol Principal
# ==========================================

def construir_arbol_principal(robot_context):
    root = py_trees.composites.Selector(name="Control_Hibrido", memory=False)
    
    # --- RAMA 1: EVASIÓN (Prioridad) ---
    rama_evasion = py_trees.composites.Sequence(name="Secuencia_Evasion", memory=False)
    
    check_peligro = CondicionPeligro("Peligro Detectado", robot_context)
    
    # Avisamos que la navegación externa debe PARAR
    avisar_parada = GestionNavegacion("Avisar_Bloqueo", robot_context, permiso=False)
    
    # El sub-árbol de rotación toma el control de /cmd_vel
    sub_arbol_rotar = BT_Evasion.construir_arbol_evas()
    
    rama_evasion.add_children([check_peligro, avisar_parada, sub_arbol_rotar])

    # --- RAMA 2: NAVEGACIÓN NORMAL ---
    # Avisa que se puede navegar Y deja pasar los comandos de /cmd_vel_teclado
    rama_navegacion = GestionNavegacion("Permitir_Nav_Externa", robot_context, permiso=True)

    root.add_children([rama_evasion, rama_navegacion])
    
    return root
# ==========================================
# Bucle Principal
# ==========================================

def main():
    rospy.init_node('robot_bt_puro')
    robot_context = RobotContext()
    
    # Inyectamos el contexto en el módulo BT_Evasion para que tu clase 'AccionRotar' 
    # original encuentre la variable global 'robot' sin errores.
    BT_Evasion.robot = robot_context

    # Hilo de monitorización del teclado (Parada de Emergencia)
    hilo_teclado = threading.Thread(target=monitor_teclado)
    hilo_teclado.daemon = True
    hilo_teclado.start()

    # Construimos y configuramos el árbol
    arbol_raiz = construir_arbol_principal(robot_context)

    arbol = py_trees_ros.trees.BehaviourTree(arbol_raiz)
    arbol.setup(timeout=15)

    rospy.loginfo("--- Arquitectura 100% Behavior Tree Iniciada ---")

    rate = rospy.Rate(10) # 10 Hz para la reactividad
    while not rospy.is_shutdown():
        # Verificación de seguridad por teclado
        if robot_context.parada_emergencia:
            robot_context.detener()
            break

        # El "latido" que evalúa todo el árbol de arriba a abajo
        arbol.tick()
        
        rate.sleep()

    # Limpieza final
    robot_context.detener()
    print("Programa terminado correctamente.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass