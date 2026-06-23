#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import time
import py_trees
import py_trees_ros
import threading
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalID
from unitree_legged_msgs.msg import HighCmd


import BT_Evasion
from BT_Evasion import RobotContext, monitor_teclado
# ==========================================
# Custom Behavior Nodes
# ==========================================

class CondicionPeligro(py_trees.behaviour.Behaviour):
    """Fails if the path is clear, Success if danger is detected."""
    def __init__(self, name, robot_context):
        super(CondicionPeligro, self).__init__(name)
        self.robot = robot_context
        self.evadiendo = False 

    def update(self):
        dist_frente = self.robot.distancias_lidar[0]
        if not self.evadiendo:
            if dist_frente < self.robot.umbral_libre:
                rospy.logwarn("--- MONITOR: Collision danger detected ---")
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
    Dual function: 
    1. Publishes if navigation is allowed on /habilitar_navegacion.
    2. If allowed, acts as a bridge for /cmd_vel_nav.
    """
    def __init__(self, name, robot_context, permiso):
        super(GestionNavegacion, self).__init__(name)
        self.robot = robot_context
        self.permiso = permiso # True or False
        self.pub_hab = rospy.Publisher('/habilitar_navegacion', Bool, queue_size=1)
        self.sub_ext = rospy.Subscriber('/cmd_vel_nav', Twist, self._cb_vel)
        self.vel_externa = Twist()

    def _cb_vel(self, msg):
        self.vel_externa = msg

    def update(self):
        # Publish the state for other programs
        self.pub_hab.publish(self.permiso) 

        if self.permiso:
            # If permission is granted, the BT forwards the velocity to the real robot
            self.robot.pub_cmd.publish(self.vel_externa)
            return py_trees.common.Status.RUNNING
        else:
            # If no permission (we are in evasion branch), this node just notifies and finishes
            return py_trees.common.Status.SUCCESS
        
class Persona_detectada(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(Persona_detectada, self).__init__(name)
        self.robot = robot_context

    def update(self):
        if self.robot.ve_persona:
            rospy.loginfo_throttle(1.0,"[MISIÓN] Persona Detectada")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class Espera(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(Espera, self).__init__(name)
        self.robot = robot_context

    def initialise(self):
        # Frenazo explícito por seguridad: se ejecuta SOLO UNA VEZ
        self.robot.detener()
        rospy.loginfo_throttle(2.0,"[MISIÓN] Freno de seguridad activado.")
        
    def update(self):
        rospy.loginfo_throttle(2.0,"[MISIÓN] Persona NO Detectada, ESPERANDO")
        return py_trees.common.Status.RUNNING
    
class Gesto_detectado(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(Gesto_detectado, self).__init__(name)
        self.robot = robot_context

    def update(self):
        if self.robot.ve_gesto:
            rospy.loginfo_throttle(1.0,"[MISIÓN] Gesto Detectado")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class Sigue_persona(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(Sigue_persona, self).__init__(name)
        self.robot = robot_context
        self.distancia_objetivo = 1.0  
        self.kp_lineal = 0.5           
        self.kp_angular = 0.8          
        self.tiempo_perdido = 0.0    

    def initialise(self):
        self.tiempo_perdido = 0.0
        
    def update(self):
        if self.robot.ve_persona and self.robot.persona_pos is not None:
            self.tiempo_perdido = 0.0  
            rospy.loginfo_throttle(1.0, "[MISIÓN] Siguiendo Persona")
            
            error_lineal = self.robot.persona_pos.x - self.distancia_objetivo
            error_angular = self.robot.persona_pos.y 
            
            v_x = max(-0.2, min(0.4, error_lineal * self.kp_lineal)) 
            v_z = max(-0.5, min(0.5, error_angular * self.kp_angular))

            if abs(error_lineal) < 0.10: 
                v_x = 0.0
            if abs(error_angular) < 0.10:
                v_z = 0.0
            
            self.robot.mover(v_x, v_z)
            return py_trees.common.Status.RUNNING
        else:
            if self.tiempo_perdido == 0.0:
                self.tiempo_perdido = rospy.Time.now().to_sec()
            
            rospy.loginfo_throttle(1.0, "[MISIÓN] Persona Perdida definitivamente")
            self.robot.detener()
            return py_trees.common.Status.FAILURE

class NavegarAlDestino(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(NavegarAlDestino, self).__init__(name)
        self.robot = robot_context
        self.meta_enviada = False
        self.tiempo_envio = 0.0  

    def initialise(self):
        self.meta_enviada = False

    def update(self):
        # 1. Si no hay meta señalada, cerramos el puente y fallamos
        if self.robot.meta_destino is None:
            self.robot.en_navegacion_autonoma = False
            return py_trees.common.Status.FAILURE  

        # 2. Enviar la meta la primera vez
        if not self.meta_enviada:
            rospy.loginfo_throttle(1.0, "[MISIÓN] Enviando meta por topic /move_base_simple/goal...")
            self.robot.pub_goal.publish(self.robot.meta_destino)
            self.meta_enviada = True

            self.robot.tiempo_meta_enviada = rospy.Time.now().to_sec()
            
            self.robot.estado_navegacion = -1 
            self.tiempo_envio = rospy.Time.now().to_sec()
            # Abrimos el puente para que pasen las velocidades del Move Base
            self.robot.en_navegacion_autonoma = True  
            return py_trees.common.Status.RUNNING

        # Mantenemos el puente abierto mientras seguimos en este estado
        self.robot.en_navegacion_autonoma = True

        # Periodo de gracia de 2 segundos para dar tiempo a que empiece a calcular
        if rospy.Time.now().to_sec() - self.tiempo_envio < 2.0:
            return py_trees.common.Status.RUNNING

        # 3. Analizar el estado actual 
        estado = self.robot.estado_navegacion
        
        if estado in [0, 1, -1, 2, 8]:
            return py_trees.common.Status.RUNNING
        elif estado == 3:
            rospy.loginfo_throttle(1.0, "[MISIÓN] ¡Destino alcanzado! Preparando descarga.")
            self.robot.meta_destino = None 
            self.robot.en_navegacion_autonoma = False # Cerramos puente
            return py_trees.common.Status.SUCCESS
        else:
            rospy.logerr("[MISIÓN] Fallo en la navegación. Código de estado: {}".format(estado))
            self.robot.meta_destino = None 
            self.robot.en_navegacion_autonoma = False # Cerramos puente
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Se ejecuta si el nodo es interrumpido (ej. por Evasión de obstáculos)"""
        self.robot.en_navegacion_autonoma = False # Cerramos el puente de inmediato
        
        if new_status == py_trees.common.Status.INVALID:
            msg_cancel = GoalID()
            self.robot.pub_cancel.publish(msg_cancel)
            self.meta_enviada = False

# ==========================================
# Main Tree Construction
# ==========================================

def construir_arbol_principal(robot_context):
    root = py_trees.composites.Selector(name="Hybrid_Control", memory=False)
    
    # --- BRANCH 1: EVASION (Priority) ---
    rama_evasion = py_trees.composites.Sequence(name="Evasion_Sequence", memory=False)
    check_peligro = CondicionPeligro("Danger Detected", robot_context)
    sub_arbol_rotar = BT_Evasion.construir_arbol_evas()
    rama_evasion.add_children([check_peligro, sub_arbol_rotar])

    # --- BRANCH 2: NORMAL NAVIGATION ---
    mision_selector = py_trees.composites.Selector(name="Mision_Selector", memory=False)

    # A) Prioridad 1: Viajar al destino
    place = NavegarAlDestino("Deja objeto", robot_context)

    # B) Prioridad 2: Seguir persona
    follow_sequence = py_trees.composites.Sequence(name="Follow_Sequence", memory=False)

    detect_person = py_trees.composites.Selector(name="Person_Detection", memory=False)
    detect_person.add_children([Persona_detectada("Persona..", robot_context), Espera("Espera persona", robot_context)])

    detect_pointing = py_trees.composites.Selector(name="Pointing_Detection", memory=False)
    detect_pointing.add_children([Gesto_detectado("Senala..", robot_context), Sigue_persona("Sigue a la persona", robot_context)])

    follow_sequence.add_children([detect_person, detect_pointing])
    mision_selector.add_children([place, follow_sequence])

    # Ensamblaje final
    root.add_children([rama_evasion, mision_selector])
    
    return root
# ==========================================
# Main Loop
# ==========================================

def main():
    rospy.init_node('robot_bt_puro')
    robot_context = RobotContext()
    
    # Inject the context into the BT_Evasion module so the original 'AccionRotar'
    # class can find the global 'robot' variable without errors.
    BT_Evasion.robot = robot_context

    # Keyboard monitoring thread (Emergency Stop)
    hilo_teclado = threading.Thread(target=monitor_teclado)
    hilo_teclado.daemon = True
    hilo_teclado.start()

    # Build and configure the tree
    arbol_raiz = construir_arbol_principal(robot_context)

    arbol = py_trees_ros.trees.BehaviourTree(arbol_raiz)
    arbol.setup(timeout=15)

    rospy.loginfo("--- Arquitectura 100% Behavior Tree Iniciada ---")

    rate = rospy.Rate(10) # 10 Hz for reactivity
    while not rospy.is_shutdown():
        # Keyboard safety check
        if robot_context.parada_emergencia:
            robot_context.detener()
            break

        arbol.tick()
        
        rate.sleep()

    # Final cleanup
    robot_context.detener()
    print("Program terminated successfully.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass