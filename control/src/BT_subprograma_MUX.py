#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import py_trees
import py_trees_ros
import threading

from move_base_msgs.msg import MoveBaseActionGoal
from actionlib_msgs.msg import GoalID
from std_msgs.msg import Bool

import BT_Evasion_MUX
from BT_Evasion_MUX import RobotContext, monitor_teclado

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
        self.ultimo_permiso = None # Memoria del estado anterior

    def update(self):
        # Publish the state for other programs
        if self.permiso != self.ultimo_permiso:
            self.pub_hab.publish(self.permiso) 
            self.ultimo_permiso = self.permiso
            
            if self.permiso:
                estado_str = "HABILITADA" 
            else:
                estado_str = "BLOQUEADA"
            rospy.loginfo_throttle(1.0,"[SEGURIDAD] Navegación Externa: {}".format(estado_str))

        return py_trees.common.Status.SUCCESS
        
class Persona_detectada(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(Persona_detectada, self).__init__(name)
        self.robot = robot_context

    def update(self):
        if self.robot.ve_persona:
            rospy.loginfo_throttle(1.0,"[MISIÓN] Persona Detectada")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class Espera(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(Espera, self).__init__(name)
        self.robot = robot_context

    def initialise(self):
        # Frenazo explícito por seguridad: se ejecuta SOLO UNA VEZ al entrar al estado
        self.robot.detener()
        rospy.loginfo_throttle(2.0,"[MISIÓN] Freno de seguridad activado.")
        
    def update(self):
        self.robot.detener()
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
        else:
            return py_trees.common.Status.FAILURE

class Sigue_persona(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(Sigue_persona, self).__init__(name)
        self.robot = robot_context
        self.distancia_objetivo = 1.0  
        self.kp_lineal = 0.5           
        self.kp_angular = 0.8          
        self.tiempo_perdido = 0.0      

    # ---> ¡AÑADE ESTO! Se ejecuta justo cuando el nodo pasa a estar activo <---
    def initialise(self):
        self.tiempo_perdido = 0.0

    def update(self):
        if self.robot.ve_persona and self.robot.persona_pos is not None:
            self.tiempo_perdido = 0.0  
            rospy.loginfo_throttle(1.0, "[MISIÓN] Siguiendo Persona")
            
            error_lineal = self.robot.persona_pos.x - self.distancia_objetivo
            error_angular = self.robot.persona_pos.y 
            
            v_x = error_lineal * self.kp_lineal
            v_z = error_angular * self.kp_angular
            
            v_x = max(-0.2, min(0.4, v_x)) 
            v_z = max(-0.5, min(0.5, v_z))

            if abs(error_lineal) < 0.10: v_x = 0.0
            if abs(error_angular) < 0.10: v_z = 0.0
            
            self.robot.mover(v_x, v_z)
            return py_trees.common.Status.RUNNING
            
        else:
            if self.tiempo_perdido == 0.0:
                self.tiempo_perdido = rospy.Time.now().to_sec()

            if rospy.Time.now().to_sec() - self.tiempo_perdido < 1.5:
                self.robot.detener()
                return py_trees.common.Status.RUNNING
            
            rospy.loginfo_throttle(1.0, "[MISIÓN] Persona Perdida definitivamente")
            self.robot.detener()
            return py_trees.common.Status.FAILURE


class NavegarAlDestino(py_trees.behaviour.Behaviour):
    def __init__(self, name, robot_context):
        super(NavegarAlDestino, self).__init__(name)
        self.robot = robot_context
        self.meta_enviada = False
        self.tiempo_envio = 0.0  # <--- NUEVO: Para medir el tiempo de gracia

    def initialise(self):
        self.meta_enviada = False

    def update(self):
        # 1. Si no hay meta señalada, FALLA para que el BT pase a la fase de seguir
        if self.robot.meta_destino is None:
            return py_trees.common.Status.FAILURE  

        # 2. Enviar la meta la primera vez
        if not self.meta_enviada:
            rospy.loginfo_throttle(1.0, "[MISIÓN] Enviando meta por topic /move_base_simple/goal...")
            
            self.robot.pub_goal.publish(self.robot.meta_destino)
            
            self.meta_enviada = True
            self.robot.estado_navegacion = -1 
            self.tiempo_envio = rospy.Time.now().to_sec()  # <--- NUEVO: Guardamos el momento exacto
            return py_trees.common.Status.RUNNING

        # --- NUEVO: Periodo de gracia de 1 segundo ---
        # Ignoramos cualquier estado que nos dé move_base durante el primer segundo
        if rospy.Time.now().to_sec() - self.tiempo_envio < 2.0:
            return py_trees.common.Status.RUNNING

        # 3. Analizar el estado actual (ahora sí, move_base ha tenido tiempo de actualizarse)
        estado = self.robot.estado_navegacion
        
        if estado in [0, 1, -1, 2, 8]:
            return py_trees.common.Status.RUNNING
            
        elif estado == 3:
            rospy.loginfo_throttle(1.0, "[MISIÓN] ¡Destino alcanzado! Preparando descarga.")
            self.robot.meta_destino = None 
            return py_trees.common.Status.SUCCESS
            
        else:
            rospy.logerr("[MISIÓN] Fallo en la navegación. Código de estado: {}".format(estado))
            self.robot.meta_destino = None 
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
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
    avisar_parada = GestionNavegacion("Notify_Block", robot_context, permiso=False)
    sub_arbol_rotar = BT_Evasion_MUX.construir_arbol_evas()
    rama_evasion.add_children([check_peligro, avisar_parada, sub_arbol_rotar])

    # --- BRANCH 2: NORMAL NAVIGATION ---
    rama_mision = py_trees.composites.Sequence(name="Mision_Sequence", memory=True)
    navigation_ok = GestionNavegacion("Habilita navegacion", robot_context, permiso=True)

    # ¡NUEVO!: Selector principal de misión
    mision_selector = py_trees.composites.Selector(name="Mision_Selector", memory=True)

    # A) Prioridad 1: Viajar al destino (solo si hay meta)
    place = NavegarAlDestino("Deja objeto", robot_context)

    # B) Prioridad 2: Seguir persona (solo si NO estamos yendo al destino)
    # Al tener memory=False, obliga a re-evaluar la detección en cada tick
    follow_sequence = py_trees.composites.Sequence(name="Follow_Sequence", memory=False)

    detect_person = py_trees.composites.Selector(name="Person_Detection", memory=False)
    check = Persona_detectada("Persona..", robot_context)
    wait = Espera("Espera persona", robot_context)
    detect_person.add_children([check, wait])

    detect_pointing = py_trees.composites.Selector(name="Pointing_Detection", memory=False)
    check_gesto = Gesto_detectado("Senala..", robot_context)
    follow = Sigue_persona("Sigue a la persona", robot_context)
    detect_pointing.add_children([check_gesto, follow])

    follow_sequence.add_children([detect_person, detect_pointing])

    # Unimos ambas opciones
    mision_selector.add_children([place, follow_sequence])

    # Ensamblaje final
    rama_mision.add_children([navigation_ok, mision_selector])
    root.add_children([rama_evasion, rama_mision])
    
    return root
# ==========================================
# Main Loop
# ==========================================

def main():
    rospy.init_node('robot_bt_puro')
    robot_context = RobotContext()
    
    # Inject the context into the BT_Evasion module so the original 'AccionRotar'
    # class can find the global 'robot' variable without errors.
    BT_Evasion_MUX.robot = robot_context

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