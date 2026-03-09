#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import py_trees
import py_trees_ros
import threading
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

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

# ==========================================
# Main Tree Construction
# ==========================================

def construir_arbol_principal(robot_context):
    root = py_trees.composites.Selector(name="Hybrid_Control", memory=False)
    
    # --- BRANCH 1: EVASION (Priority) ---
    rama_evasion = py_trees.composites.Sequence(name="Evasion_Sequence", memory=False)
    
    check_peligro = CondicionPeligro("Danger Detected", robot_context)
    
    # Notify that external navigation must STOP
    avisar_parada = GestionNavegacion("Notify_Block", robot_context, permiso=False)
    
    # The rotation sub-tree takes the control of /cmd_vel
    sub_arbol_rotar = BT_Evasion.construir_arbol_evas()
    
    rama_evasion.add_children([check_peligro, avisar_parada, sub_arbol_rotar])

    # --- BRANCH 2: NORMAL NAVIGATION ---
    # Notifies that navigation is allowed and forwards /cmd_vel_nav commands
    rama_navegacion = GestionNavegacion("Permitir_Nav_Externa", robot_context, permiso=True)

    root.add_children([rama_evasion, rama_navegacion])
    
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