#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import csv
import time
import os
import py_trees
import py_trees_ros
import threading
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalID
from unitree_legged_msgs.msg import HighCmd


import BT_Evasion
from BT_Evasion import RobotContext, monitor_teclado

# --- INITIALIZE METRICS LOGGER ---
# CSV file to save experiment/execution data
# script_dir = os.path.dirname(os.path.abspath(__file__))
# absolute_csv_path = os.path.join(script_dir, 'metrics/follow/metrics_bt_follow(5).csv')
# csv_file = open(absolute_csv_path, mode='wb')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['Timestamp', 'Status', 'Collision_Distance']) # Evasion behavior
# csv_writer.writerow(['Timestamp', 'Status', 'Person_Detected', 'Distance']) # Follow behavior
# csv_writer.writerow(['Timestamp', 'Status', 'Gesture_Detected', 'Goal (3 - Reached, [0, 1, -1, 2, 8] - Running, Rest - Aborted)']) # Drop object behavior

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
        
class Persona_detectada(py_trees.behaviour.Behaviour):
    """Succeeds if a person is detected, fails otherwise."""
    def __init__(self, name, robot_context):
        super(Persona_detectada, self).__init__(name)
        self.robot = robot_context

    def update(self):
        if self.robot.ve_persona:
            rospy.loginfo_throttle(1.0, "[MISSION] Person Detected")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class Espera(py_trees.behaviour.Behaviour):
    """Commands the robot to lie down and rest while waiting for a person."""
    def __init__(self, name, robot_context):
        super(Espera, self).__init__(name)
        self.robot = robot_context

    def initialise(self):
        # Explicit emergency braking for safety: executes ONLY ONCE
        self.robot.detener()
        self.robot.tumbarse_async()
        rospy.loginfo_throttle(2.0, "[MISSION] Safety brake triggered.")
        
    def update(self):
        if self.robot.estado_postura == "TUMBANDOSE":
            return py_trees.common.Status.RUNNING

        rospy.loginfo_throttle(2.0, "[MISSION] Waiting. Robot resting.")
        return py_trees.common.Status.RUNNING 
    
class Gesto_detectado(py_trees.behaviour.Behaviour):
    """Succeeds if a pointing/stop gesture is detected."""
    def __init__(self, name, robot_context):
        super(Gesto_detectado, self).__init__(name)
        self.robot = robot_context

    def update(self):
        if self.robot.ve_gesto:
            rospy.loginfo_throttle(1.0, "[MISSION] Gesture Detected")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class Sigue_persona(py_trees.behaviour.Behaviour):
    """Controls the robot to dynamically track and follow a detected person."""
    def __init__(self, name, robot_context):
        super(Sigue_persona, self).__init__(name)
        self.robot = robot_context
        self.distancia_objetivo = 0.9  # Target distance
        self.kp_lineal = 0.5           # Proportional gain for linear velocity
        self.kp_angular = 0.8          # Proportional gain for angular velocity
        self.tiempo_perdido = 0.0      # Timestamp when the person was lost

    def initialise(self):
        self.tiempo_perdido = 0.0

    def update(self):
        if self.robot.estado_postura == "TUMBADO":
            self.robot.levantarse_async()
            return py_trees.common.Status.RUNNING
            
        # If still standing up, wait.
        if self.robot.estado_postura == "LEVANTANDOSE":
            rospy.loginfo_throttle(1.0, "[MISSION] Standing up...")
            return py_trees.common.Status.RUNNING
        
        if self.robot.ve_persona and self.robot.persona_pos is not None:
            self.tiempo_perdido = 0.0  
            rospy.loginfo_throttle(1.0, "[MISSION] Following Person")
            
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
            
            rospy.loginfo_throttle(1.0, "[MISSION] Person permanently lost")
            self.robot.detener()
            return py_trees.common.Status.FAILURE

class NavegarAlDestino(py_trees.behaviour.Behaviour):
    """Manages autonomous navigation using Move Base via an enabled bridge."""
    def __init__(self, name, robot_context):
        super(NavegarAlDestino, self).__init__(name)
        self.robot = robot_context
        self.meta_enviada = False
        self.tiempo_envio = 0.0  

    def initialise(self):
        self.meta_enviada = False

    def update(self):
        # If no target goal is available, close the navigation bridge and fail
        if self.robot.meta_destino is None:
            self.robot.en_navegacion_autonoma = False
            return py_trees.common.Status.FAILURE  

        # Send the goal on the first iteration
        if not self.meta_enviada:
            rospy.loginfo_throttle(1.0, "[MISSION] Sending goal via topic /move_base_simple/goal...")
            self.robot.pub_goal.publish(self.robot.meta_destino)
            self.meta_enviada = True

            self.robot.tiempo_meta_enviada = rospy.Time.now().to_sec()
            
            self.robot.estado_navegacion = -1 
            self.tiempo_envio = rospy.Time.now().to_sec()
            # Open the bridge to allow Move Base velocity commands to pass through
            self.robot.en_navegacion_autonoma = True  
            return py_trees.common.Status.RUNNING

        # Keep the bridge open while remaining in this state
        self.robot.en_navegacion_autonoma = True

        # 2-second grace period to allow the planner to start computing
        if rospy.Time.now().to_sec() - self.tiempo_envio < 2.0:
            return py_trees.common.Status.RUNNING

        # Analyze current navigation status 
        estado = self.robot.estado_navegacion
        
        if estado in [0, 1, -1, 2, 8]:
            return py_trees.common.Status.RUNNING
        elif estado == 3:
            rospy.loginfo_throttle(1.0, "[MISSION] Destination reached! Preparing unloading.")
            self.robot.tumbarse_async()
            self.robot.meta_destino = None 
            self.robot.en_navegacion_autonoma = False # Close navigation bridge
            return py_trees.common.Status.SUCCESS
        else:
            rospy.logerr("[MISSION] Navigation failed. Status code: {}".format(estado))
            self.robot.meta_destino = None 
            self.robot.en_navegacion_autonoma = False # Close navigation bridge
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Executes if the node is interrupted (e.g., by obstacle evasion)."""
        self.robot.en_navegacion_autonoma = False # Close the bridge immediately
        
        if new_status == py_trees.common.Status.INVALID:
            msg_cancel = GoalID()
            self.robot.pub_cancel.publish(msg_cancel)
            self.meta_enviada = False

# ==========================================
# Main Tree Construction
# ==========================================

def construir_arbol_principal(robot_context):
    """Assembles the hybrid Behavior Tree structure."""
    root = py_trees.composites.Selector(name="Hybrid_Control", memory=False)
    
    # --- BRANCH 1: EVASION (Priority) ---
    rama_evasion = py_trees.composites.Sequence(name="Evasion_Sequence", memory=False)
    check_peligro = CondicionPeligro("Danger Detected", robot_context)
    sub_arbol_rotar = BT_Evasion.construir_arbol_evas()
    rama_evasion.add_children([check_peligro, sub_arbol_rotar])

    # --- BRANCH 2: NORMAL NAVIGATION ---
    mision_selector = py_trees.composites.Selector(name="Mision_Selector", memory=False)

    # A) Priority 1: Travel to destination
    place = NavegarAlDestino("Deja objeto", robot_context)

    # B) Priority 2: Follow person
    follow_sequence = py_trees.composites.Sequence(name="Follow_Sequence", memory=False)

    detect_person = py_trees.composites.Selector(name="Person_Detection", memory=False)
    detect_person.add_children([Persona_detectada("Persona..", robot_context), Espera("Espera persona", robot_context)])

    detect_pointing = py_trees.composites.Selector(name="Pointing_Detection", memory=False)
    detect_pointing.add_children([Gesto_detectado("Senala..", robot_context), Sigue_persona("Sigue a la persona", robot_context)])

    follow_sequence.add_children([detect_person, detect_pointing])
    mision_selector.add_children([place, follow_sequence])

    # Final assembly
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

    rospy.loginfo("--- 100% Behavior Tree Architecture Started ---")

    rate = rospy.Rate(10) # 10 Hz for reactivity
    try:
        while not rospy.is_shutdown():
            # Keyboard safety check
            if robot_context.parada_emergencia:
                robot_context.detener()
                break

            arbol.tick()
            # current_node = arbol.tip()
            
            # if current_node is not None:
            #     current_status = current_node.name
            #     timestamp = time.time()
                
            #     # Evasion Behavior 
            #     # (Assuming BT_Evasion nodes are named "Danger Detected" or similar)
            #     if "Danger" in current_status or "Rotar" in current_status: 
            #         distance = robot_context.distancias_lidar[0] if hasattr(robot_context, 'distancias_lidar') else 0.0
            #         csv_writer.writerow([timestamp, current_status, distance])

            #         csv_file.flush()

            #     # Follow Behavior
            #     elif current_status == "Sigue a la persona":
            #         # Z corresponds to the distance in y according to your code (error_angular)
            #         z_dist = robot_context.persona_pos.x if robot_context.persona_pos else 0.0
            #         csv_writer.writerow([timestamp, current_status, robot_context.ve_persona, z_dist])

            #         csv_file.flush()

            #     # Drop Object / Navigation Behavior
            #     elif current_status == "Deja objeto":
            #         csv_writer.writerow([timestamp, current_status, robot_context.ve_gesto, robot_context.estado_navegacion])

            #         csv_file.flush()
                    
            #     # Other states (Wait, Person.., etc)
            #     else:
            #         csv_writer.writerow([timestamp, current_status, '-', '-'])

            #         csv_file.flush()
            
            rate.sleep()
    finally:
        # Final cleanup
        #csv_file.close()
        robot_context.detener()
        #rospy.loginfo("Files saved and program terminated successfully.")
        print("Program terminated successfully.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass