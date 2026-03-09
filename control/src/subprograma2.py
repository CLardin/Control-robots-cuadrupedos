#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import smach
import smach_ros
import threading
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from BT_Evasion import RobotContext, EvasionState, monitor_teclado

class EstadoNavegacionRemota(smach.State):
    """
    This state delegates work to the external navigation node.
    While active, it publishes True. If the LIDAR detects danger, it publishes False.
    """
    def __init__(self, robot_context):
        smach.State.__init__(self, outcomes=['finalizado', 'interrumpido'])
        self.robot = robot_context
        self.pub_hab = rospy.Publisher('/habilitar_navegacion', Bool, queue_size=1)

    def execute(self, userdata):
        rospy.loginfo("Enabling external navigation node...")
        rate = rospy.Rate(10) 
        
        while not rospy.is_shutdown():
            # If MonitorState detects danger and requests to interrupt this state...
            if self.preempt_requested():
                rospy.logwarn("Danger detected. Braking external navigation node.")
                self.pub_hab.publish(False) # Revoke permission
                self.service_preempt()
                return 'interrumpido'
            
            if self.robot.parada_emergencia:
                self.pub_hab.publish(False)
                return 'interrumpido'
            
            # If everything is OK, keep the permission active
            self.pub_hab.publish(True)
            rate.sleep()
            
        return 'finalizado'

def crear_condicion_choque(robot_instancia):
    def callback(userdata, msg):
        # Call the original function to update distances in robot.distancias_lidar
        robot_instancia.cb_lidar(msg) 
        
        # Evaluate safety using the data that cb_lidar just saved
        distancia_frontal = robot_instancia.distancias_lidar[0]
        
        if distancia_frontal < robot_instancia.umbral_obstaculo:
            rospy.logwarn("--- MONITOR: Peligro de choque detectado ---")
            return False  # Triggers the transition to Evasion (BT)
        
        return True # Everything OK, continue with normal behavior
    return callback

robot_global = None
sm_top = None

def main():
    global robot_global, sm_top 

    rospy.init_node('robot_sistema_hibrido')
    robot_global = RobotContext()
    sm_top = smach.StateMachine(outcomes=['mision_total_finalizada'])

    hilo_teclado = threading.Thread(target=monitor_teclado)
    hilo_teclado.daemon = True
    hilo_teclado.start()

    sis = smach_ros.IntrospectionServer('servidor_robot', sm_top, '/SM_ROOT')
    sis.start()

    with sm_top:
        nav_concurrente = smach.Concurrence(
            outcomes=['peligro', 'terminado'],
            default_outcome='terminado',
            child_termination_cb=lambda so: True, 
            outcome_map={'peligro': {'VIGILANTE': 'invalid'}}
        )

        with nav_concurrente:
            smach.Concurrence.add('VIGILANTE', 
                smach_ros.MonitorState("/scan", LaserScan, crear_condicion_choque(robot_global))) 
            
            # NUEVO: Usamos el estado remoto en lugar del genérico
            smach.Concurrence.add('TAREA_ACTUAL', 
                EstadoNavegacionRemota(robot_global))

        smach.StateMachine.add('ZONA_DE_TRABAJO', nav_concurrente,
                               transitions={'peligro': 'EVASION_POR_BT',
                                            'terminado': 'mision_total_finalizada'})

        smach.StateMachine.add('EVASION_POR_BT', EvasionState(robot_global),
                               transitions={'despejado': 'ZONA_DE_TRABAJO',
                                            'emergencia': 'mision_total_finalizada'})

    sm_top.execute()
    sis.stop()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        if robot_global:
            robot_global.detener()
        print("Closing program safely..")
