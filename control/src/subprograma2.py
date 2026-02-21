#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Autor/a: Carmen Lardin Sanchez

import rospy
import smach
import smach_ros
import threading
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from BT_Evasion import RobotContext, EvasionState, monitor_teclado

class EstadoNavegacionRemota(smach.State):
    """
    Este estado delega el trabajo al nodo de navegación externo.
    Mientras está activo, publica True. Si el LIDAR detecta peligro, publica False.
    """
    def __init__(self, robot_context):
        smach.State.__init__(self, outcomes=['finalizado', 'interrumpido'])
        self.robot = robot_context
        # NUEVO: Publicador para dar permiso al otro nodo
        self.pub_hab = rospy.Publisher('/habilitar_navegacion', Bool, queue_size=1)

    def execute(self, userdata):
        rospy.loginfo("Habilitando nodo de navegación externo...")
        rate = rospy.Rate(10) 
        
        while not rospy.is_shutdown():
            # 1. Si el MonitorState detecta peligro y pide interrumpir este estado...
            if self.preempt_requested():
                rospy.logwarn("Peligro detectado. Frenando nodo de navegación externo.")
                self.pub_hab.publish(False) # Quitamos permiso
                self.service_preempt()
                return 'interrumpido'
            
            if self.robot.parada_emergencia:
                self.pub_hab.publish(False)
                return 'interrumpido'
            
            # 2. Si todo está bien, mantenemos el permiso activo
            self.pub_hab.publish(True)
            rate.sleep()
            
        return 'finalizado'

def crear_condicion_choque(robot_instancia):
    def callback(userdata, msg):
        # Llamamos a la funcion original para que actualice las distancias en robot.distancias_lidar
        robot_instancia.cb_lidar(msg) 
        
        # Ahora evaluamos la seguridad usando los datos que cb_lidar acaba de guardar
        distancia_frontal = robot_instancia.distancias_lidar[0]
        
        if distancia_frontal < robot_instancia.umbral_obstaculo:
            rospy.logwarn("--- MONITOR: Peligro de choque detectado ---")
            return False  # ESTO dispara la transicion a Evasion (BT)
        
        return True # Todo OK, seguimos con el comportamiento normal
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
        # Esto se ejecuta SIEMPRE al cerrar el programa
        print("Cerrando programa con seguridad...")
        # Intentamos una ultima parada manual si el nodo sigue vivo
        # o simplemente imprimimos confirmacion.
