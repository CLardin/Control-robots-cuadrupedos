#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Autor/a: Carmen Lardin Sanchez

import rospy
import smach
import smach_ros
import threading
from sensor_msgs.msg import LaserScan


# Importamos tus modulos
from BT_Evasion import RobotContext, EvasionState, monitor_teclado

class EstadoGenerico(smach.State):
    """
    Este estado envuelve cualquier comportamiento externo 
    que no sabe nada de SMACH.
    """
    def __init__(self, funcion_comportamiento, robot_context):
        smach.State.__init__(self, outcomes=['finalizado', 'interrumpido'])
        self.comportamiento = funcion_comportamiento
        self.robot = robot_context

    def execute(self, userdata):
        rospy.loginfo("Iniciando comportamiento externo...")
        rate = rospy.Rate(20) # 20 Hz
        
        # Ejecutamos el bucle del comportamiento
        while not rospy.is_shutdown():
            # 1. Verificamos si SMACH nos pide parar (porque el monitor detecto peligro)
            if self.preempt_requested():
                rospy.logwarn("Comportamiento interrumpido por seguridad.")
                self.service_preempt()
                return 'interrumpido'
            
            # Verificar parada de emergencia manual
            if self.robot.parada_emergencia:
                return 'interrumpido'
            
            # 2. Ejecutamos la logica que sea (navegar, mover un brazo, etc.)
            self.comportamiento(self.robot)
            
            # Un pequeno sleep para no saturar la CPU
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

class Control:
    def __init__(self, robot_global, funcion_comportamiento):
        self.robot = robot_global
        self.comportamiento = funcion_comportamiento
        self.sm_top = smach.StateMachine(outcomes=['mision_total_finalizada'])

        self._configurar_maquina()

    def _configurar_maquina(self):
        with self.sm_top:
            # --- CONTENEDOR DE CONCURRENCIA ---
            # Si el monitor falla (obstaculo), el contenedor devuelve 'peligro'
            # Si la tarea termina sola, devuelve 'terminado'
            nav_concurrente = smach.Concurrence(
                outcomes=['peligro', 'terminado'],
                default_outcome='terminado',
                child_termination_cb=lambda so: True, # Detener ambos si uno termina
                outcome_map={'peligro': {'VIGILANTE': 'invalid'}}
            )

            with nav_concurrente:
                # Anadimos el vigilante (Monitor State) usando el generador de callbacks
                smach.Concurrence.add('VIGILANTE', 
                    smach_ros.MonitorState("/scan", LaserScan, crear_condicion_choque(self.robot))) 
                
                # Anadimos la tarea ciega
                smach.Concurrence.add('TAREA_ACTUAL', 
                    EstadoGenerico(self.comportamiento, self.robot))

            # Anadimos el contenedor a la maquina principal
            smach.StateMachine.add('ZONA_DE_TRABAJO', nav_concurrente,
                                transitions={'peligro': 'EVASION_POR_BT',
                                                'terminado': 'mision_total_finalizada'})

            # Anadimos el estado de tu Behavior Tree
            smach.StateMachine.add('EVASION_POR_BT', EvasionState(self.robot),
                                transitions={'despejado': 'ZONA_DE_TRABAJO',
                                                'emergencia': 'mision_total_finalizada'})

    def execute(self):
        # Lanzar hilo de teclado en segundo plano
        hilo_teclado = threading.Thread(target=monitor_teclado)
        hilo_teclado.daemon = True
        hilo_teclado.start()

        sis = smach_ros.IntrospectionServer('servidor_robot', self.sm_top, '/SM_ROOT')
        sis.start()

        # Ejecutar
        self.sm_top.execute()
        sis.stop()
