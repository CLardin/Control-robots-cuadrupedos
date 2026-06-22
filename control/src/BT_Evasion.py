#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

import rospy
import cv2 as cv
import time
import numpy as np
import py_trees
import py_trees_ros
import py_trees.display

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Bool
from actionlib_msgs.msg import GoalID, GoalStatusArray
from unitree_legged_msgs.msg import HighCmd  

# Imports for non-blocking keyboard reading 
import sys
import select
import termios
import tty
import threading

robot = None

class RobotContext:
    """
    Context Class for the Unitree Go1 robot.
    Centralizes sensor subscribers, actuator publishers, asynchronous thread 
    management for body postures, and safety flags.
    """
    def __init__(self):
        # =====================================================================
        # SUBSCRIBERS (Sensor inputs and environment state)
        # =====================================================================
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.cb_lidar)
        self.sub_persona = rospy.Subscriber('/estado_persona', Bool, self.cb_persona)
        self.sub_persona_pos = rospy.Subscriber('/posicion_persona', Point, self.cb_persona_pos)
        self.sub_gesto = rospy.Subscriber('/gesto_parada', Bool, self.cb_gesto)
        self.sub_meta = rospy.Subscriber('/meta_senalada', PoseStamped, self.cb_meta)
        self.sub_status = rospy.Subscriber('/move_base/status', GoalStatusArray, self.cb_status)
        
        # Input from the trajectory planner (Navigation velocity bridge)
        self.sub_vel_nav = rospy.Subscriber('/cmd_vel_nav', Twist, self.cb_vel_nav)

        # =====================================================================
        # PUBLISHERS (Actuator and logic control outputs)
        # =====================================================================
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_high_cmd = rospy.Publisher('/high_cmd', HighCmd, queue_size=1)
        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.pub_cancel = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
        self.pub_hab = rospy.Publisher('/habilitar_navegacion', Bool, queue_size=1)

        # =====================================================================
        # INTERNAL STATE VARIABLES (Robot data and safety)
        # =====================================================================
        # [Front, Left, Right, Rear]
        self.distancias_lidar = [10.0, 10.0, 10.0, 10.0] 
        self.umbral_libre = 0.5      # Meters to consider path clear
        self.parada_emergencia = False # Safety flag

        # Robot physical state machine tracking
        self.estado_postura = "DE_PIE"     # States: DE_PIE, TUMBANDOSE, TUMBADO, LEVANTANDOSE

        # Tracked person data
        self.ve_persona = False
        self.persona_pos = None
        self.ve_gesto = False

        # Global navigation variables
        self.meta_destino = None
        self.estado_navegacion = -1

        # Autonomous navigation logical bridge control
        self._en_navegacion_autonoma = False
        self.ultimo_permiso = None
        self.tiempo_meta_enviada = 0.0

    # =====================================================================
    # ENCAPSULATED PROPERTIES (Getters and Setters with ROS logic)
    # =====================================================================
    @property
    def en_navegacion_autonoma(self):
        """Getter: Returns whether external navigation is allowed to move the robot."""
        return self._en_navegacion_autonoma

    @en_navegacion_autonoma.setter
    def en_navegacion_autonoma(self, value):
        """
        Smart Setter: Modifies the navigation permission state.
        Publishes changes to ROS and logs safety entries only on actual state mutations.
        """
        self._en_navegacion_autonoma = value
        if value != self.ultimo_permiso:
            self.pub_hab.publish(value)
            self.ultimo_permiso = value
            estado_str = "HABILITADA" if value else "BLOQUEADA"
            rospy.loginfo("[SAFETY] External Navigation: {}".format(estado_str))

    # =====================================================================
    # PRIVATE CONTROL METHODS (Low-level control and timed sequences)
    # ===================================================================== 

    def _publicar_modo(self, mode, duration):
        """
        Streams physical mode commands to the Unitree hardware at a high frequency (50Hz),
        blocking execution for the specified duration.
        """
        cmd = HighCmd()
        cmd.head = [0xFE, 0xEF]  # Mandatory Unitree binary packet header
        cmd.levelFlag = 0x00
        cmd.velocity = [0.0, 0.0]
        cmd.yawSpeed = 0.0
        cmd.mode = mode 
        
        rate = rospy.Rate(50)
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < duration:
            if rospy.is_shutdown(): 
                break
            self.pub_high_cmd.publish(cmd)
            rate.sleep()

    def _rutina_tumbarse(self):
        """Internal Thread: Sequenced modes to safely lie the robot down on the ground."""
        self.estado_postura = "TUMBANDOSE"
        rospy.loginfo("[POSTURE] Robot going down...")
        for m in [6, 5, 7]:
            self._publicar_modo(m, 0.5)
        self.estado_postura = "TUMBADO"
        rospy.loginfo("[POSTURA] Robot resting.")

    def _rutina_levantarse(self):
        """Internal Thread: Mode choreography to transition the robot to an upright standing posture."""
        self.estado_postura = "LEVANTANDOSE"
        rospy.loginfo("[POSTURA] Robot standing up...")
        modos = [5, 6, 1, 2]
        tiempos = [1.0, 2.5, 1.0, 1.0]
        for m, t in zip(modos, tiempos):
            self._publicar_modo(m, t)
        self.estado_postura = "DE_PIE"
        rospy.loginfo("[POSTURA] Robot ready to walk.")

    # =====================================================================
    # PUBLIC CONTROL METHODS (Non-blocking Robot API)
    # =====================================================================
    def tumbarse_async(self):
        """Asynchronously triggers (Thread) the routine to lay the robot down."""
        if self.estado_postura == "DE_PIE":
            threading.Thread(target=self._rutina_tumbarse).start()

    def levantarse_async(self):
        """Asynchronously triggers (Thread) the routine to stand the robot up."""
        if self.estado_postura == "TUMBADO":
            threading.Thread(target=self._rutina_levantarse).start()

    def mover(self, x, z):
        """Publishes raw translation (x) and rotation (z) velocities to the robot."""
        vel = Twist()
        vel.linear.x = x
        vel.angular.z = z
        self.pub_cmd.publish(vel)

    def detener(self):
        """Stops the motors immediately by commanding zero velocities."""
        self.mover(0.0, 0.0)

    # =====================================================================
    # ROS CALLBACKS (Asynchronous Topic Event Management)
    # =====================================================================
    def cb_vel_nav(self, msg):
        """Velocity Bridge: Forwards velocity directly to motors if autonomous navigation is allowed and safe."""
        if self.en_navegacion_autonoma and not self.parada_emergencia:
            self.pub_cmd.publish(msg)

    def cb_lidar(self, msg):
        """Segments the 2D planar LiDAR range array into sectors and computes minimum distances."""
        ranges = np.array(msg.ranges)
        dist_lidar_temp = []

        def get_min_range_sector(centro_idx, window_size):
            """Analyzes a window around an angular index and returns the closest obstacle distance."""
            start = max(0, centro_idx - window_size)
            end = min(len(ranges), centro_idx + window_size + 1)
            sector = ranges[start:end]
            # Filter out infinite readings or values below the physical sensor limits
            validos = sector[(np.isfinite(sector)) & (sector > 0.1)]
            return float(np.min(validos)) if validos.size > 0 else 10.0

        # Spatial segmentation via angular windows
        dist_lidar_temp.append(get_min_range_sector(180, 45))  # FRONT (Center 180°)
        dist_lidar_temp.append(get_min_range_sector(90, 45))   # LEFT (Center 90°)
        dist_lidar_temp.append(get_min_range_sector(270, 45))  # RIGHT (Center 270°)
        dist_lidar_temp.append(10.0)                            # REAR (Default value due to missing data)

        self.distancias_lidar = dist_lidar_temp

    def cb_persona(self, msg):
        """Updates the target human visual presence flag."""
        self.ve_persona = msg.data

    def cb_persona_pos(self, msg):
        """Stores the relative/absolute coordinates of the tracked human."""
        self.persona_pos = msg

    def cb_gesto(self, msg):
        """Updates the gesture detection flag (e.g., hand raised to trigger a stop command)."""
        self.ve_gesto = msg.data

    def cb_meta(self, msg):
        """Registers a new external coordinate goal target."""
        self.meta_destino = msg

    def cb_status(self, msg):
        """Monitors the active execution status of the global path planner (Move Base)."""
        if len(msg.status_list) > 0:
            self.estado_navegacion = msg.status_list[-1].status

# ==========================================
# Behavior Tree (Evasion Condition)
# ==========================================
class VerificarCaminoLibre(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(VerificarCaminoLibre, self).__init__(name)

    def update(self):
        global robot
        #  If there is enough space in front, SUCCESS (ends the BT)
        if robot.distancias_lidar[0] > robot.umbral_libre:
            rospy.loginfo("[BT] Path clear. Ending evasion.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

# ==========================================
# Behavior Tree (Rotate Action)
# ==========================================
class AccionRotar(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(AccionRotar, self).__init__(name)

    def update(self):
        global robot
        # Rotate until a gap is found
        izq_lidar = robot.distancias_lidar[1]
        der_lidar = robot.distancias_lidar[2]

        rospy.loginfo_throttle(1.0, "[BT] Obstacle nearby. Rotating...")

        if der_lidar > izq_lidar:
            # Right is clearer -> Turn right (negative angular velocity)
            #rospy.loginfo("Turning right R > L {%.2f}", robot.distancias_lidar[0])
            robot.mover(0.0, -0.3) 
        elif izq_lidar > der_lidar:
            # Left is clearer -> Turn left (positive angular velocity)
            #rospy.loginfo("Turning left L > R {%.2f}", robot.distancias_lidar[0])
            robot.mover(0.0, 0.3)
        else:
            # They are equal -> Turn left by default
            #rospy.loginfo("Turning left L = R {%.2f}", robot.distancias_lidar[0])
            robot.mover(0.0, 0.3)
        return py_trees.common.Status.RUNNING

def construir_arbol_evas():
    # Structure: Fallback (Selector)
    # 1. Is path clear? -> Yes, SUCCESS (exits evasion)
    # 2. If not, Rotate -> RUNNING (stays in evasion)
    root = py_trees.composites.Selector(name="Evasion", memory=False)
    check = VerificarCaminoLibre("Camino Libre..")
    rotate = AccionRotar("Rotar")
    root.add_children([check, rotate]) # If check is SUCCESS returns, if FAILURE executes rotate
    return root

# ==========================================
# Keyboard Monitoring Thread (Optional)
# ==========================================

def getKey(timeout=0.1):
    settings = termios.tcgetattr(sys.stdin)
    key = None
    try:
        # Set terminal to 'raw' mode (direct reading)
        tty.setcbreak(sys.stdin.fileno())

        # Disable signals (ISIG)
        # This prevents Ctrl+C from killing the process and allows reading it as '\x03'
        mode = termios.tcgetattr(sys.stdin.fileno())
        mode[3] = mode[3] & ~termios.ISIG
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, mode)

        # Check if there is something in the buffer (select)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
    except Exception as e:
        print(e)
    finally:
        # Restore original terminal configuration
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def monitor_teclado():
    """
    This thread runs in parallel and checks if 's' or 'Space' is pressed.
    """
    rospy.loginfo("--- KEYBOARD MONITOR ACTIVE: Press 's' for EMERGENCY STOP ---")

    while not rospy.is_shutdown():
        # getKey has a small timeout, so the loop doesn't spin wildly
        key = getKey(timeout=0.2) 
        
        if key == 's'or key == ' ' or key == '\x03':
            rospy.loginfo("\n\n EMERGENCY STOP DETECTED (Key pressed) \n")
            
            # Block commands in robot context
            if robot:
                robot.parada_emergencia = True
                
                # Send stop commands aggressively 
                for _ in range(10):
                    robot.detener()
                    time.sleep(0.05)
            
            # Shutdown ROS
            rospy.signal_shutdown("ser pressed emergency stop")
            # Exit thread
            sys.exit(0)
