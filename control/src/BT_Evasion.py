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

import smach
import smach_ros

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# --- Imports for non-blocking keyboard reading ---
import sys
import select
import termios
import tty
import threading

robot = None
cv_image = None

class RobotContext:
    def __init__(self):
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.cb_lidar)
        self.sub_cam_rgb = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.cb_camaraRGB)
        #self.sub_cam_depth = rospy.Subscriber('/camera/depth/image_rect_raw/compressed', CompressedImage, self.cb_camaraD)
        
       # [Front, Left, Right, Rear]
        self.distancias_lidar = [10.0, 10.0, 10.0, 10.0] 
        self.umbral_obstaculo = 0.7  # Meters to detect collision
        self.umbral_libre = 0.8      # Meters to consider path clear
        self.parada_emergencia = False # Safety flag
        self.imagenRGB = None

    def mover(self, x, z):
        vel = Twist()
        vel.linear.x = x
        vel.angular.z = z
        self.pub_cmd.publish(vel)

    def detener(self):
        self.mover(0.0, 0.0)

    def cb_lidar(self, msg):
        # --- LIDAR logic ---
        #rospy.loginfo_throttle(1.0, "[LIDAR] Receiving array of %d puntos", len(msg.ranges))
        ranges = np.array(msg.ranges)
        dist_lidar_temp = []

        def get_min_range_sector(centro_idx, window_size):
            # Calculate index handling boundaries
            start = max(0, centro_idx - window_size)
            end = min(len(ranges), centro_idx + window_size + 1)
            sector = ranges[start:end]
            # Filter
            validos = sector[(np.isfinite(sector)) & (sector > 0.1)]
            return float(np.min(validos)) if validos.size > 0 else 10.0

        # FRONT (Center 180)
        dist_lidar_temp.append(get_min_range_sector(180, 45))
        
        # LEFT (Center 90)
        dist_lidar_temp.append(get_min_range_sector(90, 45))
        
        # RIGHT (Center 270)
        dist_lidar_temp.append(get_min_range_sector(270, 45))

        self.distancias_lidar = dist_lidar_temp
        # Throttle debug to avoid saturating the console
        #rospy.loginfo_throttle(2.0, "LIDAR: F={:.2f} L={:.2f} R={:.2f}".format(self.distancias_lidar[0], self.distancias_lidar[1], self.distancias_lidar[2]))
        #rospy.loginfo("[LIDAR CALLBACK] Calculated real Front: %.2f", self.distancias_lidar[0])

    def cb_camaraRGB(self, msg):
        try:
            # Decoding
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv.imdecode(np_arr, cv.IMREAD_COLOR)
            self.imagenRGB = cv_image
            
            if cv_image is None:
                return
          
            # rospy.loginfo("[BT] Data type: %s", type(cv_image))

        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

    def cb_camaraD(self, msg):
        try:
            # Manual decoding for 16-bits PNGs
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv.imdecode(np_arr, cv.IMREAD_ANYDEPTH)

            if cv_image is None:
                return
            
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

# ==========================================
# Behavior Tree (Evasion State)
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
# SMACH State (Optional)
# ==========================================

class EvasionState(smach.State):
    def __init__(self, robot_instance):
        smach.State.__init__(self, outcomes=['despejado', 'emergencia']) # Clear, emergency
        global robot

        robot = robot_instance
        self.arbol = construir_arbol_evas()

    def execute(self, userdata):
        rospy.loginfo("--- Iniciando Control por Behavior Tree ---")
        
        rate = rospy.Rate(10) # 10Hz for tree tick
        while not rospy.is_shutdown():
            # Check if keyboard thread triggered emergency
            if robot.parada_emergencia:
                robot.detener()
                return 'emergencia'

            # Tick the tree
            self.arbol.tick_once()
            
            # Analyze tree status
            # If the tree returns SUCCESS, it means VerificarCaminoLibre passed
            if self.arbol.status == py_trees.common.Status.SUCCESS:
                robot.detener()
                return 'despejado'
            
            rate.sleep()

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
