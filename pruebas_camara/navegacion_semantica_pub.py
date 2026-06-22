#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Carmen Lardin Sanchez

from __future__ import print_function
import rospy
import cv2
import csv
import math
import numpy as np
import os, sys
import time
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Missing ultralytics library.")
    sys.exit(1)

# =============================================================
# KALMAN FILTER CLASS (X, Y, Z) (OpenCV Wrapper)
# =============================================================
class OpenCVKalmanFilter:
    def __init__(self, x, y, z):
        # 6 state variables (X, Y, Z, Vx, Vy, Vz)
        # 3 measurement variables (X, Y, Z read by the camera)
        self.kf = cv2.KalmanFilter(6, 3)
        
        # Measurement matrix (H): We only read the first 3 variables (position)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        # Transition matrix (F): Initialized here, but 'dt' is updated dynamically
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)

        # Q (Process noise covariance) and R (Measurement noise covariance)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        
        # Trust Z (depth) slightly less than X and Y
        self.kf.measurementNoiseCov = np.array([
            [50.0, 0.0, 0.0],
            [0.0, 50.0, 0.0],
            [0.0, 0.0, 150.0]
        ], np.float32)

        # Initial state (Initial position, 0 velocity)
        self.kf.statePre = np.array([[x], [y], [z], [0], [0], [0]], np.float32)
        self.kf.statePost = np.array([[x], [y], [z], [0], [0], [0]], np.float32)
        
        self.last_time = time.time()
        self.x = (x, y, z)

    def predict(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Update kinematics with the real elapsed time
        self.kf.transitionMatrix[0, 3] = dt
        self.kf.transitionMatrix[1, 4] = dt
        self.kf.transitionMatrix[2, 5] = dt
        
        # OpenCV handles the prediction step
        pred = self.kf.predict()
        self.x = (pred[0, 0], pred[1, 0], pred[2, 0])
        return self.x

    def update(self, measurement):
        # Convert the reading to the matrix structure required by OpenCV
        z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], np.float32)
        
        # OpenCV handles the correction/estimation step
        est = self.kf.correct(z)
        self.x = (est[0, 0], est[1, 0], est[2, 0])
        return self.x

# =============================================================
# KALMAN FILTER CLASS (X, Y) (OpenCV Wrapper)
# =============================================================
class MapKalmanFilter:
    """2D Kalman Filter to refine global positions of static objects in the map."""
    def __init__(self, x, y):
        # 4 state variables (X, Y, Vx, Vy), 2 measurement variables (X, Y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        
        # Process noise is very low because static objects (doors, tables) do not move on their own
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4 
        
        # Initial state
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)

    def update(self, x_meas, y_meas, reading_weight):
        # Dynamic covariance adjustment:
        # If reading weight is high (e.g., short-range LiDAR), trust the measurement more.
        noise = 1.0 / (reading_weight + 0.001)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * noise

        self.kf.predict()
        z = np.array([[x_meas], [y_meas]], np.float32)
        est = self.kf.correct(z)

        # Since these are static objects, cancel out any phantom velocity 'inertia' in Vx and Vy
        self.kf.statePost[2] = 0.0
        self.kf.statePost[3] = 0.0

        return float(est[0, 0]), float(est[1, 0])

# =============================================================
# GLOBAL VARIABLES 
# =============================================================
posicion_robot = None

# For asynchronous buffers
latest_left = None
latest_right = None
latest_scan = None

# Publishers
pub_persona = None
pub_gesto = None
pub_pos_persona = None
pub_meta = None

contador_gesto = 0

# =============================================================
# MEMORY AND MAP CONFIGURATION
# =============================================================
pose_robot = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
memoria_semantica = [] 
siguiente_id = 1       
mapa_grid = None 

# =============================================================
# CAMERA AND YOLO CONFIGURATION
# =============================================================
CALIB_PATH   = "../calibration/calibracion_go1.npz"
TARGET_CLASSES = ['person', 'chair', 'backpack', 'sports ball', 'door', 'dining table', 'dinning table', 'tv']

print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt') 
model_p = YOLO('best.pt')              
model_pose = YOLO('yolov8n-pose.pt')   

if not os.path.exists(CALIB_PATH):
    print("[ERROR] Cannot find calibration file at %s" % CALIB_PATH)
    sys.exit(1)

data = np.load(CALIB_PATH)
K_l, D_l = data['K_left'].astype(np.float64), data['D_left'].astype(np.float64)
K_r, D_r = data['K_right'].astype(np.float64), data['D_right'].astype(np.float64)
R = data['R'].astype(np.float64).reshape(3, 3)
T = data['T'].astype(np.float64).reshape(3, 1)
img_size = tuple(data['img_size'])

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, np.zeros((1, 5)), K_r, np.zeros((1, 5)), img_size, R, T, alpha=0)

fx_rect = float(P1[0, 0])                           
baseline_rect = abs(float(P2[0, 3]) / fx_rect)      
cx_rect = float(P1[0, 2])                           
cy_rect = float(P1[1, 2])                           

map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(K_l, D_l, R1, P1[:3, :3], img_size, cv2.CV_32FC1)
map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(K_r, D_r, R2, P2[:3, :3], img_size, cv2.CV_32FC1)

window_size = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=window_size,
    P1=8 * 3 * window_size**2, P2=32 * 3 * window_size**2,
    disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=50, speckleRange=16,
    mode=cv2.STEREO_SGBM_MODE_SGBM
)

# Dictionary to hold active Kalman filter trackers
kf_trackers = {}

# =============================================================
# CALLBACKS
# =============================================================
def callback_left(msg):
    global latest_left
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None: latest_left = img 

def callback_right(msg):
    global latest_right
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None: latest_right = img 

def callback_lidar(msg):
    global latest_scan
    latest_scan = msg

def callback_odom(msg):
    global pose_robot, posicion_robot
    posicion_robot = msg.pose.pose 
    pose_robot['x'] = msg.pose.pose.position.x
    pose_robot['y'] = msg.pose.pose.position.y
    pose_robot['theta'] = obtener_yaw(msg.pose.pose.orientation)

def callback_map(msg):
    global mapa_grid
    mapa_grid = msg

def es_punto_valido(x_global, y_global):
    """Ensures that the point lies within a safe zone of the map grid."""
    if mapa_grid is None: return True 
    res = mapa_grid.info.resolution
    origen_x = mapa_grid.info.origin.position.x
    origen_y = mapa_grid.info.origin.position.y
    ancho = mapa_grid.info.width
    alto = mapa_grid.info.height
    px = int((x_global - origen_x) / res)
    py = int((y_global - origen_y) / res)
    
    if 0 <= px < ancho and 0 <= py < alto:
        indice = py * ancho + px
        valor_celda = mapa_grid.data[indice]
        if valor_celda == -1: return False 
        return True
    return False

def obtener_yaw(q):
    # Used in callback_odom 
    # Quaternion to Euler conversion: Yaw computation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# === (PERPENDICULAR MATHEMATICS) ===
def is_pointing_at(p_codo, p_muneca, bbox):
    """
    Calculates the perpendicular distance from the object center to the pointing ray line.
    Returns: (distance, is_forward, center_x, center_y)
    """
    x1, y1 = p_codo
    x2, y2 = p_muneca
    
    vx = x2 - x1
    vy = y2 - y1
    
    if vx == 0 and vy == 0:
        return float('inf'), False, (0, 0)
        
    bx_min, by_min, bx_max, by_max = bbox
    cx = int((bx_min + bx_max) / 2.0)
    cy = int((by_min + by_max) / 2.0)
    
    vx_obj = cx - x1
    vy_obj = cy - y1
    
    producto_escalar = (vx * vx_obj) + (vy * vy_obj)
    hacia_adelante = producto_escalar > 0
    
    numerador = abs(vy * cx - vx * cy + (vx * y1 - vy * x1))
    denominador = math.sqrt(vy**2 + vx**2)
    distancia = numerador / denominador
    
    return distancia, hacia_adelante, (cx, cy)

def obtener_distancia_lidar(scan_data, X_filt, Z_filt):
    if scan_data is None: return -1.0
    OFFSET_LIDAR_Z = 165.0 
    angulo_obj_rad = math.atan2(-X_filt, Z_filt + OFFSET_LIDAR_Z)
    angulo_obj_grados = math.degrees(angulo_obj_rad)
    indice = int(180.0 - angulo_obj_grados)
    if 0 <= indice < len(scan_data.ranges):
        ventana = 3
        inicio = max(0, indice - ventana)
        fin = min(len(scan_data.ranges), indice + ventana + 1)
        rayos_validos = [r for r in scan_data.ranges[inicio:fin] if r > 0.1 and not math.isinf(r) and not math.isnan(r)]
        if rayos_validos:
            return min(rayos_validos) * 1000.0 - OFFSET_LIDAR_Z
    return -1.0

# =============================================================
# VISUAL PROCESSING
# =============================================================
def procesar_imagenes(img_l, img_r, scan_data):
    global pose_robot, pub_persona, pub_gesto, pub_pos_persona, pub_meta, contador_gesto

    # Remapping operations for image rectifying
    rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    disparity_map = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    
    # YOLO Inference runs
    resultados_yolo = model.track(rect_l, conf=0.35, persist=True, verbose=False)
    resultados_yolo_p = model_p.track(rect_l, conf=0.35, persist=True, verbose=False)
    resultados_pose = model_pose.track(rect_l, conf=0.50, persist=True, verbose=False)

    # Internal state tracking variables
    persona_vista = False
    gesto_detectado = False
    brazos = []
    objetos_detectados = []

    # PERSON AND ARMS EXTRACTION
    for r in resultados_pose:
        if r.boxes is None: continue
        for i, box in enumerate(r.boxes):
            persona_vista = True
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            track_id = int(box.id[0]) if box.id is not None else -1
            
            area_pixeles = (x2 - x1) * (y2 - y1)
            if area_pixeles < 1500: continue
            
            h, w = rect_l.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            # Stereo disparity calculations for human keypoint tracking
            Z_raw = -1.0
            y_bottom = int(y1 + (y2 - y1) * 0.60)                                           
            roi_disparidad = disparity_map[y1:y_bottom, x1:x2]      
            valores_validos = roi_disparidad[roi_disparidad > 0]                            

            if len(valores_validos) > 0:
                disp_cercana = np.percentile(valores_validos, 90)                           
                umbral_fondo = disp_cercana * 0.65
                pixeles_primer_plano = valores_validos[valores_validos > umbral_fondo]      
                if len(pixeles_primer_plano) > 0:
                    disp_robusta = np.median(pixeles_primer_plano)                          
                    Z_raw = (fx_rect * baseline_rect) / disp_robusta                        

            # If YOLO assigned a tracking ID (e.g. "Person #5"), use it. Otherwise, use a generic identifier.
            if track_id != -1:
                filter_key = track_id 
            else:
                filter_key = 'person'

            medicion_valida = (0 < Z_raw <= 4000)
            # Convert 2D pixel coordinates to real 3D camera metrics (in millimeters)
            if medicion_valida:
                X_raw = (cx - cx_rect) * Z_raw / fx_rect  
            else:  X_raw = 0.0
            
            if medicion_valida:
                Y_raw = (cy - cy_rect) * Z_raw / fx_rect  
            else: 
                Y_raw = 0.0

            # Query LiDAR data beforehand using the current input or previous tracked filter prediction state
            if medicion_valida:
                Z_busqueda = Z_raw  
            elif filter_key in kf_trackers:
                # Access estimated internal state vector (x[2] maps to tracked distance estimation)
                Z_busqueda = kf_trackers[filter_key]['kf'].x[2]
            else: 
                Z_busqueda = 1000.0

            if medicion_valida:
                X_busqueda = X_raw   
            elif filter_key in kf_trackers:
                # Access estimated internal state vector (x[0] maps to tracked lateral displacement)
                X_busqueda = kf_trackers[filter_key]['kf'].x[0]
            else:
                X_busqueda = 0.0
            distancia_lidar = obtener_distancia_lidar(scan_data, X_busqueda, Z_busqueda)
            
            ancho_pixeles = x2 - x1
            alto_pixeles = y2 - y1

            # Physical and Geometric Consistency Filters
            coherente = True
            # Check if LiDAR returned a valid distance range reading
            if distancia_lidar > 0:
                distancia_metros = distancia_lidar / 1000.0
                        
                # 2. Sanity check for empty/mismatched LiDAR readings (Floating false positive filter)
                if medicion_valida:
                    # Reject reading if stereo points a person at 1m but laser bounces beyond 2m
                    error_profundidad = abs(Z_raw - distancia_lidar)
                    if error_profundidad > 1000.0: 
                        coherente = False 
                    
            # Evaluate consistency properties before feeding Kalman tracking updates
            if not medicion_valida or not coherente:
                # If target is already known, fallback to the current filter prediction values
                if filter_key in kf_trackers:
                    kf_trackers[filter_key]['kf'].predict()
                    Z_filt = float(kf_trackers[filter_key]['kf'].x[2])

                    # Utilize current YOLO visual center indices alongside predicted depth metrics
                    # to keep the lateral estimation from drifting due to system inertia
                    X_filt = (cx - cx_rect) * Z_filt / fx_rect
                    Y_filt = (cy - cy_rect) * Z_filt / fx_rect

                    # Kill horizontal inertia elements inside Kalman matrix to freeze propagation error
                    kf_trackers[filter_key]['kf'].kf.statePost[3] = 0.0 
                    kf_trackers[filter_key]['kf'].kf.statePost[4] = 0.0
                
                else:
                    continue 
            
            # Object sensor checks succeeded
            else:
                # First time seeing target: Instantiate filter wrapper at current raw values
                if filter_key not in kf_trackers:
                    kf_trackers[filter_key] = {
                        'kf': OpenCVKalmanFilter(X_raw, Y_raw, Z_raw),
                        'nombre': 'person'
                    }
                    X_filt, Y_filt, Z_filt = X_raw, Y_raw, Z_raw
                
                # Active tracking target: Process standard predict-and-correct tracking frame loops
                else:
                    kf_trackers[filter_key]['kf'].predict()
                    estado_corregido = kf_trackers[filter_key]['kf'].update([X_raw, Y_raw, Z_raw])
                    
                    X_filt = float(estado_corregido[0])
                    Y_filt = float(estado_corregido[1])
                    Z_filt = float(estado_corregido[2])

            # Frame transformation to global coordinate workspace mapping
            cos_t = math.cos(pose_robot['theta'])
            sin_t = math.sin(pose_robot['theta'])

            distancia_real = distancia_lidar / 1000.0 if distancia_lidar > 0 else Z_filt / 1000.0
            X_adelante = distancia_real
            Y_izquierda = -X_filt / 1000.0

            X_global_calculado = pose_robot['x'] + (X_adelante * cos_t - Y_izquierda * sin_t)
            Y_global_calculado = pose_robot['y'] + (X_adelante * sin_t + Y_izquierda * cos_t)

            objetos_detectados.append({
                'nombre': 'person', 'bbox': (x1, y1, x2, y2), 'centro': (int(cx), int(cy)),
                'X': X_filt, 'Y': Y_filt, 'Z_cam': Z_filt, 'Z_lidar': distancia_lidar, 
                'X_global': X_global_calculado, 'Y_global': Y_global_calculado,
                'track_id': track_id, 'tipo': 'NORMAL', 'senalado': False
            })

            # Extract paired human body skeletal keypoints by array index configuration matching
            if r.keypoints is not None and i < len(r.keypoints.data):
                kpts = r.keypoints.data[i]
                if len(kpts) >= 11:
                    codo_l, codo_r = kpts[7], kpts[8]
                    muneca_l, muneca_r = kpts[9], kpts[10]
                    
                    if codo_l[2] > 0.5 and muneca_l[2] > 0.5:
                      brazos.append({'codo': (int(codo_l[0]), int(codo_l[1])), 'muneca': (int(muneca_l[0]), int(muneca_l[1]))})

    # GENERAL OBJECT PROCESSING
    h, w = rect_l.shape[:2]
    
    for r in resultados_yolo:
        if r.boxes is None: continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            nombre_clase = model.names[cls_id]
            confianza = float(box.conf[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            # --- Dynamic periphery confidence scaling (Fisheye lens distortion handling) ---
            # Enforce higher strictness thresholds on the outermost 20% peripheral field
            if cx < (w * 0.2) or cx > (w * 0.8):
                if confianza < 0.70:
                    continue
            
            # Skip 'person' class in general context iterations to avoid redundancy
            if nombre_clase == 'person': continue
            if nombre_clase not in TARGET_CLASSES: continue
            if nombre_clase not in ['sports ball'] and confianza < 0.60: continue
                
            track_id = int(box.id[0]) if box.id is not None else -1

            area_pixeles = (x2 - x1) * (y2 - y1)
            umbral_area = 300 if nombre_clase == 'sports ball' else 1500
            if area_pixeles < umbral_area: continue

            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            Z_raw = -1.0
            y_bottom = int(y1 + (y2 - y1) * 0.60)                                           
            roi_disparidad = disparity_map[y1:y_bottom, x1:x2]      
            valores_validos = roi_disparidad[roi_disparidad > 0]                            

            if len(valores_validos) > 0:
                disp_cercana = np.percentile(valores_validos, 90)                           
                umbral_fondo = disp_cercana * 0.65
                pixeles_primer_plano = valores_validos[valores_validos > umbral_fondo]      
                
                if len(pixeles_primer_plano) > 0:
                    disp_robusta = np.median(pixeles_primer_plano)                          
                    Z_raw = (fx_rect * baseline_rect) / disp_robusta                        

            filter_key = track_id if track_id != -1 else nombre_clase
            medicion_valida = (0 < Z_raw <= 4000)

            if medicion_valida:
                X_raw = (cx - cx_rect) * Z_raw / fx_rect  
            else:  X_raw = 0.0
            
            if medicion_valida:
                Y_raw = (cy - cy_rect) * Z_raw / fx_rect  
            else: 
                Y_raw = 0.0

            if medicion_valida:
                Z_busqueda = Z_raw  
            elif filter_key in kf_trackers:
                Z_busqueda = kf_trackers[filter_key]['kf'].x[2]
            else: 
                Z_busqueda = 1000.0

            if medicion_valida:
                X_busqueda = X_raw   
            elif filter_key in kf_trackers:
                X_busqueda = kf_trackers[filter_key]['kf'].x[0]
            else:
                X_busqueda = 0.0

            distancia_lidar = obtener_distancia_lidar(scan_data, X_busqueda, Z_busqueda)
            ancho_pixeles = x2 - x1
            alto_pixeles = y2 - y1

            coherente = True
            if distancia_lidar > 0:
                distancia_metros = distancia_lidar / 1000.0
                
                # Size-to-Distance scaling verification check
                if nombre_clase == 'chair':
                    # Reject if bounding box size profiles too small relative to detected laser depth
                    if alto_pixeles < (300 / distancia_metros): 
                        coherente = False 
                        
                if medicion_valida:
                    error_profundidad = abs(Z_raw - distancia_lidar)
                    if error_profundidad > 1000.0: 
                        coherente = False 
                    
            if not medicion_valida or not coherente:
                if filter_key in kf_trackers:
                    kf_trackers[filter_key]['kf'].predict()
                    Z_filt = float(kf_trackers[filter_key]['kf'].x[2])

                    X_filt = (cx - cx_rect) * Z_filt / fx_rect
                    Y_filt = (cy - cy_rect) * Z_filt / fx_rect

                    kf_trackers[filter_key]['kf'].kf.statePost[3] = 0.0 
                    kf_trackers[filter_key]['kf'].kf.statePost[4] = 0.0
                
                else:
                    continue 
            
            else:
                if filter_key not in kf_trackers:
                    kf_trackers[filter_key] = {
                        'kf': OpenCVKalmanFilter(X_raw, Y_raw, Z_raw),
                        'nombre': nombre_clase
                    }
                    X_filt, Y_filt, Z_filt = X_raw, Y_raw, Z_raw
                
                else:
                    kf_trackers[filter_key]['kf'].predict()
                    estado_corregido = kf_trackers[filter_key]['kf'].update([X_raw, Y_raw, Z_raw])
                    
                    X_filt = float(estado_corregido[0])
                    Y_filt = float(estado_corregido[1])
                    Z_filt = float(estado_corregido[2])
           
            cos_t = math.cos(pose_robot['theta'])
            sin_t = math.sin(pose_robot['theta'])

            distancia_real = distancia_lidar / 1000.0 if distancia_lidar > 0 else Z_filt / 1000.0
            X_adelante = distancia_real
            Y_izquierda = -X_filt / 1000.0

            X_global_calculado = pose_robot['x'] + (X_adelante * cos_t - Y_izquierda * sin_t)
            Y_global_calculado = pose_robot['y'] + (X_adelante * sin_t + Y_izquierda * cos_t)

            objetos_detectados.append({
                'nombre': nombre_clase, 'bbox': (x1, y1, x2, y2), 'centro': (int(cx), int(cy)),
                'X': X_filt, 'Y': Y_filt, 'Z_cam': Z_filt, 'Z_lidar': distancia_lidar, 
                'X_global': X_global_calculado, 'Y_global': Y_global_calculado,
                'track_id': track_id, 'tipo': 'NORMAL', 'senalado': False, 'confianza': confianza
            })
    
    # DOOR DETECTOR MODEL PIPELINE
    for r in resultados_yolo_p:
        if r.boxes is None: continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            nombre_clase = model_p.names[cls_id]
            confianza = float(box.conf[0])
            
            if nombre_clase not in TARGET_CLASSES: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if cx < (w * 0.2) or cx > (w * 0.8):
                if confianza < 0.70:
                    continue

            track_id = int(box.id[0]) if box.id is not None else -1

            area_pixeles = (x2 - x1) * (y2 - y1)
            if area_pixeles < 1500: continue 

            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            Z_raw = -1.0
            y_bottom = int(y1 + (y2 - y1) * 0.60)                                           
            roi_disparidad = disparity_map[y1:y_bottom, x1:x2]      
            valores_validos = roi_disparidad[roi_disparidad > 0]                            

            if len(valores_validos) > 0:
                disp_cercana = np.percentile(valores_validos, 90)                           
                umbral_fondo = disp_cercana * 0.65
                pixeles_primer_plano = valores_validos[valores_validos > umbral_fondo]      
                if len(pixeles_primer_plano) > 0:
                    disp_robusta = np.median(pixeles_primer_plano)                          
                    Z_raw = (fx_rect * baseline_rect) / disp_robusta                        

            if Z_raw <= 0 or Z_raw > 4000: continue
            
            X_raw = (cx - cx_rect) * Z_raw / fx_rect    
            Y_raw = (cy - cy_rect) * Z_raw / fx_rect    
            
            filter_key = track_id if track_id != -1 else nombre_clase
            medicion_valida = (0 < Z_raw <= 4000)

            if medicion_valida:
                X_raw = (cx - cx_rect) * Z_raw / fx_rect  
            else:  X_raw = 0.0
            
            if medicion_valida:
                Y_raw = (cy - cy_rect) * Z_raw / fx_rect  
            else: 
                Y_raw = 0.0

            if medicion_valida:
                Z_busqueda = Z_raw  
            elif filter_key in kf_trackers:
                Z_busqueda = kf_trackers[filter_key]['kf'].x[2]
            else: 
                Z_busqueda = 1000.0

            if medicion_valida:
                X_busqueda = X_raw   
            elif filter_key in kf_trackers:
                X_busqueda = kf_trackers[filter_key]['kf'].x[0]
            else:
                X_busqueda = 0.0
            distancia_lidar = obtener_distancia_lidar(scan_data, X_busqueda, Z_busqueda)
            
            ancho_pixeles = x2 - x1
            alto_pixeles = y2 - y1

            coherente = True
            if distancia_lidar > 0:
                distancia_metros = distancia_lidar / 1000.0
                
                if nombre_clase == 'chair':
                    if alto_pixeles < (300 / distancia_metros): 
                        coherente = False 
                        
                if medicion_valida:
                    error_profundidad = abs(Z_raw - distancia_lidar)
                    if error_profundidad > 1000.0: 
                        coherente = False 
                    
            if not medicion_valida or not coherente:
                if filter_key in kf_trackers:
                    kf_trackers[filter_key]['kf'].predict()
                    Z_filt = float(kf_trackers[filter_key]['kf'].x[2])

                    X_filt = (cx - cx_rect) * Z_filt / fx_rect
                    Y_filt = (cy - cy_rect) * Z_filt / fx_rect

                    kf_trackers[filter_key]['kf'].kf.statePost[3] = 0.0 
                    kf_trackers[filter_key]['kf'].kf.statePost[4] = 0.0
                
                else:
                    continue 
            
            else:
                if filter_key not in kf_trackers:
                    kf_trackers[filter_key] = {
                        'kf': OpenCVKalmanFilter(X_raw, Y_raw, Z_raw),
                        'nombre': nombre_clase
                    }
                    X_filt, Y_filt, Z_filt = X_raw, Y_raw, Z_raw
                
                else:
                    kf_trackers[filter_key]['kf'].predict()
                    estado_corregido = kf_trackers[filter_key]['kf'].update([X_raw, Y_raw, Z_raw])
                    
                    X_filt = float(estado_corregido[0])
                    Y_filt = float(estado_corregido[1])
                    Z_filt = float(estado_corregido[2])

            cos_t = math.cos(pose_robot['theta'])
            sin_t = math.sin(pose_robot['theta'])

            distancia_real = distancia_lidar / 1000.0 if distancia_lidar > 0 else Z_filt / 1000.0
            X_adelante = distancia_real
            Y_izquierda = -X_filt / 1000.0

            X_global_calculado = pose_robot['x'] + (X_adelante * cos_t - Y_izquierda * sin_t)
            Y_global_calculado = pose_robot['y'] + (X_adelante * sin_t + Y_izquierda * cos_t)

            objetos_detectados.append({
                'nombre': nombre_clase, 'bbox': (x1, y1, x2, y2), 'centro': (int(cx), int(cy)),
                'X': X_filt, 'Y': Y_filt, 'Z_cam': Z_filt, 'Z_lidar': distancia_lidar,
                'X_global': X_global_calculado, 'Y_global': Y_global_calculado,
                'track_id': track_id, 'tipo': 'NORMAL', 'senalado': False, 'confianza': confianza
            })

    # Clear outdated/stale trackers from dict memory
    tracks_activos = set()
    for r in resultados_yolo:
        if r.boxes is not None:
            for box in r.boxes:
                if box.id is not None: tracks_activos.add(int(box.id[0]))
    for r in resultados_yolo_p:
        if r.boxes is not None:
            for box in r.boxes:
                if box.id is not None: tracks_activos.add(int(box.id[0]))
    for r in resultados_pose:
            if r.boxes is not None:
                for box in r.boxes:
                    if box.id is not None: tracks_activos.add(int(box.id[0]))

    keys_a_borrar = [k for k in kf_trackers if isinstance(k, int) and k not in tracks_activos]
    for k in keys_a_borrar: del kf_trackers[k]

    # Human positioning data packing
    pos_persona = Point()
    for obj in objetos_detectados:
        if obj['nombre'] == 'person':
            dist_frontal = obj['Z_lidar'] / 1000.0 if obj['Z_lidar'] > 0 else obj['Z_cam'] / 1000.0
            desvio_lateral = -obj['X'] / 1000.0
            pos_persona.x = dist_frontal
            pos_persona.y = desvio_lateral
            pos_persona.z = 0.0

    # === GESTURE DETECTION AND TARGET VERIFICATION ===
    alguno_senalado = False
    
    for brazo in brazos:
        objetos_cruzados = []
        for obj in objetos_detectados:
            if obj['nombre'] in ['door', 'chair', 'sports ball', 'backpack', 'dining table']:
                distancia, hacia_adelante, centro = is_pointing_at(brazo['codo'], brazo['muneca'], obj['bbox'])
                
                ancho_caja = obj['bbox'][2] - obj['bbox'][0]
                alto_caja = obj['bbox'][3] - obj['bbox'][1]
                umbral_tolerancia = max(ancho_caja, alto_caja) / 2.0 
                
                if hacia_adelante and distancia < umbral_tolerancia:
                    objetos_cruzados.append({'obj': obj, 'dist': distancia, 'centro': centro})

        if objetos_cruzados:
            objetos_cruzados.sort(key=lambda x: x['dist']) # Closest ray intersection center wins
            
            ganador = objetos_cruzados[0]
            ganador['obj']['senalado'] = True
            ganador['obj']['tipo'] = 'TARGET'
            brazo['target_centro'] = ganador['centro'] # Snap pointing visual boundary to target center
            
            alguno_senalado = True
            
            # Label remaining aligned items as False Positives
            for item in objetos_cruzados[1:]:
                item['obj']['senalado'] = True
                if item['obj']['tipo'] != 'TARGET':
                    item['obj']['tipo'] = 'FP'

    # Filter confirmation hysteresis loop logic
    if alguno_senalado:
        contador_gesto += 1
        if contador_gesto >= 5:
            gesto_detectado = True
            
            # Fetch target location details to resolve navigation coordinates
            for obj in objetos_detectados:
                if obj['tipo'] == 'TARGET':
                    x_centro_obj = obj['X_global']
                    y_centro_obj = obj['Y_global']

                    objeto_en_memoria = None
                    distancia_minima = float('inf')

                    for obj_mem in memoria_semantica:
                        if obj_mem['nombre'] == obj['nombre']:
                            # Query matching spatial anchors in short distance radius range (under 2.0 meters)
                            dist = math.sqrt((x_centro_obj - obj_mem['x'])**2 + (y_centro_obj - obj_mem['y'])**2)
                            if dist < 2.0 and dist < distancia_minima:
                                objeto_en_memoria = obj_mem
                                distancia_minima = dist
                    
                    if objeto_en_memoria:
                        rospy.loginfo("Goal found in semantic memory! Harnessing Kalman Filter map positions.")
                        x_centro_final = objeto_en_memoria['x']
                        y_centro_final = objeto_en_memoria['y']

                    else:
                        rospy.loginfo("New object anchor detected. Deploying direct visual input estimation.")
                        x_centro_final = x_centro_obj
                        y_centro_final = y_centro_obj

                    # Evaluate line angular orientation heading target from current robot framework positions
                    angulo_hacia_objeto = math.atan2(y_centro_final - pose_robot['y'], x_centro_final - pose_robot['x'])
                    
                    # Apply offset clearance pad cushion along heading vector lines
                    MARGEN = 0.5 
                    x_meta = x_centro_final - MARGEN * math.cos(angulo_hacia_objeto)
                    y_meta = y_centro_final - MARGEN * math.sin(angulo_hacia_objeto)

                    meta = PoseStamped()
                    meta.header.frame_id = "odom"
                    meta.header.stamp = rospy.Time(0)
                    meta.pose.position.x = x_meta
                    meta.pose.position.y = y_meta
                    meta.pose.orientation.w = 1.0 
                    
                    pub_meta.publish(meta)
                    rospy.loginfo("Gesture validated! Navigation goal dispatched to BT: X={:.2f}, Y={:.2f}".format(x_meta, y_meta))
                    break # Track single dispatch focus
                    
            contador_gesto = 0 
    else:
        # Cool down hysteresis gracefully instead of clearing to zero instantly
        contador_gesto = max(0, contador_gesto - 1)
                    
    canvas = dibujar_resultado(rect_l, objetos_detectados, brazos)
    cv2.imshow("Detection and Range", canvas)
    cv2.waitKey(1)

    pub_persona.publish(persona_vista)
    pub_gesto.publish(gesto_detectado)

    if persona_vista:
        pub_pos_persona.publish(pos_persona)
    
    publicar_mapa_semantico(objetos_detectados)

# === VISUALIZATION DRAWING FUNCTIONS ===
def dibujar_resultado(img, objetos, brazos):
    vis = img.copy()
    
    for brazo in brazos:
        p1, p2 = brazo['codo'], brazo['muneca']

        if 'target_centro' in brazo and brazo['target_centro'] is not None:
            p_end = brazo['target_centro']
            cv2.line(vis, p1, p_end, (0, 255, 255), 3) 
            cv2.circle(vis, p_end, 8, (0, 0, 255), -1) 
        else:
            diff = np.array(p2) - np.array(p1)
            p_end = tuple((np.array(p1) + diff * 15).astype(int)) # Extend far ray if no intersection triggers
            cv2.line(vis, p1, p_end, (255, 0, 255), 2) 
            
        cv2.circle(vis, p2, 6, (0, 0, 255), -1)    
        cv2.circle(vis, p1, 6, (255, 0, 0), -1)

    if not objetos:
        cv2.putText(vis, "Searching for objects...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    
    for obj in objetos:
        nombre, (x1, y1, x2, y2), (cx, cy) = obj['nombre'], obj['bbox'], obj['centro']
        X, Y, Z_cam, Z_lidar = obj['X'], obj['Y'], obj['Z_cam'], obj['Z_lidar']
        tr_id = obj['track_id']
        
        # Bounding box attribute styling according to classification type tags (TARGET / FP / NORMAL)
        if obj['senalado']:
            if obj['tipo'] == 'TARGET':
                color_caja = (0, 0, 255) # Red
                grosor = 4
                str_tipo = " (TARGET)"
            else:
                color_caja = (255, 0, 0) # Blue
                grosor = 2
                str_tipo = " (FP)"
        else:
            color_caja = (0, 255, 0) # Green
            grosor = 2
            str_tipo = ""

        etiqueta = "{} [{}]".format(nombre.upper(), tr_id) if tr_id != -1 else nombre.upper()
        etiqueta += str_tipo

        lidar_text = "Z(Lidar)= %.1f cm" % (Z_lidar / 10.0) if Z_lidar > 0 else "Z(Lidar)= NOT SEEN"
        lines = [
            "Obj: %s" % nombre.upper(), 
            "X = %5.1f cm" % (X / 10.0), 
            "Z(Cam) = %5.1f cm" % (Z_cam / 10.0), 
            lidar_text
        ]
        
        y_offset = 30
        cv2.rectangle(vis, (5, y_offset - 20), (250, y_offset + len(lines) * 25), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            color_texto = (0, 255, 0) if (i != 3 or Z_lidar > 0) else (0, 0, 255)
            cv2.putText(vis, line, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color_caja, grosor)
        cv2.putText(vis, etiqueta, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_caja, 2)
            
    return vis

def publicar_mapa_semantico(objetos_actuales):
    global memoria_semantica, pose_robot, siguiente_id
    
    for obj in objetos_actuales:
        nombre = obj['nombre']

        if nombre == 'person': 
            continue

        Z_lidar = obj['Z_lidar']
        Z_cam = obj['Z_cam']
        
        # FALLBACK STRATEGY SELECTOR LOGIC: Determine focus sensor feed
        usar_lidar = (0 < Z_lidar <= 3000)
        usar_cam = (0 < Z_cam <= 4000) # Camera lens displays slightly higher field distance limits

        if usar_lidar:
            # High confidence factor rating: Physical laser return feed
            peso_lectura = 1.0
        elif usar_cam:
            # Lower confidence rating: Stereo SGBM visual tracking output noise profiles
            peso_lectura = 0.3 
        else:
            continue # Sensors failed range checks
        
        X_global = obj['X_global']
        Y_global = obj['Y_global']
        
        if not es_punto_valido(X_global, Y_global): continue 
        
        else: distancia_tolerancia = 1.5 
        
        objeto_asociado = None
        distancia_minima = float('inf')
        
        for obj_memoria in memoria_semantica:
            if obj_memoria['nombre'] == nombre:
                dist = math.sqrt((X_global - obj_memoria['x'])**2 + (Y_global - obj_memoria['y'])**2)

                if dist < distancia_tolerancia and dist < distancia_minima:
                    distancia_minima = dist
                    objeto_asociado = obj_memoria
                    
        if objeto_asociado is not None:
            # Inject update variables through MapKalmanFilter execution
            x_filtrado, y_filtrado = objeto_asociado['kf'].update(X_global, Y_global, peso_lectura)
            
            objeto_asociado['x'] = x_filtrado
            objeto_asociado['y'] = y_filtrado

            vistos = objeto_asociado['visto_veces']
            objeto_asociado['confianza'] = (objeto_asociado['confianza'] * vistos + obj['confianza']) / (vistos + 1)
            objeto_asociado['visto_veces'] = min(15, vistos + 1)
        else:
            # Initialize a new map tracker filter tracking anchor configuration
            nuevo_kf = MapKalmanFilter(X_global, Y_global)
            memoria_semantica.append({
                'id': siguiente_id, 
                'nombre': nombre, 
                'x': X_global, 
                'y': Y_global, 
                'z': 0.0, 
                'visto_veces': 1,
                'kf': nuevo_kf,  # Keep the filter pipeline instance inside object storage
                'confianza': obj['confianza']
            })
            siguiente_id += 1

    # Database filtering logic to group redundant entries or clean phantom records
    memoria_limpia = []
    for obj in memoria_semantica:
        fusionado = False
        for valid_obj in memoria_limpia:
            if obj['nombre'] == valid_obj['nombre']:
                dist = math.sqrt((obj['x'] - valid_obj['x'])**2 + (obj['y'] - valid_obj['y'])**2)
                tolerancia_basura = 2.0 if obj['nombre'] == 'person' else 0.75
                if dist < tolerancia_basura: 
                    peso_total = valid_obj['visto_veces'] + obj['visto_veces']
                    valid_obj['x'] = (valid_obj['x'] * valid_obj['visto_veces'] + obj['x'] * obj['visto_veces']) / peso_total
                    valid_obj['y'] = (valid_obj['y'] * valid_obj['visto_veces'] + obj['y'] * obj['visto_veces']) / peso_total

                    valid_obj['confianza'] = (valid_obj['confianza'] * valid_obj['visto_veces'] + obj['confianza'] * obj['visto_veces']) / peso_total
                    
                    valid_obj['visto_veces'] = min(15, peso_total)
                    fusionado = True
                    break
        if not fusionado: memoria_limpia.append(obj)
    memoria_semantica = memoria_limpia

    marker_array = MarkerArray()

    # Dispatch a 'DELETEALL' command at array head index to flush historic markers out of RViz context instances
    borrado_general = Marker()
    borrado_general.action = 3 # 3 maps to Marker.DELETEALL enumerations
    marker_array.markers.append(borrado_general)

    for obj_mem in memoria_semantica:
        vistas_necesarias = 5 if obj_mem['nombre'] == 'chair' else 3
        if obj_mem['visto_veces'] < vistas_necesarias: continue

        id_marcador = obj_mem['id']
        marker = Marker()
        marker.header.frame_id = "odom" 
        marker.header.stamp = rospy.Time(0)
        marker.ns = "semantic_cubes"
        marker.id = id_marcador
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = obj_mem['x']
        marker.pose.position.y = obj_mem['y']
        marker.pose.position.z = obj_mem['z']
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 0.8 
        if obj_mem['nombre'] == 'chair':
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif obj_mem['nombre'] == 'dining table':
            marker.color.r, marker.color.g, marker.color.b = 0.5, 0.5, 0.0
        elif obj_mem['nombre'] == 'door':
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        elif obj_mem['nombre'] == 'tv':
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        elif obj_mem['nombre'] == 'backpack':
            marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
        elif obj_mem['nombre'] == 'sports ball':
            marker.color.r, marker.color.g, marker.color.b = 0.0, 0.5, 0.5
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 1.0
        marker_array.markers.append(marker)
        
        texto = Marker()
        texto.header.frame_id = "odom"
        texto.header.stamp = rospy.Time(0)
        texto.ns = "semantic_texts"
        texto.id = id_marcador 
        texto.type = Marker.TEXT_VIEW_FACING
        texto.action = Marker.ADD
        texto.pose.position.x = obj_mem['x']
        texto.pose.position.y = obj_mem['y']
        texto.pose.position.z = obj_mem['z'] + 0.4
        texto.pose.orientation.w = 1.0
        texto.text = "{} #{}".format(obj_mem['nombre'].upper(), id_marcador)
        texto.scale.z = 0.2
        texto.color.a, texto.color.r, texto.color.g, texto.color.b = 1.0, 1.0, 1.0, 1.0
        marker_array.markers.append(texto)

    if marker_array.markers: marker_pub.publish(marker_array)

# Validation exporting utility logic
# def exportar_mapa_final_csv(memoria):
#     objetos_consolidados = [obj for obj in memoria if obj['visto_veces'] >= 3]
#     if not objetos_consolidados:
#         rospy.loginfo("[EVALUATION] No consolidated anchors found to export.")
#         return
#
#     nombre_archivo = 'mapa_semantico_final(2).csv'
#     with open(nombre_archivo, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['ID', 'Object', 'X_global', 'Y_global', 'Observation_Count', 'Mean_Confidence'])
#         for obj in objetos_consolidados:
#             writer.writerow([obj['id'], obj['nombre'], obj['x'], obj['y'], obj['visto_veces'], round(obj['confianza'], 3)])
#     rospy.loginfo(f"[EVALUATION] Semantic map successfully logged to {nombre_archivo}")

def main():
    global latest_left, latest_right, latest_scan, marker_pub, pub_persona, pub_gesto, pub_pos_persona, pub_meta
    rospy.init_node('detectar_yolo_estereo', anonymous=True)

    # ROS Subscribers Configuration
    rospy.Subscriber("camera1/left/image_raw/compressed", CompressedImage, callback_left, queue_size=1)
    rospy.Subscriber("camera1/right/image_raw/compressed", CompressedImage, callback_right, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, callback_lidar, queue_size=1)
    rospy.Subscriber("/odom", Odometry, callback_odom, queue_size=1)
    rospy.Subscriber("/map", OccupancyGrid, callback_map, queue_size=1)
    
    # ROS Publishers Configuration
    marker_pub = rospy.Publisher('/mapa_semantico', MarkerArray, queue_size=1)
    pub_persona = rospy.Publisher('/estado_persona', Bool, queue_size=1)
    pub_gesto = rospy.Publisher('/gesto_parada', Bool, queue_size=1)
    pub_pos_persona = rospy.Publisher('/posicion_persona', Point, queue_size=1)
    pub_meta = rospy.Publisher('/meta_senalada', PoseStamped, queue_size=1)

    cv2.namedWindow("Detection and Range", cv2.WINDOW_NORMAL)
    rospy.loginfo("Launching perception tracking pipeline node. Awaiting frame streams...")

    rate = rospy.Rate(30) 
    try:
        while not rospy.is_shutdown():
            if latest_left is not None and latest_right is not None:
                procesar_imagenes(latest_left.copy(), latest_right.copy(), latest_scan)
            rate.sleep()
    finally:
        # Triggered via Ctrl+C exit paths
        #exportar_mapa_final_csv(memoria_semantica)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()