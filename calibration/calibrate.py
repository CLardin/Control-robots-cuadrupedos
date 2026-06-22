#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibracion estereo para la camara frontal del Unitree Go1.
Compatible con Python 2.7 y OpenCV 3.x / 4.x.

Uso:
    python calibrate.py -w 9 -H 6 --square_size 23 --calib_dir ./imag_calib --debug

Notas:
  - -w y -H son esquinas INTERIORES (cuadrados - 1).
    Tablero 9x6 cuadrados -> -w 8 -H 5
  - square_size en milimetros reales del cuadrado impreso.
  - Las fotos deben ser pares: left_00.jpg / right_00.jpg
"""
"""
Fotos tablero
      ↓
findChessboardCorners → puntos 2D detectados
      ↓
Zhang: homografías → K inicial → Levenberg-Marquardt → K, D finales
      ↓
stereoCalibrate → R, T entre cámaras (baseline)
      ↓
stereoRectify → R1, R2, P1, P2 (alineación epipolar)
      ↓
initUndistortRectifyMap → mapas de corrección
      ↓
remap → imagen rectificada lista para disparidad
      ↓
Z = fx * baseline / disparidad
"""

from __future__ import print_function
import cv2
import numpy as np
import os
import glob
import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Calibracion estereo Go1 - Python 2.7")
    parser.add_argument("--debug", action="store_true",
                        help="Guardar imagenes con esquinas detectadas en ./output_calib_debug/")
    parser.add_argument("-w", "--width", type=int, default=8,
                        help="Esquinas interiores eje X (defecto: 8)")
    parser.add_argument("-H", "--height", type=int, default=5,
                        help="Esquinas interiores eje Y (defecto: 5)")
    parser.add_argument("--square_size", type=float, default=25.0,
                        help="Tamano del cuadrado en mm (defecto: 25.0)")
    parser.add_argument("--calib_dir", type=str, default="./imag_calib",
                        help="Carpeta con las imagenes (defecto: ./imag_calib)")
    parser.add_argument("--output", type=str, default="./calibracion_go1.json",
                        help="Archivo JSON de salida (defecto: ./calibracion_go1.json)")
    return parser.parse_args()


def find_corners(img_path, pattern_size, pattern_points, debug_dir=None):
    img = cv2.imread(img_path)
    if img is None:
        print("  [ERROR] No se pudo cargar: %s" % img_path)
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

    if not found:
        print("  [FALLO] %s" % os.path.basename(img_path))
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
    corners  = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    if debug_dir:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(debug_dir, name + "_corners.jpg"), vis)

    print("  [OK] %s" % os.path.basename(img_path))
    # fisheye necesita shape (N, 1, 2)
    return corners.reshape(-1, 1, 2).astype(np.float64), pattern_points.astype(np.float64)

def calibrate_stereo(left_paths, right_paths, pattern_size, square_size, debug_dir=None):
    # 1. Definir los puntos del patrón primero
    pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float64)
    pattern_points[:, :2] = np.mgrid[0:pattern_size[0],
                                      0:pattern_size[1]].T.reshape(-1, 2)
    pattern_points *= square_size
    pattern_points = pattern_points.reshape(-1, 1, 3)

    # 2. Inicializar las listas VACÍAS antes del bucle
    obj_pts_s  = []
    ipts_left  = []
    ipts_right = []

    print("\nBuscando tablero en pares...")
    for lp, rp in zip(sorted(left_paths), sorted(right_paths)):
        rl = find_corners(lp, pattern_size, pattern_points, debug_dir)
        rr = find_corners(rp, pattern_size, pattern_points, debug_dir)
        if rl is not None and rr is not None:
            # Forzamos float64 desde la captura para evitar el error 'arithm_op'
            ipts_left.append(rl[0].astype(np.float64))
            ipts_right.append(rr[0].astype(np.float64))
            obj_pts_s.append(pattern_points.astype(np.float64))

    # 3. Verificar si tenemos suficientes pares
    if len(obj_pts_s) < 5:
        raise RuntimeError("Solo %d pares validos." % len(obj_pts_s))

    # 4. AHORA las variables ya existen y puedes operar con ellas
    img = cv2.imread(sorted(left_paths)[0])
    img_size = (img.shape[1], img.shape[0])

    # Inicializar matrices de cámara con float64
    K_l = np.zeros((3, 3), dtype=np.float64)
    D_l = np.zeros((4, 1), dtype=np.float64)
    K_r = np.zeros((3, 3), dtype=np.float64)
    D_r = np.zeros((4, 1), dtype=np.float64)

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
             cv2.fisheye.CALIB_CHECK_COND +
             cv2.fisheye.CALIB_FIX_SKEW)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    # Calibrar cámara izquierda individualmente
    print("  Calibrando izquierda...")
    rms_l, K_l, D_l, _, _ = cv2.fisheye.calibrate(
        obj_pts_s, ipts_left, img_size,
        K_l, D_l,
        flags=flags,
        criteria=criteria
    )
    print("  RMS izquierda: %.4f  fx=%.2f cx=%.2f cy=%.2f" % (
          rms_l, K_l[0,0], K_l[0,2], K_l[1,2]))

    # Calibrar cámara derecha individualmente
    K_r = np.zeros((3, 3))
    D_r = np.zeros((4, 1))
    print("  Calibrando derecha...")
    rms_r, K_r, D_r, _, _ = cv2.fisheye.calibrate(
        obj_pts_s, ipts_right, img_size,
        K_r, D_r,
        flags=flags,
        criteria=criteria
    )
    print("  RMS derecha  : %.4f  fx=%.2f cx=%.2f cy=%.2f" % (
          rms_r, K_r[0,0], K_r[0,2], K_r[1,2]))

    # Calibración estéreo fisheye
    print("  Calibrando estereo fisheye...")
    R = np.zeros((3, 3), dtype=np.float64)
    T = np.zeros((3, 1), dtype=np.float64)

    rms_s, K_l, D_l, K_r, D_r, R, T = cv2.fisheye.stereoCalibrate(
        obj_pts_s, ipts_left, ipts_right,
        K_l, D_l, K_r, D_r,
        img_size,
        R, T,
        flags=flags,
        criteria=criteria
    )

    baseline = np.linalg.norm(T)
    print("  RMS estereo  : %.4f px" % rms_s)
    print("  Baseline     : %.2f mm" % baseline)

    return {
        "img_size"    : img_size,
        "K_left"      : K_l,
        "D_left"      : D_l,
        "K_right"     : K_r,
        "D_right"     : D_r,
        "R"           : R,
        "T"           : T,
        "E"           : np.zeros((3,3)),  # fisheye no devuelve E y F
        "F"           : np.zeros((3,3)),
        "rms_left"    : rms_l,
        "rms_right"   : rms_r,
        "rms_stereo"  : rms_s,
        "baseline_mm" : baseline,
    }

def compute_rectification(res):
    # Aseguramos que todas las entradas sean float64 (indispensable en fisheye)
    K1 = np.array(res["K_left"],  dtype=np.float64)
    D1 = np.array(res["D_left"],  dtype=np.float64)
    K2 = np.array(res["K_right"], dtype=np.float64)
    D2 = np.array(res["D_right"], dtype=np.float64)
    R  = np.array(res["R"],       dtype=np.float64)
    T  = np.array(res["T"],       dtype=np.float64)
    img_size = res["img_size"]

    # En OpenCV fisheye, a veces es mejor NO pre-inicializar 
    # y dejar que la funcion devuelva las matrices, 
    # O inicializarlas con la forma exacta.
    
    # Intentaremos la llamada pasando None para que las cree internamente 
    # y evitar el conflicto de tipos de escalares.
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1,
        K2, D2,
        img_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=img_size,
        balance=0.0,
        fov_scale=1.0
    )

    print("\n  Rectificacion calculada:")
    print("  P1 fx=%.2f cx=%.2f cy=%.2f" % (P1[0,0], P1[0,2], P1[1,2]))
    print("  P2 fx=%.2f cx=%.2f cy=%.2f" % (P2[0,0], P2[0,2], P2[1,2]))

    return R1, R2, P1, P2, Q


def save_results(res, output_path):
    """
    Guarda JSON legible y .npz para numpy.
    """
    data = {
        "descripcion"  : "Calibracion estereo camara frontal Unitree Go1",
        "img_size_wh"  : list(res["img_size"]),
        "rms_left_px"  : round(float(res["rms_left"]),   4),
        "rms_right_px" : round(float(res["rms_right"]),  4),
        "rms_stereo_px": round(float(res["rms_stereo"]), 4),
        "baseline_mm"  : round(float(res["baseline_mm"]),4),
        "camara_izquierda": {
            "camera_matrix": res["K_left"].tolist(),
            "dist_coeffs"  : res["D_left"].tolist(),
            "fx": round(float(res["K_left"][0,0]), 4),
            "fy": round(float(res["K_left"][1,1]), 4),
            "cx": round(float(res["K_left"][0,2]), 4),
            "cy": round(float(res["K_left"][1,2]), 4),
        },
        "camara_derecha": {
            "camera_matrix": res["K_right"].tolist(),
            "dist_coeffs"  : res["D_right"].tolist(),
            "fx": round(float(res["K_right"][0,0]), 4),
            "fy": round(float(res["K_right"][1,1]), 4),
            "cx": round(float(res["K_right"][0,2]), 4),
            "cy": round(float(res["K_right"][1,2]), 4),
        },
        "relacion_estereo": {
            "R": res["R"].tolist(),
            "T": res["T"].tolist(),
        }
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    npz_path = output_path.replace(".json", ".npz")
    np.savez(npz_path,
             K_left  = res["K_left"],
             D_left  = res["D_left"],
             K_right = res["K_right"],
             D_right = res["D_right"],
             R       = res["R"],
             T       = res["T"],
             E       = res["E"],
             F       = res["F"],
             img_size= np.array(res["img_size"]),
             R1      = res["R1"],
             R2      = res["R2"],
             P1      = res["P1"],
             P2      = res["P2"],
             Q       = res["Q"])

    print("\n  JSON guardado : %s" % output_path)
    print("  NPZ guardado  : %s" % npz_path)


def main():
    args = parse_args()

    debug_dir = None
    if args.debug:
        debug_dir = "./output_calib_debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        print("Debug en: %s" % debug_dir)

    left_paths  = sorted(glob.glob(os.path.join(args.calib_dir, "left_*.jpg")))
    right_paths = sorted(glob.glob(os.path.join(args.calib_dir, "right_*.jpg")))

    if not left_paths or not right_paths:
        print("[ERROR] No se encontraron imagenes en '%s'" % args.calib_dir)
        print("  Busca: left_00.jpg, right_00.jpg, left_01.jpg, right_01.jpg ...")
        sys.exit(1)

    n = min(len(left_paths), len(right_paths))
    if len(left_paths) != len(right_paths):
        print("[AVISO] Numero de imagenes distinto. Usando %d pares." % n)
    left_paths  = left_paths[:n]
    right_paths = right_paths[:n]

    print("Encontrados %d pares en '%s'" % (n, args.calib_dir))
    print("Tablero: %dx%d esquinas interiores, cuadrado=%.1fmm" % (
          args.width, args.height, args.square_size))

    pattern_size = (args.width, args.height)

    try:
        result = calibrate_stereo(
            left_paths, right_paths,
            pattern_size, args.square_size,
            debug_dir
        )

        R1, R2, P1, P2, Q = compute_rectification(result)
        result["R1"] = R1
        result["R2"] = R2
        result["P1"] = P1
        result["P2"] = P2
        result["Q"]  = Q

        save_results(result, args.output)

        # Resumen final
        print("\n" + "="*55)
        print("RESUMEN")
        print("="*55)
        print("  Pares usados  : %d" % n)
        print("  RMS izquierda : %.4f px  (bueno si < 1.0)" % result["rms_left"])
        print("  RMS derecha   : %.4f px  (bueno si < 1.0)" % result["rms_right"])
        print("  RMS estereo   : %.4f px  (bueno si < 1.0)" % result["rms_stereo"])
        print("  Baseline      : %.2f mm" % result["baseline_mm"])

        rms = result["rms_stereo"]
        if   rms < 0.5: print("\n  Calibracion EXCELENTE")
        elif rms < 1.0: print("\n  Calibracion BUENA")
        elif rms < 2.0: print("\n  Calibracion ACEPTABLE - captura mas fotos")
        else:           print("\n  Calibracion POBRE - repite la captura")

    except RuntimeError as e:
        print("\n[ERROR] %s" % str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()