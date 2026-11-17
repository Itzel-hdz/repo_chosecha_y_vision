"""Captura fotos con la RealSense D435 para armar el dataset.

's' = guardar una foto | 'a' = auto-captura cada 1 s (on/off) | 'q' = salir
Los nombres llevan fecha/hora: reiniciar el script NO sobrescribe fotos anteriores.
"""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

CARPETA = "dataset_fotos"
os.makedirs(CARPETA, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

auto = False
ultimo_auto = 0.0
total = 0


def guardar(img):
    global total
    nombre = f"{CARPETA}/fresa_{time.strftime('%Y%m%d_%H%M%S')}_{total:04d}.png"
    cv2.imwrite(nombre, img)
    total += 1
    print(f"guardada ({total}): {nombre}")


try:
    while True:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())

        vista = img.copy()
        cv2.putText(vista, f"fotos: {total}  auto: {'ON' if auto else 'off'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Captura para dataset", vista)

        if auto and time.time() - ultimo_auto >= 3.0:
            guardar(img)
            ultimo_auto = time.time()

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("s"), ord("S"), 32):
            guardar(img)
        elif k == ord("a"):
            auto = not auto
        elif k in (ord("q"), ord("Q")):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
