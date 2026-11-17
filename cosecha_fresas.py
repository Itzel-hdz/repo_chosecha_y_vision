import argparse, time
import cv2
import numpy as np
import rtde_receive
from ultralytics import YOLO
from scipy.spatial.transform import Rotation

RECOGER = {"grande_roja", "chica_roja"}
ROBOT_IP = "192.168.0.10"

# Calibracion mano-ojo: cámara -> TCP (marco NATIVO de la D435: X derecha, Y abajo, Z adelante)
# Unidades: metros. Sale de calibrar_camara.py (25 poses).
T_TCP_CAM = np.array([
    [-0.999500,  0.028256,  0.014210,  0.020598],
    [-0.029022, -0.997956, -0.056933,  0.080521],
    [ 0.012572, -0.057317,  0.998277, -0.113376],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
])

def pose_a_matriz(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    return T

class Robot:
    """Lee la pose del TCP y se reconecta si el robot cortó RTDE (p. ej. por freedrive)."""
    def __init__(self, ip):
        self.ip = ip
        self.r = None
        self._conectar()

    def _conectar(self):
        try:
            self.r = rtde_receive.RTDEReceiveInterface(self.ip)
        except Exception:
            self.r = None

    def pose(self):
        for _ in range(5):
            try:
                if self.r is None:
                    self._conectar()
                if self.r is not None:
                    return self.r.getActualTCPPose()
            except Exception:
                self.r = None
            time.sleep(0.3)
        return None

def dibujar(img, caja, nombre, conf, xyz=None):
    x1, y1, x2, y2 = map(int, caja)
    rec = nombre in RECOGER
    color = (0, 200, 0) if rec else (0, 0, 255)
    txt = f"{nombre} {conf:.2f} -> {'RECOGER' if rec else 'NO'}"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if xyz is not None:
        t = f"BASE X{xyz[0]:.0f} Y{xyz[1]:.0f} Z{xyz[2]:.0f} mm"
        cv2.putText(img, t, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def abrir(a):
    if a.webcam:
        return None, None, None, None, None, cv2.VideoCapture(0)
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    return rs, pipe, align, intr, scale, None

def bucle(model, a):
    rs, pipe, align, intr, scale, cap = abrir(a)
    robot = Robot(ROBOT_IP)
    ultimo = 0
    try:
        while True:
            depth = None
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                fr = align.process(pipe.wait_for_frames())
                c, d = fr.get_color_frame(), fr.get_depth_frame()
                if not c or not d:
                    continue
                frame = np.asanyarray(c.get_data())
                depth = np.asanyarray(d.get_data())

            # Pose del robot en el instante de ESTE frame (una sola lectura por frame)
            pose_tcp = robot.pose() if depth is not None else None

            r = model(frame, conf=a.conf, imgsz=640, verbose=False)[0]
            salida = []
            for b in r.boxes:
                nombre = r.names[int(b.cls[0])]
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                xyz = None
                if depth is not None and nombre in RECOGER and pose_tcp is not None:
                    u = int((x1 + x2) / 2)
                    v = int((y1 + y2) / 2)
                    region_depth = depth[max(0, v - 4):v + 5, max(0, u - 4):u + 5]
                    region_depth = region_depth[region_depth > 0]
                    if region_depth.size:
                        z = float(np.median(region_depth)) * scale
                        # Punto en el marco de la cámara (metros)
                        p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
                        P_CAM = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])

                        # cámara -> TCP -> base
                        P_TCP = T_TCP_CAM @ P_CAM
                        P_BASE = pose_a_matriz(pose_tcp) @ P_TCP

                        xyz = [P_BASE[0] * 1000, P_BASE[1] * 1000, P_BASE[2] * 1000]
                        salida.append((nombre, xyz, P_TCP[:3] * 1000, np.array(p_cam) * 1000))
                dibujar(frame, (x1, y1, x2, y2), nombre, float(b.conf[0]), xyz)

            if depth is not None and pose_tcp is None:
                cv2.putText(frame, "SIN LECTURA DEL ROBOT", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if salida and time.time() - ultimo > 1.0:
                for n, pb, ptcp, pc in salida:
                    print(f"{n}: CAM=({pc[0]:.0f},{pc[1]:.0f},{pc[2]:.0f})  "
                          f"TCP=({ptcp[0]:.0f},{ptcp[1]:.0f},{ptcp[2]:.0f})  "
                          f"BASE=({pb[0]:.0f},{pb[1]:.0f},{pb[2]:.0f}) mm")
                ultimo = time.time()

            cv2.imshow("cosecha (q = salir)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q"), 27):
                break
    finally:
        if pipe:
            pipe.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="runs/detect/fresas_3clases/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--img", help="probar con una foto (sin camara)")
    ap.add_argument("--webcam", action="store_true", help="webcam normal, sin 3D")
    a = ap.parse_args()
    model = YOLO(a.modelo)
    if a.img:
        frame = cv2.imread(a.img)
        r = model(frame, conf=a.conf, imgsz=640, verbose=False)[0]
        for b in r.boxes:
            dibujar(frame, b.xyxy[0].tolist(), r.names[int(b.cls[0])], float(b.conf[0]))
        cv2.imshow("detectar", frame)
        cv2.waitKey(0)
    else:
        bucle(model, a)

main()