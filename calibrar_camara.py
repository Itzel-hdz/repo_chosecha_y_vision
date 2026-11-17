import cv2, numpy as np, pyrealsense2 as rs, time
from rtde_receive import RTDEReceiveInterface

# ---------- CONFIGURACIÓN ----------
COLS, ROWS = 8, 6          # esquinas interiores
SQ = 0.025                 # lado del cuadro en metros
ROBOT_IP = "192.168.0.10"
# -----------------------------------

def leer_pose():
    for i in range(10):
        try:
            r = RTDEReceiveInterface(ROBOT_IP)
            p = r.getActualTCPPose()
            r.disconnect()
            return p
        except Exception:
            print("  Robot no responde, reintentando (%d/10)..." % (i + 1))
            time.sleep(1.5)
    raise RuntimeError("No se pudo leer el robot.")

def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M

def orden_fijo(c):
    # la primera esquina siempre es la más alta en la imagen
    if c[0, 0, 1] > c[-1, 0, 1]:
        c = c[::-1].copy()
    return c

pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
prof = pipe.start(cfg)
i_ = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[i_.fx, 0, i_.ppx], [0, i_.fy, i_.ppy], [0, 0, 1]])
D = np.array(i_.coeffs)

objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQ
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

Ttcp2base, Tboard2cam = [], []
print("Mueve el robot (freedrive) a una pose donde se vea el tablero.")
print("ESPACIO = guardar pose | q = calcular (mínimo 12 poses)")
while True:
    f = pipe.wait_for_frames().get_color_frame()
    img = np.asanyarray(f.get_data())
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCorners(g, (COLS, ROWS), flags)
    vis = img.copy()
    if ok:
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit)
        c = orden_fijo(c)
        cv2.drawChessboardCorners(vis, (COLS, ROWS), c, ok)   # el rojo siempre debe quedar ARRIBA
    cv2.putText(vis, "poses: %d  %s" % (len(Ttcp2base), "DETECTADO" if ok else "NO detecta"),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.imshow("calibracion", vis)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' ') and ok:
        print("Suelta el robot y no lo toques...")
        time.sleep(2.0)
        for _ in range(10):
            pipe.wait_for_frames()
        rvs, tvs = [], []
        for _ in range(15):
            f2 = pipe.wait_for_frames().get_color_frame()
            g2 = cv2.cvtColor(np.asanyarray(f2.get_data()), cv2.COLOR_BGR2GRAY)
            ok2, c2 = cv2.findChessboardCorners(g2, (COLS, ROWS), flags)
            if ok2:
                c2 = cv2.cornerSubPix(g2, c2, (11, 11), (-1, -1), crit)
                c2 = orden_fijo(c2)
                _, rv_, tv_ = cv2.solvePnP(objp, c2, K, D)
                rvs.append(rv_.ravel()); tvs.append(tv_.ravel())
        if len(rvs) < 8:
            print("No se detectó estable, se descarta esta pose."); continue
        p = leer_pose()
        Tboard2cam.append(T(cv2.Rodrigues(np.mean(rvs, 0))[0], np.mean(tvs, 0)))
        Ttcp2base.append(T(cv2.Rodrigues(np.array(p[3:]))[0], p[:3]))
        print("Pose %d guardada (desv. frames: %.2f mm)" % (len(Ttcp2base), np.std(tvs, 0).mean() * 1000))
    if k == ord('q'):
        break
pipe.stop(); cv2.destroyAllWindows()

if len(Ttcp2base) < 8:
    raise SystemExit("Muy pocas poses (%d). Haz al menos 12." % len(Ttcp2base))

mejor = None
for nombre, m in [("TSAI", cv2.CALIB_HAND_EYE_TSAI), ("PARK", cv2.CALIB_HAND_EYE_PARK),
                  ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
                  ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
                  ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)]:
    R, t = cv2.calibrateHandEye(
        [x[:3, :3] for x in Ttcp2base], [x[:3, 3] for x in Ttcp2base],
        [x[:3, :3] for x in Tboard2cam], [x[:3, 3] for x in Tboard2cam], method=m)
    Tc = T(R, t)
    pts = np.array([(a @ Tc @ b)[:3, 3] for a, b in zip(Ttcp2base, Tboard2cam)])
    dev = np.linalg.norm(pts.std(axis=0)) * 1000
    print("%-10s desviación: %.1f mm" % (nombre, dev))
    if mejor is None or dev < mejor[0]:
        mejor = (dev, nombre, Tc)
print("\nMejor: %s (%.1f mm)" % (mejor[1], mejor[0]))
T_cam2tcp = mejor[2]

pts = np.array([(a @ T_cam2tcp @ b)[:3, 3] for a, b in zip(Ttcp2base, Tboard2cam)])
print("\nDesviación del tablero en la base (mm):", (pts.std(axis=0) * 1000).round(2),
      " (debe ser < 3-5 mm)")
errs = np.linalg.norm(pts - pts.mean(0), axis=1) * 1000
for n, e in enumerate(errs):
    print("Pose %d: %.1f mm" % (n + 1, e))

print("\nCopia y pega en detectar.py:")
print("T_cam2tcp = np.array([")
for fila in T_cam2tcp:
    print("    [" + ", ".join("%.6f" % v for v in fila) + "],")
print("])")