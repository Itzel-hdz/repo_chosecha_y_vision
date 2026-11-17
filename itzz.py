import argparse, json, shutil, time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation

# =============================================================================
# 1. CONFIG   (lo marcado "AJUSTAR" son valores por medir)
# =============================================================================
MODELO = "runs/detect/fresas_3clases/weights/best.pt"
CONF = 0.5
RECOGER = {"grande_roja", "chica_roja"}            # clases a cosechar (nombres de data.yaml)
SALIDA = Path("objetivos")
ROBOT_IP = "192.168.0.10"

# Camara -> TCP (calibracion mano-ojo, calibrar_camara.py, 25 poses). Metros.
# Marco de la camara = nativo de la D435 (X derecha, Y abajo, Z adelante).
# OJO: vale solo mientras la camara siga rigida en el mismo lugar y NO se cambie el TCP del robot.
# Cuando recalibres con el filtrado de poses, reemplaza SOLO este bloque.
T_TCP_CAM = np.array([
    [-0.999500,  0.028256,  0.014210,  0.020598],
    [-0.029022, -0.997956, -0.056933,  0.080521],
    [ 0.012572, -0.057317,  0.998277, -0.113376],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
])

# La profundidad mide la SUPERFICIE de la fresa; se suma el radio para apuntar al centro.
RADIO_FRESA = {"grande_roja": 0.018, "chica_roja": 0.0165}    # [m]

Z_MIN, Z_MAX = 0.15, 1.0     # [m] rango de profundidad aceptado (la D435 es poco fiable muy cerca)
ORIENTACION = None           # [rx,ry,rz] fija de agarre. None = la misma de la pose de observacion
DIST_APROX = 0.05            # [m] aproximacion: retroceder sobre el eje z de la herramienta
VEL_QUIETO = 0.005           # [m/s] por debajo de esto el TCP se considera quieto

# Seguridad. AJUSTAR.
# ZONA = ((xmin,xmax),(ymin,ymax),(zmin,zmax)) en metros, en la BASE del robot. Obligatoria con --mover.
# Para medirla: con freedrive lleva el TCP a dos esquinas opuestas del volumen libre y lee la pose en el pendant.
ZONA = None
MAX_DIST_OBS = 0.35          # [m] el objetivo no puede estar mas lejos que esto de la pose de observacion

# Movimiento (primeras pruebas lentas)
V_J, A_J = 0.5, 1.0          # moveJ_IK  [rad/s, rad/s2]
V_ENTRAR, V_SALIR, A_L = 0.05, 0.10, 0.3     # moveL [m/s, m/s, m/s2]
GRIPPER_DO, T_GRIPPER = 0, 0.8               # AJUSTAR: salida digital del gripper (revisa si cierra con True o False)

# ArUco (AJUSTAR: diccionario y lado REAL impreso del marcador, en metros)
ARUCO_DICT = "DICT_4X4_50"
ARUCO_LADO = 0.05
ARUCO_ZONA = {}              # opcional: {id_del_marcador: "A"} para nombrar la zona por ArUco


# =============================================================================
# 2. VISION   (marco: CAMARA)
# =============================================================================
def abrir_camara():
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)                 # mismo pixel = mismo punto en color y profundidad
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    return rs, pipe, align, intr, scale


def depth_estable(frames):
    """Mediana pixel a pixel de varios cuadros de profundidad (quita ruido). 0 = sin dato."""
    pila = np.stack(frames).astype(np.float32)
    pila[pila == 0] = np.nan
    med = np.nanmedian(pila, axis=0)
    return np.nan_to_num(med, nan=0).astype(np.uint16)


def detectar_fresas(model, frame, depth, rs, intr, scale, conf, zmin=Z_MIN, zmax=Z_MAX):
    """Fresas a recoger con profundidad valida. Cada una: clase, conf, caja, u, v, z_m, p_cam (m).
    z_m = distancia al CENTRO de la fresa (superficie medida + radio). Ordenadas: 1 = la mas cercana."""
    r = model(frame, conf=conf, imgsz=640, agnostic_nms=True, verbose=False)[0]   # 1 recuadro por fresa
    fresas = []
    for b in r.boxes:
        clase = r.names[int(b.cls[0])]
        if clase not in RECOGER:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)              # pixel central
        ventana = depth[max(0, v - 4):v + 5, max(0, u - 4):u + 5]  # mediana 9x9; 0 = sin dato
        ventana = ventana[ventana > 0]
        if not ventana.size:
            continue
        z = float(np.median(ventana)) * scale + RADIO_FRESA.get(clase, 0.0)
        if not (zmin <= z <= zmax):
            continue
        p_cam = np.array(rs.rs2_deproject_pixel_to_point(intr, [u, v], z))    # X=(u-cx)z/fx, Y=(v-cy)z/fy
        fresas.append({"clase": clase, "conf": float(b.conf[0]), "caja": [x1, y1, x2, y2],
                       "u": u, "v": v, "z_m": z, "p_cam": p_cam})
    fresas.sort(key=lambda f: f["z_m"])
    return fresas


def _detector_aruco():
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        return det.detectMarkers
    p = cv2.aruco.DetectorParameters_create()
    return lambda img: cv2.aruco.detectMarkers(img, d, parameters=p)


def detectar_arucos(frame, intr, lado=ARUCO_LADO):
    """Marcadores ArUco visibles. Cada uno: id, esquinas, T_cam_marker (4x4, metros)."""
    if not hasattr(cv2, "aruco"):
        return []
    esquinas, ids, _ = _detector_aruco()(frame)
    if ids is None:
        return []
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
    h = lado / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)   # orden de esquinas ArUco
    out = []
    for c, i in zip(esquinas, ids.flatten()):
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float32), K, np.zeros(5),
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.flatten()
        out.append({"id": int(i), "esquinas": c.reshape(4, 2), "T_cam_marker": T})
    return out


# =============================================================================
# 3. TRANSFORMADAS   (metros)
# =============================================================================
def pose_a_matriz(pose):
    """Pose UR [x,y,z,rx,ry,rz] (vector de rotacion, NO roll-pitch-yaw) -> matriz 4x4."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    return T


def matriz_a_pose(T):
    return list(T[:3, 3]) + list(Rotation.from_matrix(T[:3, :3]).as_rotvec())


def camara_a_tcp(p_cam):
    return (T_TCP_CAM @ np.append(p_cam, 1.0))[:3]


def tcp_a_base(p_tcp, pose_tcp):
    return (pose_a_matriz(pose_tcp) @ np.append(p_tcp, 1.0))[:3]


def pose_objetivo(p_base, pose_obs):
    """Posicion de la fresa + orientacion FIJA (la de la pose de observacion, o ORIENTACION)."""
    rot = ORIENTACION if ORIENTACION is not None else list(pose_obs[3:6])
    return [float(p_base[0]), float(p_base[1]), float(p_base[2])] + [float(x) for x in rot]


def pose_aprox(objetivo, dist=DIST_APROX):
    """Retroceder 'dist' sobre el eje z de la herramienta."""
    T = pose_a_matriz(objetivo)
    T[:3, 3] = (T @ np.array([0, 0, -dist, 1.0]))[:3]
    return matriz_a_pose(T)


def en_zona(p):
    if ZONA is None:
        return None
    return all(lo <= p[i] <= hi for i, (lo, hi) in enumerate(ZONA))


# =============================================================================
# 4. ROBOT   (RTDE: pip install ur_rtde)
# =============================================================================
class Robot:
    """Lectura con reconexion automatica (pasar el robot a manual/freedrive corta RTDE).
    El control (movimiento) solo se abre al cosechar y se cierra al terminar."""

    def __init__(self, ip, mover=False):
        self.ip, self.mover = ip, mover
        self.rx = None
        try:
            self._conectar()
        except Exception as e:
            print("AVISO: no se pudo conectar al robot todavia (%s). Se reintentara al leer." % e)

    def _conectar(self):
        import rtde_receive
        self.rx = rtde_receive.RTDEReceiveInterface(self.ip)

    def _leer(self, f):
        for i in range(8):
            try:
                if self.rx is None:
                    self._conectar()
                return f(self.rx)
            except Exception:
                self.rx = None
                print("  Robot no responde, reintentando (%d/8)..." % (i + 1))
                time.sleep(1.0)
        raise RuntimeError("No se pudo leer el robot. Revisa que este encendido, en red y sin otra herramienta conectada.")

    def pose_tcp(self):
        return self._leer(lambda r: [float(x) for x in r.getActualTCPPose()])

    def quieto(self):
        return self._leer(lambda r: float(np.linalg.norm(r.getActualTCPSpeed()[:3])) < VEL_QUIETO)

    def cosechar(self, objetivo, aprox, pose_obs, solo_aprox=False, esperar=True):
        """Bloquea hasta terminar cada movimiento. El robot debe estar en modo REMOTO (no manual)."""
        import rtde_control, rtde_io
        try:
            ctrl = rtde_control.RTDEControlInterface(self.ip)
            io = rtde_io.RTDEIOInterface(self.ip)
        except Exception as e:
            print("No se pudo tomar el control del robot (%s)." % e)
            print("Pon el robot en modo REMOTO en el pendant y que no este en freedrive ni en paro.")
            return False
        try:
            ctrl.moveJ_IK(aprox, V_J, A_J)                   # ir a DIST_APROX de la fresa
            if solo_aprox:
                print("Robot en el punto de aproximacion. Revisa la punta contra la fresa.")
                if esperar:
                    input("ENTER para volver a la pose de observacion...")
                else:
                    time.sleep(3.0)
            else:
                ctrl.moveL(objetivo, V_ENTRAR, A_L)          # entrar en linea recta, lento
                io.setStandardDigitalOut(GRIPPER_DO, True)   # cerrar gripper
                time.sleep(T_GRIPPER)
                ctrl.moveL(aprox, V_SALIR, A_L)              # retirar en linea recta
            ctrl.moveJ_IK(pose_obs, V_J, A_J)                # volver a la pose de observacion
            return True
        finally:
            try:
                ctrl.stopScript()
            except Exception:
                pass

    def cerrar(self):
        try:
            if self.rx is not None:
                self.rx.disconnect()
        except Exception:
            pass


# =============================================================================
# 5. REGISTRO Y GUARDADO
# =============================================================================
def construir_registro(fresas, arucos, pose, quieto, elegida, zona):
    """Calcula p_tcp / p_base / p_marker, el objetivo y la aproximacion; devuelve (dict_json, datos)."""
    T_base_tcp = pose_a_matriz(pose) if pose is not None else None
    marc = []
    for m in arucos:
        d = {"id": m["id"], "p_cam": np.round(m["T_cam_marker"][:3, 3], 4).tolist(), "p_base": None}
        if T_base_tcp is not None:
            d["p_base"] = np.round((T_base_tcp @ T_TCP_CAM @ m["T_cam_marker"])[:3, 3], 4).tolist()
        marc.append(d)
    lista = []
    for n, f in enumerate(fresas, 1):
        p_tcp = camara_a_tcp(f["p_cam"])
        p_base = tcp_a_base(p_tcp, pose) if pose is not None else None
        p_marker = {str(m["id"]): np.round((np.linalg.inv(m["T_cam_marker"]) @ np.append(f["p_cam"], 1.0))[:3], 4).tolist()
                    for m in arucos}
        lista.append({"n": n, "clase": f["clase"], "conf": round(f["conf"], 3), "u": f["u"], "v": f["v"],
                      "z_m": round(f["z_m"], 4), "p_cam": np.round(f["p_cam"], 4).tolist(),
                      "p_tcp": np.round(p_tcp, 4).tolist(),
                      "p_base": None if p_base is None else np.round(p_base, 4).tolist(),
                      "p_marker": p_marker})
    sel = lista[elegida]
    objetivo = aprox = ok_zona = dist_obs = None
    if sel["p_base"] is not None:
        objetivo = pose_objetivo(sel["p_base"], pose)
        aprox = pose_aprox(objetivo)
        dist_obs = float(np.linalg.norm(np.array(objetivo[:3]) - np.array(pose[:3])))
        z1, z2 = en_zona(objetivo[:3]), en_zona(aprox[:3])
        ok_zona = None if z1 is None else bool(z1 and z2 and dist_obs <= MAX_DIST_OBS)
    if zona is None and arucos:
        zona = next((ARUCO_ZONA[m["id"]] for m in arucos if m["id"] in ARUCO_ZONA), None)
    reg = {"fecha": datetime.now().isoformat(timespec="seconds"), "unidades": "metros", "zona": zona,
           "pose_obs": None if pose is None else [round(x, 5) for x in pose], "tcp_quieto": quieto,
           "arucos": marc, "fresas": lista, "elegida": sel["n"],
           "objetivo": None if objetivo is None else [round(x, 5) for x in objetivo],
           "aproximacion": None if aprox is None else [round(x, 5) for x in aprox],
           "dist_a_pose_obs": None if dist_obs is None else round(dist_obs, 4), "en_zona": ok_zona}
    return reg, {"objetivo": objetivo, "aprox": aprox, "en_zona": ok_zona, "dist_obs": dist_obs}


def guardar(reg, imagen):
    SALIDA.mkdir(exist_ok=True)
    nombre = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ruta = SALIDA / f"{nombre}.json"
    ruta.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    cv2.imwrite(str(SALIDA / f"{nombre}.png"), imagen)
    shutil.copyfile(ruta, SALIDA / "ultimo_objetivo.json")
    return ruta


# =============================================================================
# 6. MAIN
# =============================================================================
def dibujar(img, fresas, arucos=(), numerar=False):
    for i, f in enumerate(fresas, 1):
        x1, y1, x2, y2 = map(int, f["caja"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        t = f"#{i} {f['clase']} {f['conf']:.2f}" if numerar else f"{f['clase']} {f['conf']:.2f}"
        cv2.putText(img, t, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        cv2.putText(img, "z=%.0f mm" % (f["z_m"] * 1000), (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    for m in arucos:
        pts = m["esquinas"].astype(int)
        cv2.polylines(img, [pts], True, (255, 150, 0), 2)
        cv2.putText(img, f"ArUco {m['id']}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 2)


def texto(img, s, y=20):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def elegir(img, n):
    """Espera el numero de la fresa. Devuelve indice, o None si se cancela."""
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord("r"), ord("R")):
            return None
        if k == 13:
            return 0
        if ord("1") <= k <= ord("9") and k - ord("1") < n:
            return k - ord("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=MODELO)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--robot", default=ROBOT_IP, help=f"IP del UR (por defecto {ROBOT_IP})")
    ap.add_argument("--sin-robot", action="store_true", help="no conectar al robot (solo p_cam/p_tcp, no hay p_base)")
    ap.add_argument("--mover", action="store_true", help="ademas de guardar, MUEVE el robot (requiere robot y ZONA)")
    ap.add_argument("--solo-aprox", action="store_true",
                    help="con --mover: va SOLO al punto de aproximacion (sin entrar ni cerrar el gripper). Para medir el error")
    ap.add_argument("--auto", action="store_true",
                    help="con --mover: AUTOMATICO. Toma la foto solo, elige la fresa alcanzable mas cercana y la cosecha, en ciclo. q = parar")
    ap.add_argument("--max-ciclos", type=int, default=0, help="con --auto: parar tras N cosechas (0 = sin limite)")
    ap.add_argument("--zona", help="nombre de la pose de observacion (A, B...) para el registro")
    ap.add_argument("--zmin", type=float, default=Z_MIN)
    ap.add_argument("--zmax", type=float, default=Z_MAX)
    a = ap.parse_args()
    if a.sin_robot:
        a.robot = None
    if a.mover and not a.robot:
        ap.error("--mover necesita el robot (quita --sin-robot)")
    if a.mover and ZONA is None:
        ap.error("--mover necesita definir ZONA en la seccion CONFIG (volumen permitido en la base)")
    if a.solo_aprox and not a.mover:
        ap.error("--solo-aprox se usa junto con --mover")
    if a.auto and not a.mover:
        ap.error("--auto se usa junto con --mover")

    model = YOLO(a.modelo)
    robot = Robot(a.robot, a.mover) if a.robot else None
    rs, pipe, align, intr, scale = abrir_camara()
    ultimos = deque(maxlen=5)                           # ultimos cuadros de profundidad
    estables = 0                                        # cuadros seguidos con fresa detectada (modo auto)
    proximo = 0.0                                       # no disparar antes de este instante (modo auto)
    ciclos = 0
    ult_aviso = 0.0
    try:
        while True:
            fr = align.process(pipe.wait_for_frames())
            c, d = fr.get_color_frame(), fr.get_depth_frame()
            if not c or not d:
                continue
            frame, depth = np.asanyarray(c.get_data()), np.asanyarray(d.get_data())
            ultimos.append(depth)
            vista = frame.copy()
            en_vivo = detectar_fresas(model, frame, depth, rs, intr, scale, a.conf, a.zmin, a.zmax)
            dibujar(vista, en_vivo, detectar_arucos(frame, intr))
            texto(vista, ("AUTO: q = parar | ciclos: %d" % ciclos) if a.auto else "ESPACIO = foto | q = salir")
            cv2.imshow("cosecha_fresas", vista)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                break

            if a.auto:
                estables = estables + 1 if en_vivo else 0
                if estables < 8 or time.time() < proximo:      # fresa estable ~0.3 s y sin enfriamiento
                    continue
            elif k != 32:
                continue

            # ---- FOTO: pose y deteccion del MISMO instante ----
            pose = quieto = None
            if robot:
                try:
                    pose, quieto = robot.pose_tcp(), robot.quieto()
                except RuntimeError as e:
                    print(e)
                    proximo = time.time() + 3
                    continue
                if not quieto:
                    if time.time() - ult_aviso > 2:
                        print("AVISO: el robot se mueve; foto y pose no coinciden. Espera a que se detenga.")
                        ult_aviso = time.time()
                    continue
            depth_f = depth_estable(list(ultimos))
            fresas = detectar_fresas(model, frame, depth_f, rs, intr, scale, a.conf, a.zmin, a.zmax)
            arucos = detectar_arucos(frame, intr)
            if not fresas:
                print("No se detecto ninguna fresa a recoger con profundidad valida.")
                proximo = time.time() + 1
                continue
            congelada = frame.copy()
            dibujar(congelada, fresas, arucos, numerar=True)

            if a.auto:
                # primera fresa (la mas cercana a la camara) cuyo objetivo cae dentro de ZONA y de MAX_DIST_OBS
                i, reg, dat = 0, None, None
                for j in range(len(fresas)):
                    reg_j, dat_j = construir_registro(fresas, arucos, pose, quieto, j, a.zona)
                    if reg is None:
                        i, reg, dat = j, reg_j, dat_j          # por defecto la #1, para registrar por que no se movio
                    if dat_j["en_zona"] is True:
                        i, reg, dat = j, reg_j, dat_j
                        break
            else:
                texto(congelada, "Numero = elegir | Enter = #1 | Esc = cancelar")
                cv2.imshow("cosecha_fresas", congelada)
                i = elegir(congelada, len(fresas))
                if i is None:
                    print("Cancelado.")
                    continue
                reg, dat = construir_registro(fresas, arucos, pose, quieto, i, a.zona)

            ruta = guardar(reg, congelada)
            s = reg["fresas"][i]
            print(f"\nGuardado: {ruta}   (fresa #{s['n']} {s['clase']}, ArUco vistos: {[m['id'] for m in arucos]})")
            print("  p_cam  :", s["p_cam"], "m")
            print("  p_base :", s["p_base"] if s["p_base"] else "(sin --robot)", "m")
            print("  objetivo:", reg["objetivo"], "\n  aprox   :", reg["aproximacion"],
                  "\n  dist a pose obs:", reg["dist_a_pose_obs"], "m   en_zona:", reg["en_zona"])

            # ---- MOVIMIENTO (solo con --mover) ----
            if a.mover:
                if dat["en_zona"] is not True:
                    print("NO se mueve: objetivo/aproximacion fuera de ZONA, o a mas de %.2f m de la pose de observacion." % MAX_DIST_OBS)
                    proximo = time.time() + 3
                    continue
                if a.auto:
                    print(">>> AUTO: %s" % ("va al punto de aproximacion" if a.solo_aprox else "cosechando"))
                    ok = robot.cosechar(dat["objetivo"], dat["aprox"], pose, a.solo_aprox, esperar=False)
                else:
                    modo = "IR AL PUNTO DE APROXIMACION" if a.solo_aprox else "COSECHAR"
                    texto(congelada, f"{modo}? g = si | otra tecla = no", 40)
                    cv2.imshow("cosecha_fresas", congelada)
                    if cv2.waitKey(0) & 0xFF != ord("g"):
                        print("No confirmado.")
                        continue
                    ok = robot.cosechar(dat["objetivo"], dat["aprox"], pose, a.solo_aprox)
                ultimos.clear()
                estables = 0
                if ok:
                    ciclos += 1
                    print("Listo (ciclo %d)." % ciclos)
                    if a.auto:
                        proximo = time.time() + 2.0            # deja que el robot se asiente en la pose de observacion
                        if a.max_ciclos and ciclos >= a.max_ciclos:
                            print("Se alcanzo --max-ciclos. Fin.")
                            break
                    else:
                        print("Toma una foto nueva (ESPACIO).")
                elif a.auto:
                    print("AUTO detenido: el robot no acepto el movimiento. Revisa modo remoto en el pendant.")
                    break
    finally:
        pipe.stop()
        if robot:
            robot.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()