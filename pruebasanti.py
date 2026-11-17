import time
import cv2
import numpy as np
import rtde_control
import rtde_receive
import pyrealsense2 as rs

from ultralytics import YOLO
from scipy.spatial.transform import Rotation


# ============================================================
# 1. CONFIGURACION GENERAL
# ============================================================

ROBOT_IP = "192.168.0.10"
MODELO = "runs/detect/fresas_3clases/weights/best.pt"
CONFIDENCE = 0.5
RECOGER = {"grande_roja", "chica_roja"}

# Separacion vertical en pixeles para considerar detecciones de una
# misma fila. Ajustar usando las fotografias reales.
TOLERANCIA_Y_FILAS = 60

# Tiempo de espera despues de posicionar el robot para que la imagen
# y la profundidad se estabilicen.
TIEMPO_ESTABILIZACION_CAMARA = 0.5


# ============================================================
# 2. VELOCIDADES Y ESPERA DE MOVIMIENTO
# ============================================================

VELOCIDAD_J = 0.20
ACELERACION_J = 0.30
VELOCIDAD_L = 0.05
ACELERACION_L = 0.10

TIMEOUT_MOVIMIENTO = 90.0
VELOCIDAD_PARADO = 0.005
TIEMPO_ESTABLE = 0.5
TOLERANCIA_POSICION = 0.003
TOLERANCIA_ORIENTACION = 0.05


# ============================================================
# 3. POSICIONES GENERALES
# ============================================================

# HOME general. Se conserva el valor que ya tenias.
HOME = [
    0.1056, -0.0349, 0.4967,
    1.46, -0.730, -0.723
]

# IMPORTANTE:
# Debes introducir las poses CARTESIANAS [X,Y,Z,Rx,Ry,Rz] del TCP
# para tomar cada fotografia. Unidades: metros y radianes.
# No inventes ni reutilices una pose si la camara queda en otra posicion.
POS_FOTO_SECCION_1 = [0.26936, -0.08684, 0.38902, 1.588, -0.646, -0.594]
POS_FOTO_SECCION_2 = [0.00135, 0.24945, 0.41557, 1.494, -0.898, -0.758]

# Trayecto a la foto de charola 1, en 3 pasos suaves (grabar en Polyscope)
APROX_CHAROLA_1 = [
    [0.60703, 0.15518, 0.43543, 0.858, 1.444, 1.570],
    [0.52299, 0.12144, 0.42893, 1.074, 2.191, 0.966],
]
POS_FOTO_CHAROLA_1 = [0.33169, 0.03160, 0.39839, 1.154, 2.731, 0.376]
POS_FOTO_CHAROLA_2 = [0.15211, 0.29102, 0.38467, 1.276, 2.995, 0.311]


# ============================================================
# 4. MATRIZ DE TRANSFORMACION CAMARA -> TCP
# ============================================================

T_TCP_CAM = np.array([
    [-0.999500,  0.028256,  0.014210,  0.020598],
    [-0.029022, -0.997956, -0.056933,  0.080521],
    [ 0.012572, -0.057317,  0.998277, -0.113376],
    [ 0.000000,  0.000000,  0.000000,  1.000000]
], dtype=float)


# ============================================================
# 5. REFERENCIAS Y RUTINAS DE AGARRE
# ============================================================

# Referencia de la rutina que ya probaste para la seccion 1.
FRESA_REFERENCIA_SECCION_1 = np.array([
    -0.22696,
    -0.26570,
     0.522
], dtype=float)

# Puntos originales de la rutina actual de la seccion 1.
# Cada punto: [X, Y, Z, Rx, Ry, Rz]
RUTINA_SECCION_1 = [
    # PUNTO 1: rotacion de camara (se conserva como referencia; no se ejecuta abajo)
    [-0.23699, -0.27200, 0.53000, 1.278, 2.529, -2.498],

    # PUNTO 2: Pre-pick
    [-0.25508, -0.27568, 0.45000, 0.130, -0.001, -1.128],

    # PUNTO 3: Pick
    [-0.25508, -0.27568, 0.51000, 0.130, -0.001, -1.128],

    # PUNTO 4: regreso a pre-pick
    [-0.25508, -0.27568, 0.45000, 0.130, -0.001, -1.128],

    # PUNTO 5: retirada
    [-0.17696, -0.23913, 0.45000, 0.130, -0.001, -1.128],

    # PUNTO 6: orientacion de regreso
    [0.29488, -0.08080, 0.48713, 0.067, 0.108, 0.924],
]

# Aproximaciones que ya tenias para la primera fresa de la seccion 1.
APROXIMACIONES_SECCION_1 = [
    [-0.05696, -0.22259, 0.19395, 2.912, -1.179, 0.000],
    [-0.05696, -0.22182, 0.61777, 0.001, 0.003, 2.199],
    [-0.05696, -0.22182, 0.61777, 0.018, -0.005, 5.427],
    [-0.02896, -0.301416, 0.35735, 0.032, -0.003, 5.766],
]

# CONFIGURACION DE SECCION 2:
# Debes colocar aqui los puntos reales de la rutina y aproximaciones de
# la segunda seccion. Se dejan como None para evitar que el robot use
# accidentalmente una trayectoria de la seccion 1 en otra geometria.
FRESA_REFERENCIA_SECCION_2 = np.array([
    -0.22696,
    -0.26570,
     0.522
], dtype=float)  # TODO: [x, y, z] de la fresa de referencia 2
RUTINA_SECCION_2 = [
    # PUNTO 1: rotacion de camara (se conserva como referencia; no se ejecuta abajo)
    [-0.23699, -0.27200, 0.53000, 1.278, 2.529, -2.498],

    # PUNTO 2: Pre-pick
    [-0.25508, -0.27568, 0.45000, 0.130, -0.001, -1.128],

    # PUNTO 3: Pick
    [-0.25508, -0.27568, 0.51000, 0.130, -0.001, -1.128],

    # PUNTO 4: regreso a pre-pick
    [-0.25508, -0.27568, 0.45000, 0.130, -0.001, -1.128],

    # PUNTO 5: retirada
    [-0.17696, -0.23913, 0.45000, 0.130, -0.001, -1.128],

    # PUNTO 6: orientacion de regreso
    [-0.27905, 0.12496, 0.48713, 0.127, -0.093, -2.366],
]           # TODO: lista de 6 puntos [x,y,z,rx,ry,rz]
APROXIMACIONES_SECCION_2 = [
    [0.22560, -0.20637, 0.48713, 0.088, 0.087, 0.451],
    [-0.06070, -0.29967, 0.48713, 0.120, 0.032, -0.577],
    [-0.27357, -0.11967, 0.34914, 0.118, 0.041, -0.713],
]  # TODO: lista de 3 poses [x,y,z,rx,ry,rz]



MOVIMIENTOS_FIJOS_SECCION_1 = [
    [0.29940, 0.18974, 0.52150, 0.239, -0.067, -4.449],
    [0.29940, 0.18974, 0.52150, 0.044, 0.291, 4.705],
    [0.34549, -0.03408, 0.32356, 1.057, 1.348, -1.181],
    [0.32505, 0.17330, 0.13001, 1.887, 2.482, -0.004],
]

MOVIMIENTOS_FIJOS_SECCION_2 = [
    [0.07662, 0.42718, 0.55293, 0.162, 0.038, -3.543],
    [0.08594, 0.42181, 0.55348, 0.042, -0.037, -0.656],
    [0.27940, 0.32180, 0.33905, 0.347, 1.565, -0.476],
    [0.21846, 0.36210, 0.14370, 1.154, 2.965, -0.001],
]

# ============================================================
# 6. TRANSFORMACIONES
# ============================================================

def pose_a_matriz(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    return T


def transformar_a_base(p_cam, pose_tcp):
    P_CAM = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=float)
    P_TCP = T_TCP_CAM @ P_CAM
    T_BASE_TCP = pose_a_matriz(pose_tcp)
    P_BASE = T_BASE_TCP @ P_TCP
    return P_BASE[:3]


def generar_rutina_agarre(xyz_fresa, rutina_base, fresa_referencia):
    """Desplaza XYZ de los puntos 1 a 5 respecto a la fresa de referencia.
    El punto 6 conserva su pose fija, igual que en el codigo original.
    """
    xyz_fresa = np.asarray(xyz_fresa, dtype=float)
    fresa_referencia = np.asarray(fresa_referencia, dtype=float)
    desplazamiento = xyz_fresa - fresa_referencia

    rutina_nueva = []
    for i, punto in enumerate(rutina_base):
        nuevo_punto = np.asarray(punto, dtype=float).copy()
        if i < 5:
            nuevo_punto[:3] += desplazamiento
        rutina_nueva.append(nuevo_punto.tolist())
    return rutina_nueva


# ============================================================
# 7. ESPERA Y FUNCIONES DE MOVIMIENTO
# ============================================================

def error_orientacion(actual, objetivo):
    R_actual = Rotation.from_rotvec(actual[3:6])
    R_objetivo = Rotation.from_rotvec(objetivo[3:6])
    return (R_objetivo * R_actual.inv()).magnitude()


def esperar_movimiento(receptor, objetivo, timeout=TIMEOUT_MOVIMIENTO):
    inicio = time.time()
    inicio_estable = None
    error_pos = float("inf")
    error_rot = float("inf")
    rapidez = float("inf")

    while time.time() - inicio < timeout:
        pose_actual = receptor.getActualTCPPose()
        velocidad_tcp = receptor.getActualTCPSpeed()

        error_pos = np.linalg.norm(
            np.asarray(pose_actual[:3]) - np.asarray(objetivo[:3])
        )
        error_rot = error_orientacion(pose_actual, objetivo)
        rapidez = np.linalg.norm(np.asarray(velocidad_tcp))

        llego = (
            error_pos <= TOLERANCIA_POSICION
            and error_rot <= TOLERANCIA_ORIENTACION
        )
        esta_parado = rapidez <= VELOCIDAD_PARADO

        if llego and esta_parado:
            if inicio_estable is None:
                inicio_estable = time.time()
            elif time.time() - inicio_estable >= TIEMPO_ESTABLE:
                return True
        else:
            inicio_estable = None

        time.sleep(0.05)

    print("ERROR: timeout esperando movimiento.")
    print("Error posicion:", error_pos, "m")
    print("Error orientacion:", error_rot, "rad")
    print("Rapidez TCP:", rapidez)
    return False


def validar_pose(pose, nombre="pose"):
    if pose is None or len(pose) != 6:
        raise ValueError(f"Falta configurar {nombre} con 6 valores [x,y,z,rx,ry,rz].")
    if not np.all(np.isfinite(np.asarray(pose, dtype=float))):
        raise ValueError(f"{nombre} contiene valores no validos.")


def mover_j(robot, receptor, posicion):
    validar_pose(posicion, "objetivo MoveJ")
    print("\nMoveJ cartesiano:", posicion)
    resultado = robot.moveJ_IK(posicion, VELOCIDAD_J, ACELERACION_J)
    if not resultado:
        raise RuntimeError("RTDE no pudo iniciar MoveJ_IK.")
    if not esperar_movimiento(receptor, posicion):
        raise RuntimeError("El robot no llego al objetivo MoveJ_IK.")


def mover_l(robot, receptor, posicion):
    validar_pose(posicion, "objetivo MoveL")
    print("\nMoveL cartesiano:", posicion)
    resultado = robot.moveL(posicion, VELOCIDAD_L, ACELERACION_L)
    if not resultado:
        raise RuntimeError("RTDE no pudo iniciar MoveL.")
    if not esperar_movimiento(receptor, posicion):
        raise RuntimeError("El robot no llego al objetivo MoveL.")


def validar_rutina(rutina, nombre):
    if rutina is None or len(rutina) != 6:
        raise ValueError(f"{nombre} debe contener exactamente 6 puntos.")
    for i, punto in enumerate(rutina):
        validar_pose(punto, f"{nombre}, punto {i + 1}")


# ============================================================
# 8. ORDENAR FRESAS: ABAJO->ARRIBA, IZQUIERDA->DERECHA
# ============================================================

def ordenar_por_prioridad(fresas, tolerancia_y=TOLERANCIA_Y_FILAS):
    """Agrupa por filas usando pixel v. La fila inferior va primero;
    dentro de cada fila, las fresas van de izquierda a derecha.
    """
    if not fresas:
        return []

    # Procesar detecciones desde la parte inferior de la imagen.
    detecciones = sorted(
        fresas,
        key=lambda f: f["pixel"][1],
        reverse=True
    )

    filas = []
    for fresa in detecciones:
        y = fresa["pixel"][1]
        mejor_fila = None
        menor_diferencia = None

        # Buscar la fila cuyo centro vertical este mas cerca.
        for fila in filas:
            centro_y = float(np.mean([item["pixel"][1] for item in fila]))
            diferencia = abs(y - centro_y)
            if diferencia <= tolerancia_y and (
                menor_diferencia is None or diferencia < menor_diferencia
            ):
                mejor_fila = fila
                menor_diferencia = diferencia

        if mejor_fila is None:
            filas.append([fresa])
        else:
            mejor_fila.append(fresa)

    # Ordenar filas de abajo hacia arriba.
    filas.sort(
        key=lambda fila: float(np.mean([f["pixel"][1] for f in fila])),
        reverse=True
    )

    ordenadas = []
    for fila in filas:
        fila.sort(key=lambda f: f["pixel"][0])
        ordenadas.extend(fila)

    return ordenadas


# ============================================================
# 9. REALSENSE: INICIALIZACION Y CAPTURA
# ============================================================

def iniciar_camara():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Iniciando streams de RealSense...")
    profile = pipeline.start(config)
    device = profile.get_device()
    print("Camara conectada:", device.get_info(rs.camera_info.name))
    print("Numero de serie:", device.get_info(rs.camera_info.serial_number))

    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    scale = device.first_depth_sensor().get_depth_scale()
    print("RealSense inicializada correctamente.")
    return pipeline, align, intr, scale


def capturar_foto_seccion(pipeline, align, receptor, nombre_seccion):
    """Captura un par RGB/profundidad y registra la pose TCP de ESA foto."""
    print(f"\nPreparando fotografia de {nombre_seccion}...")
    time.sleep(TIEMPO_ESTABILIZACION_CAMARA)

    # Descartar algunos frames iniciales para evitar capturar una imagen vieja.
    for _ in range(5):
        pipeline.wait_for_frames(10000)

    frames = pipeline.wait_for_frames(10000)
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise RuntimeError(f"No se obtuvo RGB y profundidad para {nombre_seccion}.")

    frame = np.asanyarray(color_frame.get_data()).copy()
    depth = np.asanyarray(depth_frame.get_data()).copy()

    # La pose se registra en el instante de la captura, no se comparte
    # entre secciones.
    pose_tcp_foto = receptor.getActualTCPPose()

    print(f"Foto de {nombre_seccion} capturada.")
    print("Pose TCP de la foto:", pose_tcp_foto)

    return {
        "frame": frame,
        "depth": depth,
        "pose_tcp": np.asarray(pose_tcp_foto, dtype=float).copy()
    }


# ============================================================
# 10. DETECTAR FRESAS DE UNA FOTOGRAFIA
# ============================================================

def detectar_fresas(model, frame, depth, intr, scale, pose_tcp_foto, nombre_seccion):
    resultado = model(
        frame,
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )[0]

    fresas_detectadas = []
    alto, ancho = depth.shape[:2]

    for b in resultado.boxes:
        nombre = resultado.names[int(b.cls[0])]
        confianza = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        if nombre not in RECOGER:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            continue

        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        if not (0 <= u < ancho and 0 <= v < alto):
            continue

        region = depth[
            max(0, v - 4):min(alto, v + 5),
            max(0, u - 4):min(ancho, u + 5)
        ]
        region = region[region > 0]

        if region.size == 0:
            print(f"Fresa {nombre} descartada: profundidad invalida.")
            continue

        z = float(np.median(region)) * scale
        p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
        xyz_base = transformar_a_base(p_cam, pose_tcp_foto)

        fresa = {
            "nombre": nombre,
            "xyz": xyz_base.copy(),
            "confianza": confianza,
            "pixel": (u, v),
            "seccion": nombre_seccion
        }
        fresas_detectadas.append(fresa)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        texto = (
            f"{nombre} {confianza:.2f} "
            f"X:{xyz_base[0] * 1000:.0f} "
            f"Y:{xyz_base[1] * 1000:.0f} "
            f"Z:{xyz_base[2] * 1000:.0f} mm"
        )
        cv2.putText(
            frame,
            texto,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2
        )

    fresas_detectadas = ordenar_por_prioridad(fresas_detectadas)
    return fresas_detectadas, frame


# ============================================================
# 11. EJECUTAR RUTINA DE AGARRE
# ============================================================

def ejecutar_rutina_agarre(
    robot,
    receptor,
    xyz_fresa,
    rutina_base,
    fresa_referencia,
    aproximaciones,
    es_primera_de_seccion,
    movimientos_fijos
):
    print("\n================================")
    print("INICIANDO RUTINA DE AGARRE")
    print("================================")

    rutina_actual = generar_rutina_agarre(
        xyz_fresa,
        rutina_base,
        fresa_referencia
    )
    validar_rutina(rutina_actual, "rutina generada")

    print("Puntos calculados:")
    for i, punto in enumerate(rutina_actual):
        print(f"Punto {i + 1}: {punto}")

    TIEMPO_AGARRE = 7.0

    # La aproximacion se repite en la primera fresa de CADA seccion.
    if es_primera_de_seccion:
        print("Ejecutando aproximaciones de entrada de la seccion...")
        for i, pose in enumerate(aproximaciones):
            if i == 2:
                mover_l(robot, receptor, pose)
            else:
                mover_j(robot, receptor, pose)
    else:
        print("Se omiten aproximaciones iniciales: no es la primera fresa de la seccion.")

    # Se conserva el comportamiento del codigo original:
    # punto 1 no se ejecuta; punto 2 MoveJ; puntos 3, 4 y 5 MoveL;
    # punto 6 MoveJ.
    for i, punto in enumerate(rutina_actual):
        if i == 0:
            continue

        print(f"\nEjecutando PUNTO {i + 1}/{len(rutina_actual)}")

        if i == 2:
            mover_l(robot, receptor, punto)
            print(f"Robot en PICK. Esperando {TIEMPO_AGARRE} segundos...")
            time.sleep(TIEMPO_AGARRE)
            print("Tiempo de agarre terminado.")
        elif i in (3, 4):
            mover_l(robot, receptor, punto)
        else:
            mover_j(robot, receptor, punto)

    # =====================================================
    # MOVIMIENTOS FIJOS DESPUES DEL PUNTO 6
    # =====================================================

    # =====================================================
# MOVIMIENTOS FIJOS DE LA SECCION ACTUAL
# =====================================================

    if movimientos_fijos:

        print("\nEjecutando movimientos fijos de la seccion...")

        for i, punto in enumerate(movimientos_fijos):
            print(f"\nEjecutando MOVIMIENTO FIJO {i + 1}/{len(movimientos_fijos)}")
            mover_j(robot, receptor, punto)

        # Regreso en reversa
        print("\nRegresando en reversa...")

        for i, punto in enumerate(reversed(movimientos_fijos)):
            print(f"\nRegresando por MOVIMIENTO FIJO {len(movimientos_fijos) - i}/{len(movimientos_fijos)}")
            mover_j(robot, receptor, punto)

        # Regresar al Punto 6
        print("\nRegresando al PUNTO 6...")
        mover_j(robot, receptor, rutina_actual[5])

    else:
        print("\nEsta seccion no tiene movimientos fijos.")

    print("Rutina de agarre terminada.")
    return True


# ============================================================
# 12. COSECHAR UNA FRESA Y UNA SECCION
# ============================================================

def cosechar_fresa(robot, receptor, fresa, config_seccion, es_primera_de_seccion):
    return ejecutar_rutina_agarre(
        robot=robot,
        receptor=receptor,
        xyz_fresa=fresa["xyz"],
        rutina_base=config_seccion["rutina"],
        fresa_referencia=config_seccion["referencia"],
        aproximaciones=config_seccion["aproximaciones"],
        es_primera_de_seccion=es_primera_de_seccion,
        movimientos_fijos=config_seccion["movimientos_fijos"]
    )


def cosechar_seccion(robot, receptor, nombre_seccion, fresas, config_seccion, usar_aproximaciones=True):
    print("\n============================================")
    print(f"INICIANDO COSECHA DE {nombre_seccion.upper()}")
    print("============================================")

    cosechadas = 0
    fallidas = 0

    for i, fresa in enumerate(fresas):
        print("\n--------------------------------------------")
        print(f"{nombre_seccion}: FRESA {i + 1}/{len(fresas)}")
        print("Clase:", fresa["nombre"])
        print("Pixel:", fresa["pixel"])
        print("XYZ base:", fresa["xyz"])

        try:
            exito = cosechar_fresa(
                robot,
                receptor,
                fresa,
                config_seccion,
                es_primera_de_seccion=(i == 0 and usar_aproximaciones)
            )
            if exito:
                cosechadas += 1
                print("Fresa cosechada.")
            else:
                fallidas += 1
                print("Fresa no cosechada.")

        except Exception as e:
            fallidas += 1
            print(f"Error durante la cosecha: {e}")
            # Por seguridad no continuar con mas movimientos tras un error.
            raise

    print(f"Seccion terminada: {nombre_seccion}")
    print("Cosechadas:", cosechadas)
    print("Fallidas:", fallidas)
    return cosechadas, fallidas


# ============================================================
# 13. VALIDAR CONFIGURACION ANTES DE CONECTAR/MOVER EL ROBOT
# ============================================================

def validar_configuracion():
    validar_pose(HOME, "HOME")
    validar_pose(POS_FOTO_SECCION_1, "POS_FOTO_SECCION_1")
    validar_pose(POS_FOTO_SECCION_2, "POS_FOTO_SECCION_2")

    validar_rutina(RUTINA_SECCION_1, "RUTINA_SECCION_1")
    validar_rutina(RUTINA_SECCION_2, "RUTINA_SECCION_2")
    for i, pose in enumerate(APROX_CHAROLA_1):
        validar_pose(pose, f"APROX_CHAROLA_1[{i}]")
    validar_pose(POS_FOTO_CHAROLA_1, "POS_FOTO_CHAROLA_1")
    validar_pose(POS_FOTO_CHAROLA_2, "POS_FOTO_CHAROLA_2")

    if FRESA_REFERENCIA_SECCION_1 is None or len(FRESA_REFERENCIA_SECCION_1) != 3:
        raise ValueError("Configura FRESA_REFERENCIA_SECCION_1 con [x,y,z].")
    if FRESA_REFERENCIA_SECCION_2 is None or len(FRESA_REFERENCIA_SECCION_2) != 3:
        raise ValueError("Configura FRESA_REFERENCIA_SECCION_2 con [x,y,z].")

    if APROXIMACIONES_SECCION_1 is None or len(APROXIMACIONES_SECCION_1) != 4:
        raise ValueError("APROXIMACIONES_SECCION_1 debe tener 3 poses.")
    if APROXIMACIONES_SECCION_2 is None or len(APROXIMACIONES_SECCION_2) != 3:
        raise ValueError("APROXIMACIONES_SECCION_2 debe tener 3 poses.")

    for i, pose in enumerate(APROXIMACIONES_SECCION_1):
        validar_pose(pose, f"APROXIMACIONES_SECCION_1[{i}]")
    for i, pose in enumerate(APROXIMACIONES_SECCION_2):
        validar_pose(pose, f"APROXIMACIONES_SECCION_2[{i}]")



# ============================================================
# 14. PROGRAMA PRINCIPAL
# ============================================================

def main():
    robot = None
    receptor = None
    pipeline = None

    try:
        # No se conecta ni mueve el robot si faltan poses de seccion 2
        # o posiciones reales de fotografia.
        validar_configuracion()

        print("Cargando modelo YOLO...")
        model = YOLO(MODELO)

        print("Conectando al UR3e...")
        robot = rtde_control.RTDEControlInterface(ROBOT_IP)
        receptor = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        print("Robot conectado.")

        print("Inicializando RealSense...")
        pipeline, align, intr, scale = iniciar_camara()

        # ----------------------------------------------------
        # ETAPA 1: FOTOS DE CHAROLAS Y DE FRESAS
        # ----------------------------------------------------
        capturas = {}

        print("\nETAPA 1: CAPTURA DE CHAROLAS Y SECCIONES")

        # Punto de partida: pose de fotografia de fresas, seccion 1
        print("\nMoviendo a la posicion de fotografia de seccion 1...")
        mover_j(robot, receptor, POS_FOTO_SECCION_1)

        # IDA a charola 1: 2 aproximaciones + pose final de foto
        print("\nIda a charola 1...")
        mover_j(robot, receptor, APROX_CHAROLA_1[0])
        mover_j(robot, receptor, APROX_CHAROLA_1[1])
        mover_j(robot, receptor, POS_FOTO_CHAROLA_1)

        # Foto charola 1
        capturas["charola_1"] = capturar_foto_seccion(
            pipeline, align, receptor, "charola_1"
        )

        # Ir a charola 2
        print("\nMoviendo a la foto de charola 2...")
        mover_j(robot, receptor, POS_FOTO_CHAROLA_2)
        capturas["charola_2"] = capturar_foto_seccion(
            pipeline, align, receptor, "charola_2"
        )

        # REGRESO en reversa, sin fotos
        print("\nRegreso en reversa...")
        mover_l(robot, receptor, POS_FOTO_CHAROLA_1)
        mover_j(robot, receptor, APROX_CHAROLA_1[1])
        mover_j(robot, receptor, APROX_CHAROLA_1[0])

        # De vuelta en la pose de fresas de la seccion 1: sigue tu rutina original
        print("\nRegresando a la posicion de fotografia de seccion 1...")
        mover_j(robot, receptor, POS_FOTO_SECCION_1)
        capturas["seccion_1"] = capturar_foto_seccion(
            pipeline, align, receptor, "seccion_1"
        )

        print("\nMoviendo a la posicion de fotografia de seccion 2...")
        mover_j(robot, receptor, POS_FOTO_SECCION_2)
        capturas["seccion_2"] = capturar_foto_seccion(
            pipeline, align, receptor, "seccion_2"
        )

        # ----------------------------------------------------
        # ETAPA 2: REGRESAR A HOME DESPUES DE LAS DOS FOTOS
        # ----------------------------------------------------
        print("\nETAPA 2: REGRESO A HOME")
        mover_j(robot, receptor, HOME)

        # ----------------------------------------------------
        # ETAPA 3: PROCESAR LAS DOS FOTOGRAFIAS
        # Se usa la pose TCP propia de cada fotografia.
        # ----------------------------------------------------
        print("\nETAPA 3: DETECCION Y ORDENAMIENTO")

        fresas_seccion_1, anotada_1 = detectar_fresas(
            model,
            capturas["seccion_1"]["frame"].copy(),
            capturas["seccion_1"]["depth"],
            intr,
            scale,
            capturas["seccion_1"]["pose_tcp"],
            "seccion_1"
        )

        fresas_seccion_2, anotada_2 = detectar_fresas(
            model,
            capturas["seccion_2"]["frame"].copy(),
            capturas["seccion_2"]["depth"],
            intr,
            scale,
            capturas["seccion_2"]["pose_tcp"],
            "seccion_2"
        )

        print("Fresas recogibles en seccion 1:", len(fresas_seccion_1))
        print("Fresas recogibles en seccion 2:", len(fresas_seccion_2))

        for nombre, lista in (
            ("SECCION 1", fresas_seccion_1),
            ("SECCION 2", fresas_seccion_2)
        ):
            print(f"\nOrden de cosecha: {nombre}")
            for i, fresa in enumerate(lista):
                print(
                    f"{i + 1}. pixel={fresa['pixel']} "
                    f"xyz={np.round(fresa['xyz'], 4)} "
                    f"clase={fresa['nombre']}"
                )

        cv2.imshow("Detecciones - Seccion 1", anotada_1)
        cv2.imshow("Detecciones - Seccion 2", anotada_2)
        cv2.imshow("Charola 1", capturas["charola_1"]["frame"])
        cv2.imshow("Charola 2", capturas["charola_2"]["frame"])
        print("\nRevisa ambas imagenes. Presiona una tecla para continuar o Q para cancelar.")
        tecla = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Detecciones - Seccion 1")
        cv2.destroyWindow("Detecciones - Seccion 2")
        cv2.destroyWindow("Charola 1")
        cv2.destroyWindow("Charola 2")

        if tecla in (ord("q"), ord("Q"), 27):
            print("Cosecha cancelada por el usuario.")
            return

        if len(fresas_seccion_1) == 0 and len(fresas_seccion_2) == 0:
            print("No se encontraron fresas recogibles en ninguna seccion.")
            return

        confirmar = input(
            "\nEscribe SI para iniciar cosecha de seccion 1 y despues seccion 2: "
        )
        if confirmar.strip().upper() != "SI":
            print("Cosecha cancelada por el usuario.")
            return

        # ----------------------------------------------------
        # ETAPA 4: COSECHAR SECCION 1 Y DESPUES SECCION 2
        # ----------------------------------------------------
        configuraciones = {
            "seccion_1": {
                "rutina": RUTINA_SECCION_1,
                "referencia": FRESA_REFERENCIA_SECCION_1,
                "aproximaciones": APROXIMACIONES_SECCION_1,
                "movimientos_fijos": MOVIMIENTOS_FIJOS_SECCION_1,
            },
            "seccion_2": {
                "rutina": RUTINA_SECCION_2,
                "referencia": FRESA_REFERENCIA_SECCION_2,
                "aproximaciones": APROXIMACIONES_SECCION_2,
                "movimientos_fijos": MOVIMIENTOS_FIJOS_SECCION_2,
            },
        }

        resultados = {}
        orden_secciones = [
            ("seccion_1", fresas_seccion_1),
            ("seccion_2", fresas_seccion_2),
        ]

        for nombre_seccion, fresas in orden_secciones:
            if len(fresas) == 0:
                print(f"\n{nombre_seccion}: no hay fresas para cosechar. Se omite.")
                continue

            # No regresar a HOME entre secciones.
            # Al terminar la seccion 1, la rutina de aproximacion de la
            # primera fresa de la seccion 2 realiza la transicion directa.
            resultados[nombre_seccion] = cosechar_seccion(
                robot,
                receptor,
                nombre_seccion,
                fresas,
                configuraciones[nombre_seccion]
            )

        # ----------------------------------------------------
        # ETAPA 5: RESUMEN (SIN REGRESAR A HOME)
        # ----------------------------------------------------
        print("\nCosecha terminada. El robot permanece en su ultima posicion.")

        print("\n============================================")
        print("CICLO COMPLETO DE COSECHA TERMINADO")
        print("============================================")
        print("Detectadas seccion 1:", len(fresas_seccion_1))
        print("Detectadas seccion 2:", len(fresas_seccion_2))

        total_cosechadas = 0
        total_fallidas = 0
        for nombre, (cosechadas, fallidas) in resultados.items():
            print(f"{nombre}: cosechadas={cosechadas}, fallidas={fallidas}")
            total_cosechadas += cosechadas
            total_fallidas += fallidas

        print("TOTAL cosechadas:", total_cosechadas)
        print("TOTAL fallidas:", total_fallidas)

    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")

    except Exception as e:
        print("\nERROR:", e)
        print("Por seguridad, el programa no continuara con mas movimientos.")

    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()

        if receptor is not None:
            try:
                receptor.disconnect()
            except Exception:
                pass

        if robot is not None:
            try:
                robot.stopScript()
            except Exception:
                pass

        print("Sistema detenido.")


if __name__ == "__main__":
    main()
