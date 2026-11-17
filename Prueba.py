import time
import cv2
import numpy as np
import rtde_control
import rtde_receive
import pyrealsense2 as rs
# import struct  # MiR/Modbus: temporalmente desactivado

from ultralytics import YOLO
from scipy.spatial.transform import Rotation
# from pymodbus.client import ModbusTcpClient  # MiR/Modbus: temporalmente desactivado


# ============================================================
# 1. CONFIGURACION GENERAL
# ============================================================

AJUSTE_Z_PICK = 0.020

# MIR_IP = "192.168.0.20"  # MiR/Modbus: temporalmente desactivado
# PORT = 502                  # MiR/Modbus: temporalmente desactivado
ROBOT_IP = "192.168.0.10"
MODELO = "runs/detect/fresas_3clases/weights/best.pt"
CONFIDENCE = 0.5
RECOGER = {"grande_roja", "chica_roja"}

ANTIGUEDAD_MAXIMA = 0.5


# ============================================================
# 2. VELOCIDADES DEL ROBOT
# ============================================================

VELOCIDAD_J = 0.10
ACELERACION_J = 0.15

VELOCIDAD_L = 0.03
ACELERACION_L = 0.05


# ============================================================
# 3. PARAMETROS DE ESPERA DE MOVIMIENTO
# ============================================================

TIMEOUT_MOVIMIENTO = 90.0
VELOCIDAD_PARADO = 0.005
TIEMPO_ESTABLE = 0.5

TOLERANCIA_POSICION = 0.003
TOLERANCIA_ORIENTACION = 0.05


# ============================================================
# 4. POSICION HOME
# ============================================================

HOME = [
    0.1056, -0.0349, 0.4967,
    1.46, -0.730, -0.723
]

POS_PREAPROXIMACION1 = [
    -0.05696, -0.22259, 0.19395, 2.912, -1.179, 0.000   # <-- reemplaza con tu punto de Polyscope
]

POS_PREAPROXIMACION2 = [
    -0.05696, -0.22182, 0.61777, 0.001, 0.003, 2.199  # <-- reemplaza con tu punto de Polyscope
]

POS_PREAPROXIMACION3 = [
    -0.05696, -0.22182, 0.61777, 0.018, -0.005, 5.427  # <-- reemplaza con tu punto de Polyscope
]


# ============================================================
# 5. MATRIZ DE TRANSFORMACION CAMARA -> TCP
# ============================================================

T_TCP_CAM = np.array([
    [-0.999500,  0.028256,  0.014210,  0.020598],
    [-0.029022, -0.997956, -0.056933,  0.080521],
    [ 0.012572, -0.057317,  0.998277, -0.113376],
    [ 0.000000,  0.000000,  0.000000,  1.000000]
])


# ============================================================
# 6. FRESA DE REFERENCIA
# ============================================================
# Coordenadas de la fresa para la que se programo originalmente
# la rutina en Polyscope.
#
# No es necesario que esta fresa este presente en el huerto.
# Este valor se conserva como referencia fija.

FRESA_REFERENCIA = np.array([
    -0.22696,
    -0.26570,
     0.522
], dtype=float)


# ============================================================
# 7. RUTINA ORIGINAL DE AGARRE
# ============================================================
# Estos son los puntos originales que ya probaste.
#
# Cada punto:
# [X, Y, Z, Rx, Ry, Rz]
#
# XYZ en metros.
# Orientaciones en radianes.
#
# IMPORTANTE:
# No modificar estos puntos para cada fresa.
# El programa genera automaticamente una copia desplazada.

RUTINA_AGARRE = [

    # PUNTO 1: ROTAR CAMARA
    [-0.23699, -0.272,   0.530,   1.278,  2.529, -2.498],

    # PUNTO 2: Prepick
    #[-0.2564,  -0.27771, 0.52529, 0.130, -0.001, -1.128],
    [-0.25508,  -0.27568, 0.45, 0.130, -0.001, -1.128],

    # PUNTO 3: Pick
    #[-0.25906, -0.28184, 0.56488, 0.130, -0.001, -1.128],
   # [-0.25509, -0.27567, 0.54105, 0.130, -0.001, -1.128],
    [-0.25508,  -0.27568, 0.51, 0.130, -0.001, -1.128],
    # PUNTO 4: Vuelve al pre pick
    [-0.25508,  -0.27568, 0.45, 0.130, -0.001, -1.128],

    # PUNTO 5: RETIRADA
    #[-0.23699, -0.272,   0.530,   1.278,  2.529, -2.498],
    [-0.17696, -0.23913, 0.45, 0.130, -0.001, -1.128],

    # PUNTO 6: ORIENTACION DE REGRESO
    [-0.27905,   0.12496,   0.48713,   0.127, -0.093, -2.366],
]


# MiR / MODBUS DESACTIVADO TEMPORALMENTE
# def int_a_registros_modbus(valor):
#     """
#     Convierte un entero de 32 bits en dos registros
#     de 16 bits, en formato Big Endian.
#     """
#     packed = struct.pack('>I', valor)
#     word1, word2 = struct.unpack('>HH', packed)
#     return [word1, word2]
#
#
# def escribir_registro(client, direccion, valor):
#
#     if client is None or not client.connected:
#         print("ERROR: No hay conexión con el MiR.")
#         return False
#
#     try:
#         payload = int_a_registros_modbus(valor)
#
#         resultado = client.write_registers(
#             address=direccion,
#             values=payload
#         )
#
#         if resultado is None or resultado.isError():
#             print(
#                 f"ERROR al escribir en Modbus "
#                 f"(dirección {direccion}): {resultado}"
#             )
#             return False
#
#         print(
#             f"Registro Modbus enviado correctamente. "
#             f"Dirección: {direccion}, valor: {valor}"
#         )
#
#         return True
#
#     except Exception as e:
#         print(f"ERROR de comunicación Modbus: {e}")
#         return False
#

# ============================================================
# 8. TRANSFORMACIONES
# ============================================================

def pose_a_matriz(pose):

    T = np.eye(4)

    T[:3, 3] = pose[:3]

    T[:3, :3] = Rotation.from_rotvec(
        pose[3:6]
    ).as_matrix()

    return T


def transformar_a_base(p_cam, pose_tcp):

    P_CAM = np.array([
        p_cam[0],
        p_cam[1],
        p_cam[2],
        1.0
    ])

    # Camara -> TCP
    P_TCP = T_TCP_CAM @ P_CAM

    # TCP -> Base
    T_BASE_TCP = pose_a_matriz(pose_tcp)

    P_BASE = T_BASE_TCP @ P_TCP

    return P_BASE[:3]


# ============================================================
# 9. GENERAR RUTINA PARA CADA FRESA
# ============================================================

def generar_rutina_agarre(xyz_fresa):

    xyz_fresa = np.array(xyz_fresa, dtype=float)

    # Desplazamiento de la nueva fresa respecto a la referencia
    desplazamiento = xyz_fresa - FRESA_REFERENCIA

    print("\n================================")
    print("CALCULO DE DESPLAZAMIENTO")
    print("================================")

    print("Fresa de referencia:", FRESA_REFERENCIA)
    print("Nueva fresa:", xyz_fresa)

    print("\nDesplazamiento:")
    print(f"X: {desplazamiento[0]:.4f} m")
    print(f"Y: {desplazamiento[1]:.4f} m")
    print(f"Z: {desplazamiento[2]:.4f} m")

    rutina_nueva = []

    for i, punto in enumerate(RUTINA_AGARRE):

        nuevo_punto = np.array(punto, dtype=float).copy()

        # Desplazar XYZ solo en los puntos 1 a 5.
        # El punto 6 (regreso) queda fijo.
        if i < 5:
            nuevo_punto[:3] += desplazamiento

        rutina_nueva.append(nuevo_punto.tolist())

    return rutina_nueva


# ============================================================
# 10. ESPERA DE MOVIMIENTO
# ============================================================

def error_orientacion(actual, objetivo):

    R_actual = Rotation.from_rotvec(actual[3:6])
    R_objetivo = Rotation.from_rotvec(objetivo[3:6])

    return (
        R_objetivo * R_actual.inv()
    ).magnitude()


def esperar_movimiento(
    receptor,
    objetivo,
    timeout=TIMEOUT_MOVIMIENTO
):

    inicio = time.time()
    inicio_estable = None

    while time.time() - inicio < timeout:

        pose_actual = receptor.getActualTCPPose()

        velocidad_tcp = receptor.getActualTCPSpeed()

        error_pos = np.linalg.norm(
            np.array(pose_actual[:3])
            - np.array(objetivo[:3])
        )

        error_rot = error_orientacion(
            pose_actual,
            objetivo
        )

        rapidez = np.linalg.norm(
            np.array(velocidad_tcp)
        )

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

    print("\nERROR: Tiempo de espera agotado.")
    print("Error de posicion:", error_pos, "m")
    print("Error de orientacion:", error_rot, "rad")
    print("Rapidez TCP:", rapidez)

    return False


# ============================================================
# 11. FUNCIONES DE MOVIMIENTO
# ============================================================

def mover_j(robot, receptor, posicion):

    print("\nMoveJ cartesiano:", posicion)

    resultado = robot.moveJ_IK(
        posicion,
        VELOCIDAD_J,
        ACELERACION_J
    )

    if not resultado:
        raise RuntimeError(
            "RTDE no pudo iniciar el movimiento MoveJ_IK."
        )

    if not esperar_movimiento(receptor, posicion):
        raise RuntimeError(
            "El robot no llego al objetivo MoveJ_IK."
        )

    return True


def mover_l(robot, receptor, posicion):

    print("\nMoveL cartesiano:", posicion)

    resultado = robot.moveL(
        posicion,
        VELOCIDAD_L,
        ACELERACION_L
    )

    if not resultado:
        raise RuntimeError(
            "RTDE no pudo iniciar el movimiento MoveL."
        )

    if not esperar_movimiento(receptor, posicion):
        raise RuntimeError(
            "El robot no llego al objetivo MoveL."
        )

    return True


# ============================================================
# 12. VALIDAR RUTINA
# ============================================================

def validar_rutina(rutina):

    if not rutina:
        print("ERROR: La rutina esta vacia.")
        return False

    for i, punto in enumerate(rutina):

        if len(punto) != 6:
            print(
                f"ERROR: El punto {i+1} no tiene 6 valores."
            )
            return False

        if not np.all(
            np.isfinite(np.array(punto, dtype=float))
        ):
            print(
                f"ERROR: El punto {i+1} contiene valores invalidos."
            )
            return False

    return True


# ============================================================
# 13. EJECUTAR RUTINA DE AGARRE
# ============================================================

def ejecutar_rutina_agarre(
    robot,
    receptor,
    xyz_fresa,
    es_primera=True
):

    print("\n================================")
    print("INICIANDO RUTINA DE AGARRE")
    print("================================")

    # Generar puntos desplazados para esta fresa
    rutina_actual = generar_rutina_agarre(xyz_fresa)

    if not validar_rutina(rutina_actual):
        return False

    print("\nPUNTOS QUE SE VAN A EJECUTAR:")

    for i, punto in enumerate(rutina_actual):
        print(f"Punto {i+1}: {punto}")

    TIEMPO_AGARRE = 7.0  # segundos

    # Aproximacion fija: solo para la primera fresa
    if es_primera:
        mover_j(robot, receptor, POS_PREAPROXIMACION1)
        mover_j(robot, receptor, POS_PREAPROXIMACION2)
        mover_l(robot, receptor, POS_PREAPROXIMACION3)
    else:
        print("\nSe omiten las aproximaciones (no es la primera fresa).")

    # Ejecutar los puntos en orden
    for i, punto in enumerate(rutina_actual):
        if i == 0: continue
        print(f"\nEjecutando PUNTO {i+1}/{len(rutina_actual)}")

        if i == 2:
            mover_l(robot, receptor, punto)
            print(f"\nRobot en PICK. Esperando {TIEMPO_AGARRE} segundos")
            time.sleep(TIEMPO_AGARRE)
            print("Tiempo de agarre terminado.")
        elif i == 3 or i == 4:
            mover_l(robot, receptor, punto)
        else:
            mover_j(robot, receptor, punto)

    print("\nRutina de agarre terminada.")
    return True


# ============================================================
# 14. SECUENCIA COMPLETA DE COSECHA
# ============================================================

def cosechar(robot, receptor, xyz_base, es_primera=True):

    print("\n================================")
    print("INICIANDO SECUENCIA DE COSECHA")
    print("================================")

    # Posicion y orientacion actual del TCP
    pose_actual = receptor.getActualTCPPose()

    orientacion = pose_actual[3:6]

    x, y, z = xyz_base

    # Aproximacion a la fresa sin sumar altura
    PRE_FRESA = [
        float(x),
        float(y),
        float(z),
        float(orientacion[0]),
        float(orientacion[1]),
        float(orientacion[2])
    ]

    print("\nXYZ FRESA:", xyz_base)
    print("PRE_FRESA:", PRE_FRESA)

    # Verificar solucion IK de PRE_FRESA
    q_actual = receptor.getActualQ()

    if not robot.getInverseKinematicsHasSolution(
        PRE_FRESA,
        q_actual
    ):

        print("ERROR: PRE_FRESA no tiene solucion IK.")
        return False

    # PASO 1: HOME
    print("\nPASO 1: HOME")

    #mover_j(
        #robot,
        #receptor,
        #HOME
    #)

    # PASO 2: POSICION DE SEGURIDAD

    #print("\nPASO 2: Posicion de seguridad")

    #mover_j(
        #robot,
        #receptor,
        #POS_SEGURIDAD
    #)

    # PASO 2: POSICION DE SEGURIDAD
    
    print("\nPASO 3: ROTACION")
    
    #mover_l(
        #robot,
        #receptor,
        #POS_ROTACION
    #)

    # PASO 2: HOME -> PRE_FRESA mediante MOVEJ
    print("\nPASO 2: APROXIMACION A LA FRESA CON MOVEJ")

    #mover_j(
        #robot,
        #receptor,
        #PRE_FRESA
    #)

    print("\nROBOT EN PRE_FRESA")

    # PASO 3: RUTINA DINAMICA
    print("\nPASO 3: RUTINA PERSONALIZADA")

    exito = ejecutar_rutina_agarre(
        robot,
        receptor,
        xyz_base,
        es_primera
    )

    if not exito:

        print("No se pudo completar la rutina.")
        return False

    # PASO 4: REGRESAR A HOME
    print("\nPASO 4: REGRESO A HOME")

    #mover_j(
        #robot,
        #receptor,
        #HOME
    #)

    print("\\nSECUENCIA TERMINADA.")

    # --------------------------------------------------------
    # MiR / MODBUS DESACTIVADO TEMPORALMENTE
    # Cuando se vuelva a utilizar el MiR, descomentar el envío:
    #
    # print("\\nPASO 5: ENVIAR REGISTRO AL MIR")
    # registro_enviado = escribir_registro(
    #     client,
    #     1000,
    #     1
    # )
    #
    # if registro_enviado:
    #     print("Cosecha finalizada y registro enviado al MiR.")
    # else:
    #     print("Cosecha finalizada pero no se pudo enviar el registro")
    # --------------------------------------------------------

    return True


# ============================================================
# 15. INICIALIZAR REALSENSE
# ============================================================

def iniciar_camara():

    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30
    )

    config.enable_stream(
        rs.stream.depth,
        640,
        480,
        rs.format.z16,
        30
    )

    print("Iniciando streams de RealSense...")

    profile = pipeline.start(config)

    device = profile.get_device()

    print("Camara conectada:", device.get_info(rs.camera_info.name))
    print("Numero de serie:", device.get_info(rs.camera_info.serial_number))

    align = rs.align(rs.stream.color)

    intr = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile().get_intrinsics()

    scale = (
        device.first_depth_sensor().get_depth_scale()
    )

    print("RealSense inicializada correctamente.")

    return pipeline, align, intr, scale

# ============================================================
# 16. DETECTAR TODAS LAS FRESAS DE UNA FOTO
# ============================================================

def detectar_fresas(
    model,
    frame,
    depth,
    intr,
    scale,
    pose_tcp_foto
):
    """
    Procesa una sola imagen.
    Devuelve una lista con todas las fresas recogibles
    y sus coordenadas XYZ en la base del robot.
    """

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

        x1, y1, x2, y2 = map(
            int,
            b.xyxy[0].tolist()
        )

        # Solo procesar fresas maduras recogibles
        if nombre not in RECOGER:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            continue

        # Centro del bounding box
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # Evitar centros fuera de la imagen
        if not (0 <= u < ancho and 0 <= v < alto):
            continue

        # Región de profundidad 9x9
        region = depth[
            max(0, v - 4):min(alto, v + 5),
            max(0, u - 4):min(ancho, u + 5)
        ]

        region = region[region > 0]

        if region.size == 0:
            print(
                f"Fresa {nombre} descartada: "
                "no tiene profundidad válida."
            )
            continue

        # Profundidad mediana en metros
        z = float(np.median(region)) * scale

        # Pixel -> coordenadas de cámara
        p_cam = rs.rs2_deproject_pixel_to_point(
            intr,
            [u, v],
            z
        )

        # Cámara -> base del robot
        # IMPORTANTE: usar siempre la pose de la foto inicial
        xyz_base = transformar_a_base(
            p_cam,
            pose_tcp_foto
        )

        fresa = {
            "nombre": nombre,
            "xyz": xyz_base.copy(),
            "confianza": confianza,
            "pixel": (u, v)
        }

        fresas_detectadas.append(fresa)

        # Dibujar detección válida
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

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

    # Ordenar de izquierda a derecha en la imagen
    fresas_detectadas.sort(
        key=lambda f: f["pixel"][0]
    )

    return fresas_detectadas, frame


# ============================================================
# 16. PROGRAMA PRINCIPAL
# ============================================================

# ============================================================
# 17. PROGRAMA PRINCIPAL
# ============================================================

def main():

    robot = None
    receptor = None
    pipeline = None

    try:

        print("Cargando modelo YOLO...")

        model = YOLO(MODELO)

        print("Conectando al UR3e...")

        robot = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        receptor = rtde_receive.RTDEReceiveInterface(
            ROBOT_IP
        )

        print("Robot conectado.")

        print("Inicializando RealSense...")

        pipeline, align, intr, scale = iniciar_camara()

        print("\n================================")
        print("SISTEMA DE COSECHA INICIADO")
        print("================================")
# PASO 1: VISUALIZAR CAMARA Y ESPERAR FOTO INICIAL
# ----------------------------------------------------
        print("\nIniciando visualizacion de RealSense...")
        cv2.namedWindow(
            "UR3e - Camara RealSense",
            cv2.WINDOW_NORMAL)
        
        intentos_error = 0
        MAX_INTENTOS = 5
        
        while True:
            try:

                # Esperar imagen, con timeout de 10 segundos
                frames = pipeline.wait_for_frames(10000)

                frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    print("Frame incompleto. Intentando nuevamente...")
                    continue

                # Obtener imagen RGB y profundidad
                frame = np.asanyarray(
                    color_frame.get_data()
                )

                depth = np.asanyarray(
                    depth_frame.get_data()
                )

                # Mostrar video en vivo
                vista = frame.copy()

                cv2.putText(
                    vista,
                    "PRESIONA ESPACIO PARA TOMAR FOTO",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    vista,
                    "Q = SALIR",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "UR3e - Camara RealSense",
                    vista
                )

                tecla = cv2.waitKey(1) & 0xFF

                # Tomar foto inicial
                if tecla == 32:  # ESPACIO

                    print("\nFoto inicial capturada.")

                    break

                # Salir
                elif tecla in (ord('q'), ord('Q'), 27):

                    raise KeyboardInterrupt

                intentos_error = 0

            except RuntimeError as e:

                intentos_error += 1

                print(
                    f"Error obteniendo imagen "
                    f"({intentos_error}/{MAX_INTENTOS}): {e}"
                )

                if intentos_error >= MAX_INTENTOS:

                    raise RuntimeError(
                        "RealSense no entrega imágenes. "
                        "Revisa la conexión USB y la configuración."
                    )

        # ----------------------------------------------------
        # PASO 2: GUARDAR POSE TCP DE LA FOTO
        # ----------------------------------------------------

        pose_tcp_foto = receptor.getActualTCPPose()

        print("\nPose TCP de la foto:")
        print(pose_tcp_foto)

        print("\nFoto lista para procesar.")

                # ----------------------------------------------------
                # PASO 3: DETECTAR TODAS LAS FRESAS
                # ----------------------------------------------------

        fresas_pendientes, frame_anotado = detectar_fresas(
            model,
            frame,
            depth,
            intr,
            scale,
            pose_tcp_foto
        )

        print("\n================================")
        print("RESULTADO DE LA FOTO INICIAL")
        print("================================")

        print(
            f"Fresas recogibles detectadas: "
            f"{len(fresas_pendientes)}"
        )

        for i, fresa in enumerate(fresas_pendientes):

            print(f"\nFRESA {i + 1}")

            print("Clase:", fresa["nombre"])

            print(
                "Confianza:",
                f'{fresa["confianza"]:.3f}'
            )

            print(
                "XYZ base:",
                fresa["xyz"]
            )

        # Mostrar la foto con las detecciones
        cv2.imshow(
            "UR3e - Foto inicial de fresas",
            frame_anotado
        )

        cv2.waitKey(1000)

        # ----------------------------------------------------
        # PASO 4: VALIDAR QUE EXISTAN OBJETIVOS
        # ----------------------------------------------------

        if len(fresas_pendientes) == 0:

            print(
                "\nNo se encontraron fresas recogibles."
            )

            return

        # ----------------------------------------------------
        # PASO 5: CONFIRMACION GENERAL DE SEGURIDAD
        # ----------------------------------------------------

        print("\n================================")
        print("LISTA DE OBJETIVOS")
        print("================================")

        print(
            f"El robot intentara cosechar "
            f"{len(fresas_pendientes)} fresas."
        )

        confirmar = input(
            "\nEscribe SI para iniciar la cosecha: "
        )

        if confirmar != "SI":

            print("Cosecha cancelada por el usuario.")

            return

        # ----------------------------------------------------
        # PASO 6: COSECHAR TODAS LAS FRESAS DE LA LISTA
        # ----------------------------------------------------

        total = len(fresas_pendientes)

        cosechadas = 0
        fallidas = 0

        for i, fresa in enumerate(fresas_pendientes):

            print("\n================================")
            print(
                f"COSECHANDO FRESA {i + 1}/{total}"
            )
            print("================================")

            print("Clase:", fresa["nombre"])

            print("XYZ base:", fresa["xyz"])

            try:

                exito = cosechar(
                    robot,
                    receptor,
                    fresa["xyz"],
                    es_primera=(i == 0)
                )

                if exito:

                    cosechadas += 1

                    print(
                        f"Fresa {i + 1} cosechada."
                    )

                else:

                    fallidas += 1

                    print(
                        f"Fresa {i + 1} no pudo cosecharse."
                    )

            except Exception as e:

                fallidas += 1

                print(
                    f"Error al cosechar la fresa "
                    f"{i + 1}: {e}"
                )

                # No continuar automáticamente después
                # de un error de movimiento del robot.
                raise

        # ----------------------------------------------------
        # PASO 7: RESUMEN FINAL
        # ----------------------------------------------------

        print("\n================================")
        print("CICLO DE COSECHA TERMINADO")
        print("================================")

        print("Total detectadas:", total)
        print("Cosechadas:", cosechadas)
        print("Fallidas:", fallidas)

        cv2.waitKey(0)

    except KeyboardInterrupt:

        print("\nPrograma interrumpido.")

    except Exception as e:

        print("\nERROR:", e)

    finally:

        if pipeline is not None:

            pipeline.stop()

        cv2.destroyAllWindows()

        if receptor is not None:

            receptor.disconnect()

        if robot is not None:

            try:

                robot.stopScript()

            except Exception:

                pass

        print("Sistema detenido.")


if __name__ == "__main__":
    main()