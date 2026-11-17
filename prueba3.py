import json
import paho.mqtt.client as mqtt
import numpy as np
import cv2
import pyrealsense2 as rs
import time
import os
from datetime import datetime

MQTT_BROKER = "localhost"
TOPIC_INPUT = "enviar_rasp"
TOPIC_PLC_TRIGGER = "plc/start_ur3e"

TOPIC_VERDES = "verdes" 

QR_ID = "https://qr.link/0jEqz6"
QR_SIZE_CM = 7.50
WIDTH, HEIGHT, FPS = 640, 480, 30
USAR_PROFUNDIDAD = True

FILAS_MATRIZ = 4
COLUMNAS_MATRIZ = 6

Y_INICIO_FRESAS = int(HEIGHT * 0.25) 
AREA_MINIMA_BLOB = 300 
TIEMPO_ESTABILIZACION = 0.5 

CARPETA_FOTOS = os.path.expanduser("~/fotos_realsense")
os.makedirs(CARPETA_FOTOS, exist_ok=True)

qr_detector = cv2.QRCodeDetector()

def split_matrix_by_size(matrix_str):
    arr = np.array(matrix_str, dtype=object)
    big_matrix   = np.where(np.char.lower(arr.astype(str)) == "grande", 1, 0)
    small_matrix = np.where(np.char.lower(arr.astype(str)) == "chica", 1, 0)
    return big_matrix, small_matrix

def distancia_del_blob(depth_m, contorno):
    if depth_m is None: return None
    mask = np.zeros(depth_m.shape, np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    valores = depth_m[(mask > 0) & (depth_m > 0)]
    return float(np.median(valores)) if valores.size >= 10 else None

def dibujar_cuadricula(imagen):
    alto_zona_fresas = HEIGHT - Y_INICIO_FRESAS
    ancho_celda = WIDTH / COLUMNAS_MATRIZ
    alto_celda = alto_zona_fresas / FILAS_MATRIZ

    for i in range(1, COLUMNAS_MATRIZ):
        x = int(i * ancho_celda)
        cv2.line(imagen, (x, Y_INICIO_FRESAS), (x, HEIGHT), (255, 255, 0), 1)

    for i in range(1, FILAS_MATRIZ):
        y = int(Y_INICIO_FRESAS + i * alto_celda)
        cv2.line(imagen, (0, y), (WIDTH, y), (255, 255, 0), 1)
        
    cv2.rectangle(imagen, (0, Y_INICIO_FRESAS), (WIDTH-1, HEIGHT-1), (255, 255, 0), 2)

    for r in range(FILAS_MATRIZ):
        for c in range(COLUMNAS_MATRIZ):
            num_celda = (r * COLUMNAS_MATRIZ) + c + 1
            cx = int((c + 0.5) * ancho_celda)
            cy = int(Y_INICIO_FRESAS + (r + 0.5) * alto_celda)
            cv2.putText(imagen, str(num_celda), (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def guardar_matrices_json(matriz_chicas_detectada, matriz_grandes_detectada):
    try:
        dict_matrices = {
            "detectada": {
                "chicas": matriz_chicas_detectada.tolist(),
                "grandes": matriz_grandes_detectada.tolist()
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        ruta_json = os.path.join(CARPETA_FOTOS, "ultimas_matrices.json")
        with open(ruta_json, 'w') as f:
            json.dump(dict_matrices, f, indent=4)
            
        print(f"[INFO] Matrices detectadas guardadas en: {ruta_json}")
        
    except Exception as e:
        print(f"[ERROR] No se pudieron guardar las matrices en JSON: {e}")

def escanear_fresas(client):
    print("\n[CAMARA] Encendiendo RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    if USAR_PROFUNDIDAD:
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    perfil = pipeline.start(config)
    align = rs.align(rs.stream.color) if USAR_PROFUNDIDAD else None
    intr = perfil.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    FX = intr.fx
    depth_scale = perfil.get_device().first_depth_sensor().get_depth_scale() if USAR_PROFUNDIDAD else None

    for _ in range(30): pipeline.wait_for_frames()

    tiempo_inicio_deteccion = None
    matriz_detectada = None
    contador_verdes = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align: frames = align.process(frames)
            
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            frame = np.asanyarray(color_frame.get_data())
            depth_m = None
            if USAR_PROFUNDIDAD:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_m = (np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale)

            resultado = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            dibujar_cuadricula(resultado)

            data, points, _ = qr_detector.detectAndDecode(frame)
            qr_valido = False

            if points is not None and data == QR_ID:
                puntos_qr = points[0].astype(np.float32)
                centro_qr = np.mean(puntos_qr, axis=0)
                qx, qy = centro_qr.astype(int)
                
                x, y, w, h = cv2.boundingRect(puntos_qr.astype(int))
                cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(resultado, "QR (Fuera de area)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if qy < Y_INICIO_FRESAS + 50: 
                    qr_valido = True
                    if tiempo_inicio_deteccion is None:
                        tiempo_inicio_deteccion = time.time()
                    
                    tiempo_restante = TIEMPO_ESTABILIZACION - (time.time() - tiempo_inicio_deteccion)
                    
                    if tiempo_restante > 0:
                        cv2.putText(resultado, f"ESTABILIZANDO {tiempo_restante:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(resultado, "MAPEANDO...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    tiempo_inicio_deteccion = None
                    cv2.putText(resultado, "MUEVE EL QR ARRIBA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                tiempo_inicio_deteccion = None
                cv2.putText(resultado, "BUSCANDO QR...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if qr_valido and tiempo_inicio_deteccion is not None and (time.time() - tiempo_inicio_deteccion >= TIEMPO_ESTABILIZACION):
                print("[CAMARA] QR válido. Escaneando matriz 4x6...")
                
                matriz_detectada = [["vacio" for _ in range(COLUMNAS_MATRIZ)] for _ in range(FILAS_MATRIZ)]
                contador_verdes = 0 

                alto_zona_fresas = HEIGHT - Y_INICIO_FRESAS
                ancho_celda = WIDTH / COLUMNAS_MATRIZ
                alto_celda = alto_zona_fresas / FILAS_MATRIZ

                # espacios vacíos
                mask_blanco = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))
                contornos_blancos, _ = cv2.findContours(mask_blanco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contorno in contornos_blancos:
                    area = cv2.contourArea(contorno)
                    if area < 15 or area > 200: continue 

                    (x, y), radius = cv2.minEnclosingCircle(contorno)
                    if y < Y_INICIO_FRESAS: continue
                    
                    col_idx = max(0, min(int(x / ancho_celda), COLUMNAS_MATRIZ - 1))
                    row_idx = max(0, min(int((y - Y_INICIO_FRESAS) / alto_celda), FILAS_MATRIZ - 1))
                    
                    cx_celda = (col_idx + 0.5) * ancho_celda
                    cy_celda = Y_INICIO_FRESAS + (row_idx + 0.5) * alto_celda
                    
                    if abs(x - cx_celda) < ancho_celda/3 and abs(y - cy_celda) < alto_celda/3:
                        cv2.circle(resultado, (int(x), int(y)), 5, (255, 255, 255), -1)

                # fresas rojas
                lower_red1 = np.array([0, 100, 80]); upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 100, 80]); upper_red2 = np.array([180, 255, 255])
                mask_roja = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
                
                kernel = np.ones((3, 3), np.uint8) 
                mask_roja = cv2.morphologyEx(mask_roja, cv2.MORPH_OPEN, kernel)
                mask_roja = cv2.morphologyEx(mask_roja, cv2.MORPH_CLOSE, kernel)

                contornos_rojos, _ = cv2.findContours(mask_roja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contorno in contornos_rojos:
                    area = cv2.contourArea(contorno)
                    if area < AREA_MINIMA_BLOB: continue

                    (x, y), radius = cv2.minEnclosingCircle(contorno)
                    if radius <= 0 or y < Y_INICIO_FRESAS: continue

                    z = distancia_del_blob(depth_m, contorno)
                    diametro_cm = (2 * radius) * z / FX * 100.0 if z else (2 * radius) * (QR_SIZE_CM / 100.0)

                    categoria = "vacio"
                    color = (255, 255, 255)
                    
                    if diametro_cm >= 4.0:
                        categoria = "grande"
                        color = (0, 255, 0)
                    elif diametro_cm <= 3.5: 
                        categoria = "chica"
                        color = (0, 0, 255) 

                    if categoria != "vacio":
                        cv2.drawContours(resultado, [contorno], -1, color, 2)
                        
                        col_idx = max(0, min(int(x / ancho_celda), COLUMNAS_MATRIZ - 1))
                        row_idx = max(0, min(int((y - Y_INICIO_FRESAS) / alto_celda), FILAS_MATRIZ - 1))
                        matriz_detectada[row_idx][col_idx] = categoria
                        
                        texto = f"#{categoria}"
                        cv2.putText(resultado, texto, (int(x)-30, int(y)-int(radius)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # fresas verdes
                mask_verde = cv2.inRange(hsv, np.array([40, 100, 50]), np.array([85, 255, 255]))
                mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_OPEN, kernel)
                mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_CLOSE, kernel)
                contornos_verdes, _ = cv2.findContours(mask_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contorno in contornos_verdes:
                    area = cv2.contourArea(contorno)
                    if area < AREA_MINIMA_BLOB: continue

                    (x, y), radius = cv2.minEnclosingCircle(contorno)
                    if radius <= 0 or y < Y_INICIO_FRESAS: continue

                    cv2.drawContours(resultado, [contorno], -1, (0, 255, 255), 2)
                    contador_verdes += 1 

                payload_hmi = {"fresas_verdes": contador_verdes}
                client.publish(TOPIC_VERDES, json.dumps(payload_hmi))
                print(f"[MQTT] Conteo de verdes ({contador_verdes}) enviado al dashboard.")

                marca = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(CARPETA_FOTOS, f"mapeo_{marca}.png"), resultado)
                cv2.imwrite(os.path.join(CARPETA_FOTOS, "ultima_foto.png"), resultado)
                
                print("[CAMARA] Matriz detectada:")
                for fila in matriz_detectada:
                    print("   ", fila)
                    
                break 

            cv2.imshow("Sistema integrado de vision", resultado)
            if cv2.waitKey(1) & 0xFF == 27: 
                print("[CAMARA] Escaneo cancelado.")
                return None, 0

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return matriz_detectada, contador_verdes

# mqtt
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        received_matrix = data.get("matrix", []) if isinstance(data, dict) else data

        if not received_matrix: return

        print("\n[INFO] Matriz recibida de Node-RED:")
        
        for fila in received_matrix:
            print("   ", fila)
            
        big_target, small_target = split_matrix_by_size(received_matrix)
        current_matrix, total_verdes = escanear_fresas(client)
        if current_matrix is None: return

        big_current, small_current = split_matrix_by_size(current_matrix)

        print("[INFO] Guardando matrices detectadas en JSON...")
        guardar_matrices_json(small_current, big_current)

        big_ok   = np.array_equal(big_target,   big_current)
        small_ok = np.array_equal(small_target, small_current)
        todo_ok  = big_ok and small_ok
        
        

        if todo_ok and total_verdes == 0:
            print("[ÉXITO] Validación visual correcta.")
            status = "APPROVED"
        else:
            print("[ADVERTENCIA] Diferencias encontradas o fresas verdes presentes. Movimiento abortado.")
            status = "REJECTED"

        payload_chicas = {
            "status": status, 
            "lote": "CHICAS", 
            "matrix": small_current.tolist() 
        }
        client.publish(TOPIC_PLC_TRIGGER, json.dumps(payload_chicas))
        print(f"[MQTT] Lote CHICAS enviado (Estado: {status}).")
        time.sleep(0.5)
        
        payload_grandes = {
            "status": status, 
            "lote": "GRANDES", 
            "matrix": big_current.tolist() 
        }
        client.publish(TOPIC_PLC_TRIGGER, json.dumps(payload_grandes))
        print(f"[MQTT] Lote GRANDES enviado (Estado: {status}).")
        time.sleep(0.5)
        
        client.publish(TOPIC_PLC_TRIGGER, json.dumps({"status": "FIN_SECUENCIA"}))
        print(f"[MQTT] FIN DE SECUENCIA enviado.")

    except Exception as e:
        print(f"[ERROR] Falló el procesamiento: {e}")

if __name__ == '__main__':
    client = mqtt.Client()
    client.on_message = on_message

    client.connect(MQTT_BROKER, 1883, 60)
    client.subscribe(TOPIC_INPUT)

    print("Raspberry lista. Esperando matriz en 'enviar_rasp'...")
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("Apagando sistema...")