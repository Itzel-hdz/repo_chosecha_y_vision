from pymodbus.client import ModbusTcpClient
import struct

# Configura la IP del MiR en tu red
MIR_IP = "192.168.0.20" 
PORT = 502

def int_a_registros_modbus(valor):
    # Convierte un entero de 32 bits en dos registros de 16 bits (Big Endian)
    packed = struct.pack('>I', valor)
    word1, word2 = struct.unpack('>HH', packed)
    return [word1, word2]

def escribir_registro(client, direccion, valor):
    # Prepara los dos registros de 16 bits para el valor de 32 bits
    payload = int_a_registros_modbus(valor)
    
    # Escribimos en la dirección especificada (1000 para Reg 1, 1002 para Reg 2)
    resultado = client.write_registers(address=direccion, values=payload)
    
    if resultado.isError():
        print(f"❌ Error al escribir en Modbus (Dir {direccion}): {resultado}")
    else:
        print(f"✅ Registro Modbus en dirección {direccion} modificado exitosamente al valor: {valor}")

def main():
    client = ModbusTcpClient(MIR_IP, port=PORT)
    
    if client.connect():
        print(f"Conexión TCP establecida con el MiR250 en {MIR_IP}")
        
        try:
            while True:
                comando = input("\nIngresa 'A' (Ir al Punto A), 'B' (Ir al Punto B), o 'Q' (Salir): ").strip().upper()
                
                if comando == 'A':
                    print("Enviando señal para Punto A (PLC Register 1 = 1)...")
                    # Dirección 1000 corresponde al PLC Register 1
                    escribir_registro(client, 1000, 1)
                elif comando == 'B':
                    print("Enviando señal para Punto B (PLC Register 2 = 2)...")
                    # Dirección 1002 corresponde al PLC Register 2 (cada entero de 32 bits usa 2 registros)
                    escribir_registro(client, 1002, 2)
                elif comando == 'Q':
                    print("Cerrando conexión...")
                    break
                else:
                    print("⚠️ Comando no válido.")
                    
        except KeyboardInterrupt:
            print("\nInterrupción por teclado detectada. Cerrando...")
        finally:
            client.close()
    else:
        print("❌ No se pudo conectar al servidor Modbus del MiR.")

if __name__ == "__main__":
    main()