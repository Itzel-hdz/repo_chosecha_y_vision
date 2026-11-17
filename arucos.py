import cv2
from PIL import Image, ImageDraw

DPI = 300
MM = DPI / 25.4
PX = round(60 * MM)          # 60 mm de lado negro
MARGEN = round(10 * MM)      # 10 mm de margen blanco
IDS = [0, 1, 2, 3, 4, 5]

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
W, H = 2480, 3508            # A4 a 300 dpi
hoja = Image.new("L", (W, H), 255)
dr = ImageDraw.Draw(hoja)

for n, i in enumerate(IDS):
    m = cv2.aruco.generateImageMarker(d, i, PX)
    m = cv2.copyMakeBorder(m, MARGEN, MARGEN, MARGEN, MARGEN, cv2.BORDER_CONSTANT, value=255)
    im = Image.fromarray(m)
    im.save(f"aruco_{i}.png", dpi=(DPI, DPI))
    col, fila = n % 2, n // 3
    x = col * (W // 2) + (W // 2 - im.width) // 2
    y = fila * (H // 3) + (H // 3 - im.height) // 2
    hoja.paste(im, (x, y))
    dr.text((x + MARGEN, y + im.height - 40), f"ID {i}", fill=0)

hoja.save("arucos_0_a_5_A4.png", dpi=(DPI, DPI))
hoja.convert("RGB").save("arucos_0_a_5_A4.pdf", resolution=DPI)
print("listo")