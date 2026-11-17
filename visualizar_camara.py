import cv2

cap = None
for idx in range(5):
    c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok, _ = c.read()
    if ok:
        print(f"Cámara encontrada en índice {idx}")
        cap = c
        break
    c.release()

if cap is None:
    raise SystemExit("No se encontró ninguna cámara")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Camara", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break


cap.release()
cv2.destroyAllWindows()