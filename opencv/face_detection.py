import cv2
import numpy as np

# Carica il classificatore pre-addestrato per il rilevamento dei volti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carica le immagini del cappello e degli occhiali (con canale alfa)
hat = cv2.imread('opencv/hat.png', cv2.IMREAD_UNCHANGED)
sunglasses = cv2.imread('opencv/sunglasses.png', cv2.IMREAD_UNCHANGED)

# Funzione per sovrapporre PNG con trasparenza
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    overlay_h, overlay_w = img_overlay.shape[:2]

    # Limita l'overlay ai bordi dell'immagine
    if y + overlay_h > img.shape[0]:
        overlay_h = img.shape[0] - y
    if x + overlay_w > img.shape[1]:
        overlay_w = img.shape[1] - x
    if overlay_h <= 0 or overlay_w <= 0:
        return img

    overlay = img_overlay[:overlay_h, :overlay_w]
    mask = overlay[..., 3:] / 255.0
    img_crop = img[y:y+overlay_h, x:x+overlay_w]

    img[y:y+overlay_h, x:x+overlay_w] = (1.0 - mask) * img_crop + mask * overlay[..., :3]
    return img

# Apri la webcam (usa 0 per la webcam predefinita)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Leggi un frame dalla webcam
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converti in scala di grigi
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # Rileva i volti

    for (x, y, w, h) in faces:
        # Ridimensiona cappello e occhiali in base alla larghezza del volto
        hat_width = w
        hat_height = int(h * 0.5)
        hat_resized = cv2.resize(hat, (hat_width, hat_height), interpolation=cv2.INTER_AREA)

        sunglasses_width = w
        sunglasses_height = int(h * 0.35)
        sunglasses_resized = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_AREA)

        # Calcola le posizioni
        hat_y = max(0, y - hat_height)
        frame = overlay_image_alpha(frame, hat_resized, (x, hat_y + 10))

        sunglasses_y = y + int(h * 0.35) - int(sunglasses_height / 2)
        frame = overlay_image_alpha(frame, sunglasses_resized, (x, sunglasses_y))

        # Disegna il rettangolo del volto (opzionale)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)  # Mostra il frame

    # Esci premendo 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()