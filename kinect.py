import cv2
import mediapipe as mp
import numpy as np
import datetime

# Inicializa la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Configura la captura de la webcam real con la resolución especificada
window_width, window_height = 1368, 768
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

# Crear un lienzo para dibujar el rastro
canvas = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)

# Variables para almacenar la posición anterior del dedo índice
prev_x, prev_y = -1, -1

# Variable para almacenar la ruta del dedo índice
finger_path = []

# Variable para almacenar el estado anterior de la mano (abierta o cerrada)
prev_hand_open = False

# Color y grosor del lápiz
pencil_color = (255, 255, 255)  # Color verde
pencil_thickness = 30 # Grosor de 10 píxeles

# Función para borrar la pantalla
def clear_canvas():
    global canvas
    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

# Función para guardar la imagen con fondo negro
def save_screenshot():
    global canvas
    filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    cv2.imwrite(filename, canvas)
    print(f"Screenshot guardado como {filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar fotograma de la webcam")
        break

    # Voltea el fotograma horizontalmente
    frame = cv2.flip(frame, 1)

    # Convierte el fotograma de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta manos en el fotograma
    results = hands.process(rgb_frame)

    # Dibuja una línea que sigue el movimiento del dedo índice
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Calcula la distancia entre el pulgar y los otros dedos
            thumb = landmarks[4]  # Punto de referencia del pulgar
            index_finger = landmarks[8]  # Punto de referencia del dedo índice
            dist_thumb_index = np.sqrt((thumb.x - index_finger.x) ** 2 + (thumb.y - index_finger.y) ** 2)

            # Determina si la mano está abierta o cerrada
            hand_open = dist_thumb_index > 0.1

            # Si la mano estaba cerrada y ahora está abierta, comienza una nueva línea
            if prev_hand_open and not hand_open:
                finger_path = []

            # Si la mano está abierta, añade la posición del dedo índice a la ruta
            if hand_open:
                finger_path.append((int(index_finger.x * window_width), int(index_finger.y * window_height)))

            # Dibuja la ruta del dedo índice
            for i in range(1, len(finger_path)):
                cv2.line(canvas, finger_path[i - 1], finger_path[i], pencil_color, pencil_thickness)

            # Actualiza el estado previo de la mano
            prev_hand_open = hand_open

    # Combina el lienzo con el fotograma de la cámara
    result_frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

    # Muestra el resultado
    cv2.imshow('Hand Movement', result_frame)

    # Detección de eventos del teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Presiona 'q' para salir
        break
    elif key == ord('b'):  # Presiona 'b' para borrar la pantalla
        clear_canvas()
    elif key == ord('s'):  # Presiona 's' para guardar un pantallazo
        save_screenshot()

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
