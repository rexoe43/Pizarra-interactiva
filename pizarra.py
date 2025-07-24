import cv2
import numpy as np
import mediapipe as mp
import time

# Importar Tkinter para entrada de texto
import tkinter as tk
from tkinter import simpledialog

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Crear ventana en pantalla completa
cv2.namedWindow("Pizarra Interactiva", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pizarra Interactiva", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Verificar si la cámara se ha abierto correctamente
if not cap.isOpened():
    print("Error al abrir la cámara.")
    cap.release()
    exit()

# Inicializar Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Leer primer frame para dimensiones
ret, frame = cap.read()
if not ret:
    print("No se pudo acceder a la cámara.")
    cap.release()
    exit()

height, width, _ = frame.shape
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Definir el área de dibujo (cuadro más grande)
box_top_left = (int(width * 0.1), int(height * 0.1))
box_bottom_right = (int(width * 0.9), int(height * 0.9))

# Variables para posición previa del dedo y control de dibujo
drawing_enabled = False
detection_start = None
prev_x, prev_y = None, None

# Color por defecto (rojo)
current_color = (0, 0, 255)

# Variables de texto
text_mode = False
text_position = (width // 2, height // 2)
text = ""
text_being_dragged = False
font_scale = 1.0
text_color = (255, 255, 255)

# Función para detectar si la mano está cerrada (puño)
def is_fist(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    for tip_id in tips_ids[1:]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            return False
    return True

# Función para verificar si un punto está dentro del área
def in_drawing_area(x, y):
    return box_top_left[0] <= x <= box_bottom_right[0] and box_top_left[1] <= y <= box_bottom_right[1]

# Función para mover texto
def mouse_callback(event, x, y, flags, param):
    global text_position, text_being_dragged

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_drawing_area(x, y):
            tx, ty = text_position
            if tx - 100 <= x <= tx + 100 and ty - 30 <= y <= ty + 10:
                text_being_dragged = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if text_being_dragged and in_drawing_area(x, y):
            text_position = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        text_being_dragged = False

cv2.setMouseCallback("Pizarra Interactiva", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    hand_present = bool(result and result.multi_hand_landmarks)

    # Control de tiempo antes de habilitar dibujo
    if hand_present:
        if detection_start is None:
            detection_start = time.time()
        elif not drawing_enabled and time.time() - detection_start >= 2:
            drawing_enabled = True
    else:
        detection_start = None
        drawing_enabled = False
        prev_x, prev_y = None, None

    if drawing_enabled:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            x = int(lm[8].x * width)
            y = int(lm[8].y * height)

            if in_drawing_area(x, y):
                if is_fist(lm):
                    cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                    prev_x, prev_y = None, None
                else:
                    if prev_x is not None and prev_y is not None and in_drawing_area(prev_x, prev_y):
                        cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, 5)
                    prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    out = cv2.add(frame_bg, canvas_fg)

    cv2.rectangle(out, box_top_left, box_bottom_right, (0, 255, 0), 2)
    cv2.rectangle(out, (10, 10), (60, 60), current_color, -1)
    cv2.putText(out, "Color", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Dibujar texto si está activado
    if text_mode and text:
        cv2.putText(out, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

    cv2.imshow("Pizarra Interactiva", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        drawing_enabled = False
        detection_start = None
        prev_x, prev_y = None, None
    elif key == ord('f'):
        prop = cv2.getWindowProperty("Pizarra Interactiva", cv2.WND_PROP_FULLSCREEN)
        new_state = cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty("Pizarra Interactiva", cv2.WND_PROP_FULLSCREEN, new_state)
    elif key == ord('r'):
        current_color = (0, 0, 255)
    elif key == ord('g'):
        current_color = (0, 255, 0)
    elif key == ord('b'):
        current_color = (255, 0, 0)
    elif key == ord('w'):
        current_color = (255, 255, 255)
    elif key == ord('k'):
        current_color = (0, 0, 0)
    elif key == ord('i'):
        text_mode = True
        root = tk.Tk()
        root.withdraw()
        text_input = simpledialog.askstring("Entrada de texto", "Escribe el texto a mostrar en la pizarra:")
        if text_input is not None:
            text = text_input
            text_position = (width // 2, height // 2)
        root.destroy()
    elif key == ord('+'):
        font_scale = min(font_scale + 0.1, 5.0)
    elif key == ord('-'):
        font_scale = max(font_scale - 0.1, 0.5)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
