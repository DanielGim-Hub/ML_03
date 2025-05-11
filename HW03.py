import cv2
import mediapipe as mp
import numpy as np
import math
import face_recognition

# Инициализация распознавания лиц, рук и эмоций
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

# Имена владельца программы
OWNER_NAME = 'Daniel'
OWNER_SURNAME = 'Gimaev'
OWNER_IMAGE_PATH = 'owner.jpg'

# Загрузка и распознавание лица владельца
owner_image = face_recognition.load_image_file(OWNER_IMAGE_PATH)
owner_encoding = face_recognition.face_encodings(owner_image)[0]


# Функция для определения эмоции (улыбка, удивление, грусть)
def detect_emotion(landmarks):
    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]
    mouth_left = landmarks[78]
    mouth_right = landmarks[308]
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]

    mouth_height = abs(mouth_top.y - mouth_bottom.y)
    mouth_width = abs(mouth_right.x - mouth_left.x)
    left_eye_open = abs(left_eye_bottom.y - left_eye_top.y)
    right_eye_open = abs(right_eye_bottom.y - right_eye_top.y)
    avg_eye_open = (left_eye_open + right_eye_open) / 2

    if mouth_height > 0.05 and avg_eye_open > 0.04:
        return 'удивление'
    elif mouth_height > 0.03 and avg_eye_open > 0.03:
        return 'улыбается'
    elif mouth_height < 0.02:
        return 'грустит'
    else:
        return 'нейтральное'


# Открываем веб-камеру
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Перевод в формат RGB для Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = mp_face_detection.process(rgb_frame)
    results_hands = mp_hands.process(rgb_frame)
    results_mesh = mp_face_mesh.process(rgb_frame)
    height, width, _ = frame.shape

    # Проверка лица владельца
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
    face_encodings = face_recognition.face_encodings(small_frame)
    is_owner = False
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([owner_encoding], face_encoding)
        if True in matches:
            is_owner = True
            break

    # Обработка обнаруженных лиц
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1, y1 = int(bboxC.xmin * width), int(bboxC.ymin * height)
            x2, y2 = int((bboxC.xmin + bboxC.width) * width), int((bboxC.ymin + bboxC.height) * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Метка владельца или неизвестного
            label = OWNER_NAME if is_owner else 'неизвестный'

            # Подсчет пальцев
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    fingers_up = []
                    lm_list = hand_landmarks.landmark
                    fingers_up.append(lm_list[8].y < lm_list[6].y)  # указательный
                    fingers_up.append(lm_list[12].y < lm_list[10].y)  # средний
                    fingers_up.append(lm_list[16].y < lm_list[14].y)  # безымянный
                    fingers_up.append(lm_list[20].y < lm_list[18].y)  # мизинец
                    fingers_up.append(
                        lm_list[4].x < lm_list[3].x)  # большой палец (проверка по горизонтали для правой руки)
                    count_fingers = fingers_up.count(True)

                    # Определение действия по количеству пальцев
                    if count_fingers == 1 and is_owner:
                        label = OWNER_NAME
                    elif count_fingers == 2 and is_owner:
                        label = OWNER_SURNAME
                    elif count_fingers == 3 and is_owner and results_mesh.multi_face_landmarks:
                        for face_landmarks in results_mesh.multi_face_landmarks:
                            emotion = detect_emotion(face_landmarks.landmark)
                            label = f'{OWNER_NAME} ({emotion})'
                            break

            # Добавляем текст с именем или фамилией
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Отображаем итоговый кадр
    cv2.imshow('Face and Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
