import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import cv2

import time
from IPython.display import clear_output

import detect_utils


# Выбор устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка  предобученной модели
model = fasterrcnn_mobilenet_v3_large_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)

# Перевод модели в режим предсказания и перенос на устройство
model = model.eval().to(device)

# Указать путь к видеофайлу здесь
video_path = 'input/crowd_1.mp4'

# Захват видео
cap = cv2.VideoCapture(video_path)

if (cap.isOpened() == False):
    print('Ошибка в чтении видео. Проверьте путь')

# Ширина и высота кадра
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Создание объекта для сохранения видео
out = cv2.VideoWriter('/outputs/crowd_processed.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

frame_count = 0  # Подсчет общего количества кадров
total_fps = 0  # Подсчет итогового FPS

# Пока видео не закончилось
while (cap.isOpened()):
    # Захватываем каждый кадр из видео
    ret, frame = cap.read()
    if ret == True:
        # Записываем начальное время для подсчета скорости работы
        start_time = time.time()
        with torch.no_grad():
            # Получить предсказания для текущего кадра
            boxes, classes, scores = detect_utils.predict(
                frame, model, device, 0.7)

        # Отрисовка границ объектов и предсказаний
        image = detect_utils.draw_boxes(boxes, classes, scores, frame)

        # Записываем время окончания обработки кадра
        end_time = time.time()
        # Считаем время обработки кадра
        fps = 1 / (end_time - start_time)
        # Добавляем к итоговому FPS
        total_fps += fps
        # Обновляем счетчик кадров
        frame_count += 1
        clear_output(wait=True)
        print(f"Количество кадров: {frame_count}, FPS: {fps}")

        # Преобразование BGR в RGB цветовое пространство для правильного отображения
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Запись кадра в видео
        out.write(image)

    else:
        break

# Закрытие видеопотока
cap.release()
# Закрытие видеофайла для правильной записи кадров
out.release()
# Закрытие всех кадров и окон
cv2.destroyAllWindows()

# Подсчет финального FPS
avg_fps = total_fps / frame_count
print(f"Средний FPS: {avg_fps:.3f}")
