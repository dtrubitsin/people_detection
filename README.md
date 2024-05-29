# Детекция людей на видео
Программа для детекции людей на видео с использованием модели Faster RCNN MobileNet V3 из библиотеки PyTorch. При запуске на Kaggle с GPU P100 средний FPS равен 16.446

## Конфигурация
- Pytorch
- Faster RCNN MobileNet V3 large FPN
- cv2

## Описание файлов
- В ноутбуке приведено описание проекта с подробным объяснением и выводами по проекту
- detect_utils.py Содержит функции для рабоы
- detect_video.py Основная часть программы
- coco_names.py Список имен классов COCO
- requirements.txt
- README.md
- input/crowd.mp4 Исходное видео
- outputs/crowd_processed.mp4 Обработанное видео с обнаруженными объектами
