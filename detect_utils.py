import torchvision.transforms as transforms
import cv2
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names


# Цвет для отрисовки границ объектов
COLOR = [255, 0, 0]  # Красный

# Создаем транформацию объекта в вектор
transform = transforms.Compose([
    transforms.ToTensor(),
])


def predict(image, model, device, detection_threshold):
    '''Предсказание модели.

    Функция получает на вход кадр, модель, устройство и порог обнаружения. 
    Модель получает на вход кадр. Выходом модели являются классы, обнаруженные 
    в кадре, уровень уверенности в данном классе и границы объектов, найденных в кадре.
    Затем оставляются только предсказания для класса "человек" соответствующие заданному
    порогу. Функция возвращает границы обектов, классы и уровень уверенности.

    Args:
        image (numpy.ndarray): Входной кадр в формате numpy array.
        model (torch.nn.Module): Предобученная модель для предсказаний.
        device (torch.device): Устройство (CPU или CUDA), на котором выполняется предсказание.
        detection_threshold (float): Порог уверенности для отбора предсказаний.

    Returns:
        tuple: Кортеж, содержащий три элемента:
            - boxes (numpy.ndarray): Границы обнаруженных объектов.
            - classes (list of str): Классы обнаруженных объектов.
            - scores (numpy.ndarray): Уровень уверенности для обнаруженных объектов.
    '''

    # Переводим кадр в вектор и переносим на устройство
    image = transform(image).to(device)
    image = image.unsqueeze(0)  # добавляем размерность батча
    outputs = model(image)  # делаем предсказания для кадра

    # Получаем все предсказания классов
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]

    # Получаем все показатели уверенности для классов
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    # Получаем все границы объектов
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    # Оставляем только предсказания для класса с людьми
    person_indices = [i for i, label in enumerate(
        outputs[0]['labels'].cpu().numpy()) if label == 1]

    # Выыбираем границы объектов, уверенность и класс только для людей
    person_boxes = pred_bboxes[person_indices]
    person_scores = pred_scores[person_indices]
    person_classes = [pred_classes[i] for i in person_indices]

    # Оставляем только объекты, уверенность в которых больше заданного порога
    boxes = person_boxes[person_scores >= detection_threshold].astype(np.int32)
    scores = person_scores[person_scores >= detection_threshold]
    classes = [person_classes[i] for i in range(
        len(person_classes)) if person_scores[i] >= detection_threshold]

    return boxes, classes, scores


def draw_boxes(boxes, classes, scores, image):
    '''Функция для отрисовки предсказанных объектов в кадре.

    Функция получает на вход границы объектов, классы, уровень уверенности и кадр.
    Идет отрисовка прямоугольника соответствующего границам обнаруженных объектов,
    также подписывается класс и уровень уверенности в нем.

    На выходе функции получается кадр с отрисованными отъектами.

    Args:
        boxes (numpy.ndarray): Границы обнаруженных объектов.
        classes (list of str): Классы обнаруженных объектов.
        scores (numpy.ndarray): Уровень уверенности для обнаруженных объектов.
        image (numpy.ndarray): Входной кадр в формате numpy array.

    Returns:
        numpy.ndarray: Кадр с отрисованными объектами.
    '''
    # Преобразование RGB в BGR цветовое пространство для правильной работы
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    # Для каждой коробки
    for i, box in enumerate(boxes):
        # Отображение границы
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            COLOR, 1
        )
        # Надпись
        cv2.putText(image, f'{classes[i]}: {scores[i]:.2f}', (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2,
                    lineType=cv2.LINE_AA)
    return image
