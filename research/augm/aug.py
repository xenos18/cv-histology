import torchvision.utils
from PIL import Image
import torch
from torchvision.io import read_image
from torchvision.transforms import v2


def bmp_to_jpg(img_path):
    '''
    :param img_path: полный путь до картинки с расширением: ".bmp"
    :return: сохраняет картинку в формате
    '''
    img = Image.open(img_path)
    new_img_path = img_path.replace('.bmp', '.jpg')
    img.save(new_img_path)


def aug(image_name, path, anno_dir, IMG_SIZE, upload_directory):
    '''
    :param image_name: Название файла, не учитывая его расширения
    :param path: путь к папке, в которой хранятся картинки
    :param anno_dir: путь к папке, в которой хранятся аннотации
    :param IMG_SIZE: размер картинки, которую хотим получить IMG_SIZE * IMG_SIZE
    :return: Функция создаёт файл в папке path с названием вида f'{path}/{image_name}_aug.jpg' - преобразованная картинка
    и также f"{anno_dir}/{image_name}_anno_aug.jpg" - преобразованная аннотация
    '''
    # read the image
    img_path = f"{path}/{image_name}.jpg"
    # print(img_path)
    bmp_to_jpg(img_path.replace('.jpg', '.bmp'))
    anno_path = f"{anno_dir}/{image_name}_anno.jpg"
    bmp_to_jpg(anno_path.replace('.jpg', '.bmp'))
    img = read_image(img_path)
    anno = read_image(anno_path)

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    state = torch.get_rng_state()
    new_img = transforms(img)
    torch.set_rng_state(state)
    new_anno = transforms(anno)

    # сохраняем преобразованные картинки
    torchvision.utils.save_image(new_img, f'{upload_directory}/images/{image_name}_aug.jpg')
    torchvision.utils.save_image(new_anno, f"{upload_directory}/labels/{image_name}_anno_aug.jpg")
