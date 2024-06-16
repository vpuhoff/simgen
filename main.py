import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from models.nets import ComboNet


class FacialBeautyPredictor:
    """
    Facial Beauty Predictor
    """

    def __init__(self, pretrained_model_path):
        model = ComboNet(num_out=5, backbone_net_name='SEResNeXt50')
        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Загрузка весов модели
        if torch.cuda.is_available():
            print("GPU Name:", torch.cuda.get_device_name(0))
            print("CUDA Version:", torch.version.cuda)
            print("PyTorch CUDA Version:", torch.version.cuda)
            print("We are running on", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.` if it exists
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.to(device)
        model.eval()

        self.device = device
        self.model = model

    def infer(self, img_file):
        tik = time.time()
        img = Image.open(img_file)

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = self.save(img_file, img)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension
        img = img.to(self.device)

        with torch.no_grad():
            score, cls = self.model(img)

        tok = time.time()

        return {
            'beauty': float(score.to('cpu').detach().item()),
            'elapse': tok - tik
        }

    def save(self, img_file, img):
        # Определение минимальной стороны для обрезки до квадрата
        min_side = min(img.size)
        # Обрезка до квадрата, центрирование
        left = (img.width - min_side) / 2
        top = (img.height - min_side) / 2
        right = (img.width + min_side) / 2
        bottom = (img.height + min_side) / 2
        img = img.crop((left, top, right, bottom))
        # Преобразование: Resize до 224x224
        img = img.resize((224, 224))
        img.save(f'{img_file}.jpg', format='JPEG')
        return img


def process_images_in_folder(folder_path, predictor):
    results = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith('.png'):
            result = predictor.infer(file_path)
            results.append((file_name, result))
    return results


if __name__ == '__main__':
    fbp = FacialBeautyPredictor(pretrained_model_path='ComboNet_SCUTFBP5500.pth')
    folder_path = './test_images/'
    results = process_images_in_folder(folder_path, fbp)
    for file_name, result in results:
        print(f"File: {file_name}, Beauty Score: {result['beauty']}, Time Elapsed: {result['elapse']} seconds")
