import sys
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pyautogui
import keyboard
import pygame
import tkinter as tk

sys.path.append('../')
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

    def infer(self, img):
        tik = time.time()

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img = self.process_image(img)

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

    def process_image(self, img):
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
        img.save("screen.jpg")
        return img

def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    return screenshot

running = False

def start_capture(fbp, region, label):
    global running
    while running:
        img = capture_screen(region)
        result = fbp.infer(img)
        print(f"Beauty Score: {result['beauty']}, Time Elapsed: {result['elapse']} seconds")

        # Обновление метки в GUI
        label.config(text=f"Beauty Score: {result['beauty']:.2f}")

        # Эмуляция клика мыши, если значение 'beauty' меньше 3.8
        if result['beauty'] < 3.7:
            pyautogui.click(x=220, y=890)
        else:
            # Остановка цикла и звуковой сигнал, если значение 'beauty' >= 3.8
            running = False
            pygame.mixer.init()
            pygame.mixer.music.load('alert_sound.mp3')  # Замените на путь к вашему звуковому файлу
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(1)  # Ожидание завершения воспроизведения

        time.sleep(2)  # Периодичность 2 секунды

def main():
    global running
    fbp = FacialBeautyPredictor(pretrained_model_path='ComboNet_SCUTFBP5500.pth')
    region = (1000, 50, 850, 850)  # Определенная область захвата (x, y, width, height)

    # Создание GUI
    root = tk.Tk()
    root.attributes("-topmost", True)  # Поверх всех окон
    root.geometry("+400+0")  # Позиционирование в правом верхнем углу
    label = tk.Label(root, text="Beauty Score: N/A", font=("Helvetica", 16))
    label.pack()

    def start_thread():
        global running
        if not running:
            running = True
            threading.Thread(target=start_capture, args=(fbp, region, label)).start()

    def stop_thread():
        global running
        running = False

    keyboard.add_hotkey('alt+f1', start_thread)
    keyboard.add_hotkey('alt+f2', stop_thread)

    print("Press ALT+F1 to start capturing and analyzing. Press ALT+F2 to stop.")
    threading.Thread(target=keyboard.wait, args=('esc',), daemon=True).start()  # Приложение будет работать, пока не будет нажата клавиша ESC

    root.mainloop()

if __name__ == '__main__':
    main()
