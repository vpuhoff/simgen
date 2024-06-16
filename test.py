import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

print("Numpy version:", np.__version__)
print("PyTorch version:", torch.__version__)

# Пример использования torchvision.transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Создание случайного изображения и его преобразование
random_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
image = Image.fromarray(random_image)
transformed_image = transform(image)

print("Transformation successful:", transformed_image.shape)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")
