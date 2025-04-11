# Example to generate black images
from PIL import Image
import os

os.makedirs('PlantVillage/train/Unknown', exist_ok=True)
for i in range(50):
    img = Image.new('RGB', (224, 224), color=(0, 0, 0))  # black
    img.save(f'PlantVillage/train/Unknown/black_{i}.jpg')

# You can create more with noise or blur if needed
