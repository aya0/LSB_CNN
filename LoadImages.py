import os
import pickle
from PIL import Image
import numpy as np

# Function to load CIFAR-100 batch
def load_images(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    images = data_dict[b'data']
    images = images.reshape(-1, 3, 32, 32)
    images = np.transpose(images, (0, 2, 3, 1)) 
    return images

def load_images_to_folder():
    os.makedirs("images_cifar/train", exist_ok=True)
    
    # Save train images to folder
    for i , img in enumerate(load_images('cifar-100-python/train')):
        img = Image.fromarray(img)
        img.save(f"images_cifar/train/{i}.png")
        
    # Save test images to folder
    os.makedirs("images_cifar/test", exist_ok=True)
    for i , img in enumerate(load_images('cifar-100-python/test')): 
        img = Image.fromarray(img)
        img.save(f"images_cifar/test/{i}.png")      
        


load_images_to_folder()




