import cv2
import numpy as np
import torch

def Convert_images_to_array_of_bit(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # shape: (H, W, 3), dtype=uint8

    # Convert each channel to 8-bit binary → shape: (H, W, 24)
    bit_array = np.unpackbits(img, axis=2)

    # Convert to PyTorch tensor and permute to [C, H, W] → [24, H, W]
    bit_tensor = torch.tensor(bit_array, dtype=torch.float32).permute(2, 0, 1)

    return bit_tensor

# Example usage
bit_tensor = Convert_images_to_array_of_bit("preprocessed_images/train_images_after_preprcessing/0.png")
print(bit_tensor.shape)





    
        
       
