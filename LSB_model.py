import cv2
import numpy as np 
import torch

def Convert_pixel_to_binary(pixel):
    binary_pixel = ""
    while pixel > 0:
        if (pixel // 2) % 2 == 0:
            binary_pixel += "0"
        else:
            binary_pixel += "1"
        pixel = pixel // 2
        
    if len(binary_pixel) < 8:
        binary_pixel += "0" * (8 - len(binary_pixel))    
    binary_pixel = binary_pixel[::-1]  
    return binary_pixel


def Convert_images_to_array_of_bit(image_path):
    img_on_bit = []
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    #  Convert BGR to RGB
    #  make a numPy array ( height , width , channels) 
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    H, W, C = img.shape 
    
    print(img) 
    
    # Convert each pixel to its binary representation and store in img_on_bit array
    for pixel_row in range(img.shape[0]):
        row_in_bits = []
        for pixel_col in range(img.shape[1]):
            pixel = img[pixel_row, pixel_col]  
            pixel_bits = [Convert_pixel_to_binary(channel) for channel in pixel]
            row_in_bits.append(pixel_bits)
        img_on_bit.append(row_in_bits)
    
    # Convert img_on_bit to a NumPy array   
    bit_array = np.array(img_on_bit, dtype=np.uint8)

    # Flatten channel bits → [H, W, 24]
    bit_array = bit_array.reshape(H, W, C * 8)
    

    # Transpose to [channels, H, W] for PyTorch
    bit_tensor = torch.tensor(bit_array, dtype=torch.float32).permute(2, 0, 1)

    return  bit_tensor  
            
            
print(Convert_images_to_array_of_bit("preprocessed_images/train_images_after_preprcessing/0.png"))




    
        
       
