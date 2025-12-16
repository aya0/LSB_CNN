import torch
import cv2
import numpy as np
from StegoDetector import StegoDetector
from convert_image_bit import Convert_images_to_array_of_bit
import os 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = StegoDetector().to(device)
model.load_state_dict(torch.load("models/best_stego.pth", map_location=device))
model.eval()

# Detect Stego
def detect_stego(image_path):
    bit_tensor = Convert_images_to_array_of_bit(image_path)
    bit_tensor = bit_tensor.unsqueeze(0).to(device)  

    with torch.no_grad():
        output = model(bit_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction  # 0 = Normal, 1 = Stego


# Extract Secret Message (LSB)
def extract_secret_message(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bits = []
    for pixel in img.reshape(-1, 3):
        bits.append(pixel[0] & 1)  # LSB of Red channel

    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        char = chr(int("".join(map(str, byte)), 2))
        chars.append(char)

        if "".join(chars).endswith("#####"):
            break

    return "".join(chars).replace("#####", "")



# Embed Secret Message (LSB)
def embed_secret_message(image_path, secret, output_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    binary_secret = ''.join(format(ord(c), '08b') for c in secret)

    flat_img = img.reshape(-1, 3)

    if len(binary_secret) > len(flat_img):
        raise ValueError("Message is too long for this image")

    for i in range(len(binary_secret)):
        flat_img[i][0] = (flat_img[i][0] & ~1) | int(binary_secret[i])

    stego_img = flat_img.reshape(img.shape)
    cv2.imwrite(output_path, cv2.cvtColor(stego_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    
    while True:       
        image_path = input("Enter image path: ")
        
        if image_path.strip() == "":    
            print("Image path cannot be empty. Please try again.")
            continue    
        if not os.path.isfile(image_path):  
            print(f"No file found at {image_path}. Please try again.")
            continue  
          
        result = detect_stego(image_path)

        if result == 1:
           print("\nStego Image Detected!")
           secret = extract_secret_message(image_path)
           print("Hidden Message:", secret)

        else:
            print("\nNormal Image (No Secret Message Detected)")
            choice = input("Do you want to embed a secret message? (y/n): ")

            if choice.lower() == "y":
                secret = input("Enter your secret message: ")
                output_path = input ("Enter output image path : ")
                embed_secret_message(image_path, secret, output_path)
                print("\nSecret message embedded successfully!")
                print("Saved as:", output_path)
            else:
                print("No message embedded.")