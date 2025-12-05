import os
import cv2
import numpy as np

# Function to safely load and preprocess LSB images
def preprocess_image(img_path, flip=False):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    #  convert BGR → RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # geometric augmentation 
    if flip:
        img = cv2.flip(img, 1)  # horizontal flip

    return img

# Process all images in folder
def process_image_in_folder(input_folderName , output_folderName):
    os.makedirs(output_folderName, exist_ok=True)

    for img_name in os.listdir(input_folderName ):
        img_path = os.path.join(input_folderName, img_name)
    
    # Preprocess image 
        img = preprocess_image(img_path, flip=np.random.rand() > 0.5)
    # Save images after preprocessing
        save_path = os.path.join(output_folderName, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
process_image_in_folder("images_cifar/train", "preprocessed_images/train_images_after_preprcessing")        
process_image_in_folder("images_cifar/test", "preprocessed_images/test_images_after_preprcessing")        
    
