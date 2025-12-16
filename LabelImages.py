import torch
from torch.utils.data import Dataset
from convert_image_bit import Convert_images_to_array_of_bit


class LabelImages(Dataset):
    def __init__(self, normal_paths, stego_paths):
        self.image_paths = normal_paths + stego_paths
        self.labels = [0]*len(normal_paths) + [1]*len(stego_paths)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get the bit  for the image
        bit_tensor = Convert_images_to_array_of_bit(self.image_paths[idx])
        label = self.labels[idx]
        return bit_tensor, torch.tensor(label, dtype=torch.long)