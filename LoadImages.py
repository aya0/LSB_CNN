import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Function to load CIFAR-100 batch
def load_cifar100_batch(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    images = data_dict[b'data']        # shape: (num_samples, 3072)
    labels = data_dict[b'fine_labels'] # shape: (num_samples,)
    # reshape images to (num_samples, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32)
    images = np.transpose(images, (0, 2, 3, 1))  # channel-last
    return images, np.array(labels)

# Load train and test
train_images, train_labels = load_cifar100_batch('cifar-100-python/train')
test_images, test_labels   = load_cifar100_batch('cifar-100-python/test')

# Normalize images to 0-1
train_images = train_images.astype('float32') / 255.0
test_images  = test_images.astype('float32') / 255.0

# Convert labels to one-hot
train_labels = to_categorical(train_labels, 100)
test_labels  = to_categorical(test_labels, 100)

print("Train images:", train_images.shape)
print("Test images:", test_images.shape)
print("Train labels:", train_labels.shape)


plt.figure(figsize=(10,10))

for i in range(16):  # show first 16 images
    plt.subplot(4, 4, i+1)
    plt.imshow(train_images[i])
    plt.title(str(np.argmax(train_labels[i])))
    plt.axis('off')

plt.show()

