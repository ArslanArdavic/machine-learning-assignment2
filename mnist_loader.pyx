import os
import urllib.request
import gzip
import numpy as np

class MNISTLoader:
    def __init__(self, save_path='./mnist_data'):
        self.save_path = save_path
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.key_file = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }

    def download_mnist(self):
        os.makedirs(self.save_path, exist_ok=True)
        for v in self.key_file.values():
            filename = os.path.join(self.save_path, v)
            if not os.path.exists(filename):
                urllib.request.urlretrieve(self.url_base + v, filename)

    def load_mnist_images(self, filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_mnist_labels(self, filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def load_dataset(self):
        self.download_mnist()
        train_images = self.load_mnist_images(os.path.join(self.save_path, self.key_file['train_images']))
        train_labels = self.load_mnist_labels(os.path.join(self.save_path, self.key_file['train_labels']))
        test_images = self.load_mnist_images(os.path.join(self.save_path, self.key_file['test_images']))
        test_labels = self.load_mnist_labels(os.path.join(self.save_path, self.key_file['test_labels']))
        return train_images, train_labels, test_images, test_labels

# Example usage:
mnist_loader = MNISTLoader()
train_images, train_labels, test_images, test_labels = mnist_loader.load_dataset()
print("Downloading and loading MNIST dataset complete.")
