import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        """
        data_dir : path of dataset folder
        transform : preprocessing steps
        """

        self.data_dir = data_dir
        
        self.transform = transform

        self.image_paths = []
        self.labels = []

        classes = os.listdir(data_dir)

        for label, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)

            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)

                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
    