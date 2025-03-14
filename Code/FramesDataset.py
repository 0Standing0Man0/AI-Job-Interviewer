import os
from PIL import Image
from torch.utils.data import Dataset

# Define a dataset class
class FramesDataset(Dataset):
    def __init__(self, folder_path, transform_greyscale, edge_detection):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform_greyscale = transform_greyscale
        self.edge_detection = edge_detection

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform_greyscale(img)
        img = self.edge_detection(img)
        return img