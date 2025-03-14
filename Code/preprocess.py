import cv2
import torch
import torchvision.transforms as transforms

# Define transformations
transform_greyscale = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.1])
])

class ScharrEdgeDetection(object):
    def __call__(self, img):
        img = img.numpy().squeeze(0)  # Convert tensor to numpy array and remove channel dimension
        grad_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharr_edges = cv2.magnitude(grad_x, grad_y)
        scharr_edges = cv2.normalize(scharr_edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return torch.tensor(scharr_edges, dtype=torch.float32).unsqueeze(0)  # Convert back to tensor and add channel dimension