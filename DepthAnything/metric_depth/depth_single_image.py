import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from DepthAnything.metric_depth.zoedepth.models.builder import build_model
from DepthAnything.metric_depth.zoedepth.utils.config import get_config
import cv2
import torch.nn.functional as F

class Estimator:
    def __init__(self, model_name, global_settings):
        self.model_name = model_name
        self.global_settings = global_settings
        self.model = self._load_model()

    def _load_model(self):
        # Configures the model based on the model name and global configuration
        config = get_config(self.model_name, "infer", self.global_settings['DATASET'])
        config.pretrained_resource = self.global_settings['pretrained_resource']
        
        # Builds and loads the model
        model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        return model

    def infer(self, numpy_image):
        # Converts the numpy image to a PyTorch tensor and processes it
        numpy_image_rgb = numpy_image[:, :, ::-1]  # Reverses channels from BGR to RGB
        color_image = Image.fromarray(np.uint8(numpy_image_rgb)).convert('RGB')
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Performs inference using the model
        with torch.no_grad():
            pred = self.model(image_tensor, dataset=self.global_settings['DATASET'])
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]

        # Re-projects to the original dimensions with the inverse process of preprocessing
        original_height = numpy_image.shape[0]
        original_width = numpy_image.shape[1]
        pred_original_dim = F.interpolate(pred, size=(original_height, original_width), mode='bilinear', align_corners=True)

        # Converts the output to numpy and returns it
        depth_image = pred_original_dim.squeeze().detach().cpu().numpy()

        return depth_image


