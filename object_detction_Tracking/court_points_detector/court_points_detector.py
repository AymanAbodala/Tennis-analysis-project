import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models

class CourtPointsDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        Predict keypoints from a given image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image

        Returns
        -------
        keypoints : numpy.ndarray
            The predicted keypoints in the format of [x1, y1, x2, y2, ...]
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0) 
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        # denormalize keypoints to original image size
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """
        Plot keypoints on the image

        Parameters
        ----------
        image : numpy.ndarray
            The input image
        keypoints : numpy.ndarray
            The predicted keypoints in the format of [x1, y1, x2, y2, ...]

        Returns
        -------
        image : numpy.ndarray
            The image with keypoints plotted
        """
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draws keypoints on a given list of frames.

        Parameters
        ----------
        video_frames (list): A list of frames to draw keypoints on.
        keypoints (numpy.ndarray): The predicted keypoints in the format of [x1, y1, x2, y2, ...]

        Returns
        -------
        list: A list of frames with keypoints plotted.
        """
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames  
    