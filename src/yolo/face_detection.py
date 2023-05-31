# !pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt


import torch
import torchvision.transforms as transforms
from PIL import Image
import requests

from io import BytesIO


class FaceDetector:
    def __init__(self):
        # Load YOLOv5 model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    def from_url(self, url):
        # URL of the image
        image_url = url

        # Send a GET request to fetch the image
        response = requests.get(image_url)

        # Open the image from the response content
        image = Image.open(BytesIO(response.content))
        return image

    def check_face(self, image_path=None, url=None):
        try:
            if url:
                image = self.from_url(url)
            else:
                # Open image using PIL
                image = Image.open(image_path)

            # Perform object detection
            results = self.model(image)

            # Extract detection information
            detections = results.xyxyn

            # Check if any face is detected
            for det in detections:
                if det[0][-1] == 0:  ## 0 is the key for person
                    return True

            return False

        except Exception as ex:
            return False


# Usage example
detector = FaceDetector()
image_path = "Face_Recognition/media/sorry_cat_no_face.jpeg"
# is_face_detected = detector.check_face(image_path)
is_face_detected = detector.check_face(
    url="https://img.a.transfermarkt.technology/portrait/big/3111-1478769687.jpg?lm=1"
)
print(is_face_detected)
