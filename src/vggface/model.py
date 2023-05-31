import cv2
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions


class FaceRecognition:
    def __init__(self):
        # create a vggface model
        self.model = VGGFace(model="resnet50")
        self.image_size = (224, 224)
        self.detector = MTCNN()
        self.bounding_box = [None] * 4
        self.threshold = 90

    # extract a single face from a given photograph
    def extract_face(self, img_np: np.array):
        # create the detector, using default weights
        # detect faces in the image
        results = self.detector.detect_faces(img_np)

        # extract the bounding box from the first face
        x1, y1, width, height = results[0]["box"]
        x2, y2 = x1 + width, y1 + height

        ##### It will raise an execption if there is no face detected however yolo will do the trick too
        self.bounding_box = (x1, y1), (x2, y2)
        # extract the face
        face = img_np[y1:y2, x1:x2]

        # resize img_np to the model size
        image = Image.fromarray(face)
        image = image.resize(self.image_size)
        face_array = np.asarray(image).astype(np.float32)
        return face_array

    def __predict(self, file_path=None, image_url=None):
        # load image from file
        if file_path:
            real_img_np = plt.imread(file_path)
        elif image_url:
            response = requests.get(image_url)
            real_img_np = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
        else:
            return None

        # load the photo and extract the face
        img_np = self.extract_face(real_img_np)

        samples = np.expand_dims(img_np, axis=0)

        # prepare the face for the model, e.g. center img_np
        samples = preprocess_input(samples, version=2)

        # perform prediction
        predictions = decode_predictions(self.model.predict(samples))[0]

        final_predictions = {}
        for _prediction, confidence in predictions:
            prediction = (
                _prediction.encode("utf-8")
                .decode("unicode_escape")[3:-1]
                .replace("_", " ")
            )
            final_predictions[prediction] = confidence * 100
        return final_predictions, real_img_np.astype(np.uint8)

    def __draw_bounding_boxes(self, real_img_np):
        bounding_box = self.bounding_box
        # remember cv2 apply inplace
        cv2.rectangle(real_img_np, bounding_box[0], bounding_box[1], (0, 255, 255), 2)
        return real_img_np

    def __sanity_check(self, predictions):
        return max(predictions.values()) >= self.threshold

    def plot_results(self, label, real_img_recognized):
        plt.imshow(real_img_recognized)
        plt.axis("off")
        plt.title(label)
        plt.show()
        plt.close()

    def convert_numpy_image(self, img_np):
        img = Image.fromarray(img_np, "RGB")
        data = BytesIO()
        img.save(data, "JPEG")  # pick your format
        data64 = base64.b64encode(data.getvalue())
        return "data:img/jpeg;base64," + data64.decode("utf-8")

    def predict(self, file_path=None, image_url=None, plot_results=False):
        predictions, real_img_np = self.__predict(file_path, image_url)
        real_img_recognized = self.__draw_bounding_boxes(real_img_np)
        label = max(predictions, key=predictions.get)
        confidence = round(predictions[label], 2)
        is_in_dataset = self.__sanity_check(predictions)

        if plot_results:
            self.plot_results(label, real_img_recognized)

        # This is additional step to be properly printed by html
        # Convert the array to an image using Pillow
        img = self.convert_numpy_image(real_img_recognized)

        # image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return label, confidence, img, is_in_dataset


if __name__ == "__main__":
    no_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHleuy3iFBnpOLIwRuuvVWp25InbO_yFLNdqHq3F7aU7wewm6oF2vtdCrXIjGfnktyhw0&usqp=CAU"
    schumi_url = "https://i2-prod.walesonline.co.uk/incoming/article25126388.ece/ALTERNATES/s1200c/1_JS189048963jp.jpg"
    kimi_url = "https://cdn.racingnews365.com/Riders/Raikkonen/_570x570_crop_center-center_none/f1_2021_kr_alf_lg.png?v=1643809079"
    f = FaceRecognition()
    label, confidence, real_img_recognized, is_in_dataset = f.predict(image_url=no_url)
    print(confidence, label)
    label, confidence, real_img_recognized, is_in_dataset = f.predict(
        file_path="face_Recognition/media/sorry_cat_no_face.jpeg"
    )
    print(confidence, label)
    real_img_recognized
