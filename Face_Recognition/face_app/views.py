from django.shortcuts import render
from src.vggface.model import FaceRecognition
from src.yolo.face_detection import FaceDetector
from .utils import saying_sorry, sorry_img

#change her base and index
def index(request):
    return render(request, "index.html")


def from_url(request):
    if request.method == "POST":
        image_url = request.POST.get("image-url")
        is_there_face = FaceDetector().check_face(url=image_url)
        if not is_there_face:
            no_context = {"is_there_face": is_there_face, 'saying_sorry': saying_sorry(), 'sorry_img': sorry_img()}
            return render(request, "index.html", no_context)
        label, confidence, real_img_recognized, is_in_dataset = FaceRecognition().predict(
            image_url=image_url
        )
        context = {
            "is_there_face": is_there_face,
            "label": label,
            "confidence": confidence,
            "detection": real_img_recognized,
            "is_in_dataset": is_in_dataset,
            "rotation": int(confidence * 1.8),
        }
        # Do something with the URL
        return render(request, "index.html", context)
    else:
        return render(request, "index.html")

def from_file(request):
    # you changeed to delete with file name not the path now
    if request.method == "POST":
        image_file = request.FILES.get("image_file")
        is_there_face = FaceDetector().check_face(image_path=image_file)

        if not is_there_face:
            no_context = {"is_there_face": is_there_face, 'saying_sorry': saying_sorry(), 'sorry_img': sorry_img()}
            return render(request, "index.html", no_context)

        label, confidence, real_img_recognized, is_in_dataset = FaceRecognition().predict(file_path=image_file)
        context = {
            "is_there_face": is_there_face,
            "label": label,
            "confidence": confidence,
            "detection": real_img_recognized,
            "is_in_dataset": is_in_dataset,
            "rotation": int(confidence * 1.8),
        }
        # Do something with the URL
        return render(request, "index.html", context)
    else:
        return render(request, "index.html")
