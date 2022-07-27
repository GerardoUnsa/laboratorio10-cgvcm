from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
from .utils import VideoCamera

import cv2
import tensorflow as tf
import numpy as np

# Return every frame lecture
def gen(camera):
    categories = {0:'A', 1:'E', 2:'I', 3:'O', 4:'U', 5:'nothing', 6:'space'}
    font = cv2.FONT_HERSHEY_SIMPLEX 
    kernel = np.ones((5,5), np.uint8)
    model = tf.keras.models.load_model('detection_app/SLD_CNN.model')

    while True:
        frame = camera.get_frame(font, kernel, model, categories)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Allow video streaming
def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
            content_type='multipart/x-mixed-replace; boundary=frame')

# Home view
def home(request):
    context = {}
    return render(request, 'detection_app/home.html', context)

# Detection view
def detection(request):
    context = {}
    return render(request, 'detection_app/detection.html', context)


