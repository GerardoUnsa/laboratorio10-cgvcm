import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self, font, kernel, model, categories):
        ret, frame = self.video.read() # read the frame

        # filters
        update = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        update = cv2.GaussianBlur(update, (5,5), 0)
        _, update = cv2.threshold(update, 65, 255, cv2.THRESH_BINARY)
        update = cv2.morphologyEx(update, cv2.MORPH_GRADIENT, kernel)
        update = cv2.cvtColor(update, cv2.COLOR_GRAY2RGB)
        update_resize = cv2.resize(update, (100,100))
        
        # sign prediction
        img = image.img_to_array(update_resize)
        img_tensor = np.expand_dims(img, axis=0)
        predict = np.argmax(model.predict(img_tensor))
        predict = categories[predict]
        
        # Add the prediction text
        frame = cv2.putText(frame, predict, (10, 450), font, 3, (255,255,255), 4, cv2.LINE_AA)

        # Flip the frame
        #frame_flip = cv2.flip(frame, 1)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
