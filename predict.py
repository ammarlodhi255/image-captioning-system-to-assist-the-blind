import cv2
import tensorflow as tf
import numpy as np
import os
import sys
import base64
sys.path.append(os.path.abspath('./model'))

def init_model():
    loaded_model = tf.keras.modes.load_model('')
    print('Model Loaded Successfully')
    return loaded_model

def process_image(img):
    with open('img_output.png', 'wb') as output:
        output.write(base64.b64decode(str(img, 'utf-8')))