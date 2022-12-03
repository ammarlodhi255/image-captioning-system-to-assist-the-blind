import cv2
import tensorflow as tf
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm
import base64
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers \
    import Input, Dense, add, LSTM, Embedding, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gtts import gTTS
from IPython.display import Audio
sys.path.append(os.path.abspath('./model'))

'''
    1. resnet_model1_res.hdf5
    2. vgg_model5_3.hdf5
    3. vgg16_model.hdf5
'''

vocab_size = 20572
# vocab_size = 18126
max_length = 155
shape = 4096
model_to_use = 'VGG16'
base_dir = "./static"
path_to_model = os.path.join(base_dir, "vgg_model5_3.hdf5")


def load_model(saved_model_path):
    conv_inputs = Input(shape=(shape,))
    fe1 = Dropout(0.5)(conv_inputs)
    fe2 = Dense(256, activation='relu')(fe1)

    seq_inputs = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(seq_inputs)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[conv_inputs, seq_inputs], outputs=outputs)
    model.load_weights(saved_model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def init_model():
    path = path_to_model
    loaded_model = load_model(path)
    print('Model Loaded Successfully')
    return loaded_model


def get_tokenizer():
    with open("./static/vocab.pkl", 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def idx_to_word(integer, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == integer:
            return word
    return None


def get_pretrained_model(model_name):
    if model_name == 'VGG16':
        model = VGG16()
    elif model_name == 'Resnet':
        model = ResNet50()

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def get_pre_trained_processed_img(img, model_name):
    if model_name == 'Resnet':
        img = tf.keras.applications.resnet50.preprocess_input(img)
    elif model_name == 'VGG16':
        img = tf.keras.applications.vgg16.preprocess_input(img)

    return img


def pre_process(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = get_pre_trained_processed_img(img, model_to_use)
    pretrained_model = get_pretrained_model(model_to_use)
    feature = pretrained_model.predict(img, verbose=0)
    return feature


def predict_caption(model, img_path):
    tokenizer = get_tokenizer()
    print(len(tokenizer.word_index))
    feature = pre_process(img_path)

    in_text = 'startseq'

    for i in tqdm(range(max_length)):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        next_word = model.predict(
            [feature, sequence], verbose=0)
        next_word = np.argmax(next_word)
        word = idx_to_word(next_word, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return ' '.join([word for word in in_text.split(' ') if word not in ['startseq', 'endseq']])
