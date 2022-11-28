import cv2
import tensorflow as tf
import numpy as np
import os
import sys
import base64
from tensorflow.keras.layers import Input, Dense, add, LSTM, Embedding, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
sys.path.append(os.path.abspath('./model'))


def load_model(shape, saved_model_path):
    conv_inputs = Input(shape=(shape,))
    fe1 = Dropout(0.5)(conv_inputs)
    fe2 = Dense(256, activation='relu')(fe1)
    seq_inputs = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(seq_inputs)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[conv_inputs, seq_inputs], outputs=outputs)
    model.load_weights(saved_model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def init_model():
    loaded_model = load_model(4096, '')
    print('Model Loaded Successfully')
    return loaded_model


# decode from base64 and save the image
def process_image(img):
    with open('img_output.png', 'wb') as output:
        output.write(base64.b64decode(str(img, 'utf-8')))


def predict(model):
    pass


def idx_to_word(integer, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        next_word = model.predict(
            [np.array(image), np.array(sequence)], verbose=0)
        next_word = np.argmax(next_word)
        word = idx_to_word(next_word, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text
