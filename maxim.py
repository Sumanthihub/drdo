from huggingface_hub import from_pretrained_keras
from PIL import Image

import tensorflow as tf
import numpy as np
import requests
import torchvision.transforms as T
import cv2
from keras.models import load_model
import os


def denoise(file_path):

    image = Image.open(file_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))
    model = from_pretrained_keras("google/maxim-s3-denoising-sidd")
    # model.config.to_json_file("config.json")
    predictions = model.predict(tf.expand_dims(image, 0))
    predictions_array = np.array(predictions)
    pixel_values = (predictions_array * 255).astype(np.uint8)
    image = Image.fromarray(pixel_values)
    file_path = f'{os. getcwd()}/data/denoise.png'
    image.save(file_path)
    return file_path

def indor_dehaze(file_path):

    image = Image.open(file_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))

    model = from_pretrained_keras("google/maxim-s2-dehazing-sots-indoor")
    predictions = model.predict(tf.expand_dims(image, 0))
    print(predictions)

def outdoor_dehaze(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))

    model = from_pretrained_keras("google/maxim-s2-dehazing-sots-outdoor")
    predictions = model.predict(tf.expand_dims(image, 0))
    print(predictions)


def enchancement(file_path):


    image = Image.open(file_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (256, 256))

    model = from_pretrained_keras("google/maxim-s2-enhancement-lol")
    predictions = model.predict(tf.expand_dims(image, 0))
    print(predictions)


