#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import numpy as np
import tensorflow as tf
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.initializers import RandomNormal
from keras.models import Model
from keras import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils import plot_model
import numpy
from PIL import Image, ImageOps
import os
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import models, layers
from tensorflow.keras import backend as K
import pywt
from numpy import asarray
from tensorflow.keras.utils import CustomObjectScope
@st.cache(allow_output_mutation=True)
class DWT(layers.Layer):
    """
    Discrete Wavelet transform - tensorflow - keras
    inputs:
        name - wavelet name ( from pywavelet library)
        concat - 1 - merge transform output to one channel
               - 0 - split to 4 channels ( 1 img in -> 4 smaller img out)
    """

    def __init__(self, wavelet_name='haar', concat=1, **kwargs):
        super().__init__()
        # self._name = self.name + "_" + name
        # get filter coeffs from 3rd party lib
        wavelet = pywt.Wavelet(wavelet_name)
        self.dec_len = wavelet.dec_len
        self.concat = concat
        # decomposition filter low pass and hight pass coeffs
        db2_lpf = wavelet.dec_lo
        db2_hpf = wavelet.dec_hi

        # covert filters into tensors and reshape for convolution math
        db2_lpf = tf.constant(db2_lpf[::-1])
        self.db2_lpf = tf.reshape(db2_lpf, (1, wavelet.dec_len, 1, 1))

        db2_hpf = tf.constant(db2_hpf[::-1])
        self.db2_hpf = tf.reshape(db2_hpf, (1, wavelet.dec_len, 1, 1))

        self.conv_type = "VALID"
        self.border_padd = "SYMMETRIC"
        self.wavelet_name = wavelet_name
        self.concat = concat

    def build(self, input_shape):
        # filter dims should be bigger if input is not gray scale
        if input_shape[-1] != 1:
            # self.db2_lpf = tf.repeat(self.db2_lpf, input_shape[-1], axis=-1)
            self.db2_lpf = tf.keras.backend.repeat_elements(self.db2_lpf, input_shape[-1], axis=-1)
            # self.db2_hpf = tf.repeat(self.db2_hpf, input_shape[-1], axis=-1)
            self.db2_hpf = tf.keras.backend.repeat_elements(self.db2_hpf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # border padding symatric add coulums
        inputs_pad = tf.pad(inputs, [[0, 0], [0, 0], [self.dec_len-1, self.dec_len-1], [0, 0]], self.border_padd)

        # approximation conv only rows
        a = tf.nn.conv2d(
            inputs_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # details conv only rows
        d = tf.nn.conv2d(
            inputs_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ds - down sample
        a_ds = a[:, :, 1:a.shape[2]:2, :]
        d_ds = d[:, :, 1:d.shape[2]:2, :]

        # border padding symatric add rows
        a_ds_pad = tf.pad(a_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)
        d_ds_pad = tf.pad(d_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)

        # convolution is done on the rows so we need to
        # transpose the matrix in order to convolve the colums
        a_ds_pad = tf.transpose(a_ds_pad, perm=[0, 2, 1, 3])
        d_ds_pad = tf.transpose(d_ds_pad, perm=[0, 2, 1, 3])

        # aa approximation approximation
        aa = tf.nn.conv2d(
            a_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ad approximation details
        ad = tf.nn.conv2d(
            a_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ad details aproximation
        da = tf.nn.conv2d(
            d_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # dd details details
        dd = tf.nn.conv2d(
            d_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )

        # transpose back the matrix
        aa = tf.transpose(aa, perm=[0, 2, 1, 3])
        ad = tf.transpose(ad, perm=[0, 2, 1, 3])
        da = tf.transpose(da, perm=[0, 2, 1, 3])
        dd = tf.transpose(dd, perm=[0, 2, 1, 3])

        # down sample
        ll = aa[:, 1:aa.shape[1]:2, :, :]
        lh = ad[:, 1:ad.shape[1]:2, :, :]
        hl = da[:, 1:da.shape[1]:2, :, :]
        hh = dd[:, 1:dd.shape[1]:2, :, :]

        # concate all outputs ionto tensor
        if self.concat == 0:
            x = tf.concat([ll, lh, hl, hh], axis=-1)
        elif self.concat == 2:
            x = ll
        else:
            x = tf.concat([tf.concat([ll, lh], axis=1), tf.concat([hl, hh], axis=1)], axis=2)
        return x
    def get_config(self):
        config = super(DWT, self).get_config()
        config.update({'wavelet_name': self.wavelet_name, 'concat': self.concat})
        return config

tf.keras.utils.get_custom_objects().update({'DWT': DWT})
# with CustomObjectScope({'DWT': DWT}):


# import wget

# # Replace the MODEL_LINK with your Google Drive model link
import gdown
url1 = "https://drive.google.com/uc?id=1Lx9rVKdBtKVC2Iu0jyBCm2V4ol0FJ9iw"
output1 = "lesion_model_000296.h5"
if not os.path.exists("lesion_model_000296.h5"):
    gdown.download(url1, output1, quiet=False)
# url2 = "https://drive.google.com/uc?id=1WUoZ4f18ssh8v1CK5ItxMmnvZ8chi-Ln"
# output2 = "background_model_000296.h5"
# gdown.download(url2, output2, quiet=False)
# import subprocess
# import gd_download
# def load_model():

#     save_dest = tf.constant('model')
#     tf.io.gfile.makedirs(save_dest)
    
#     f_checkpoint = tf.constant("model/fmodel.h5")

#     if not tf.io.gfile.exists(f_checkpoint):
#         with tf.compat.v1.Session() as sess:
#             from gd_download import download_file_from_google_drive
#             download_file_from_google_drive("https://drive.google.com/uc?id=1Lx9rVKdBtKVC2Iu0jyBCm2V4ol0FJ9iw", f_checkpoint)
    
#     model = tf.keras.models.load_model(f_checkpoint)
#     model.compile()
#     return model
# if not os.path.isfile('fmodel.h5'):
#     subprocess.run(['wget','-O', 'fmodel.h5','https://drive.google.com/uc?id=1Lx9rVKdBtKVC2Iu0jyBCm2V4ol0FJ9iw'])
# if not os.path.isfile('bmodel.h5'):
#     subprocess.run(['wget', '-O', 'bmodel.h5' ,'https://drive.google.com/uc?id=1WUoZ4f18ssh8v1CK5ItxMmnvZ8chi-Ln'])
# if not os.path.isfile('fmodel.h5'):
#     subprocess.run(['curl --output fmodel.h5 "https://drive.google.com/file/d/1nC5HdXt7mY-i7tDUP14GlksjTNfhp4eJ/view"'], shell=True)
# if not os.path.isfile('bmodel.h5'):
#     subprocess.run(['curl --output bmodel.h5 "https://drive.google.com/file/d/15Gk_JrkyVPPK9cTVLd6nBK561iDUjie9/view"'], shell=True)
# subprocess.run(["gdown", "https://drive.google.com/file/d/1nC5HdXt7mY-i7tDUP14GlksjTNfhp4eJ/view", "-O", "fmodel.h5"])
# subprocess.run(["gdown", "https://drive.google.com/file/d/15Gk_JrkyVPPK9cTVLd6nBK561iDUjie9/view", "-O", "bmodel.h5"])
# Replace MODEL_ID with the ID of your Google Drive file

# fmodel = tf.keras.models.load_model(fore_model_path)
# bmodel = tf.keras.models.load_model(back_model_path)    


    
fmodel = tf.keras.models.load_model("lesion_model_000296.h5")
    
# bmodel = tf.keras.models.load_model("background_model_000296.h5")



# fore_model_path = wget.download("https://drive.google.com/file/d/1nC5HdXt7mY-i7tDUP14GlksjTNfhp4eJ/view?usp=sharing",out="lesion_model_000296.h5")
# back_model_path = wget.download("https://drive.google.com/file/d/15Gk_JrkyVPPK9cTVLd6nBK561iDUjie9/view?usp=share_link",out="background_model_000296.h5")

# fmodel = tf.keras.models.load_model(fore_model_path)
# bmodel = tf.keras.models.load_model(back_model_path)    
# Download the model file using wget
# wget.download("https://drive.google.com/file/d/1nC5HdXt7mY-i7tDUP14GlksjTNfhp4eJ/view?usp=sharing",out="lesion_model_000296.h5")
# wget.download("https://drive.google.com/file/d/15Gk_JrkyVPPK9cTVLd6nBK561iDUjie9/view?usp=share_link",out="background_model_000296.h5")

# fmodel = tf.keras.models.load_model("lesion_model_000296.h5")
# bmodel = tf.keras.models.load_model("background_model_000296.h5")    
def preprocess_image(image):
#     image = tf.image.grayscale_to_rgb(image)

    image = np.array(image)
    image = np.array([image,image,image])
    image = (image.astype('float32')-127.5) / 127.5
    image = np.expand_dims(image, axis=0)
    #     image = tf.reshape(image,[1,256,256,3])
    return image

def predict(image, model):
    image = preprocess_image(image)
    image = tf.reshape(image,[1,256,256,3])
    fmask = fmodel.predict(image)
#     bmask = bmodel.predict(image)
    fmask = (fmask+1)/2
    fmask = np.squeeze(fmask, axis=0)
#     bmask = (1-bmask)/2
#     bmask = np.squeeze(bmask, axis=0)
#     mask = np.logical_or(fmask,bmask)
    mask = (fmask > 0.5).astype(np.uint8)*255 
    return tf.reshape(mask,[256,256,3])

def main():
    # Set the app title and description
    st.title("Brain Hemorrhage Lesion Segmenter from CT Images")
    st.markdown("This app uses a deep learning model to perform brain hemorrhage lesion segmentation.")

    # Load the model
#     model = load_model()

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    # Check if an image is uploaded
    if uploaded_file is not None:
        # Read the image and display it
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make a prediction and display the mask
        mask = predict(image, fmodel)
        st.image(mask, caption='Segmentated Lesion', use_column_width=True)

if __name__ == '__main__':
    main()

