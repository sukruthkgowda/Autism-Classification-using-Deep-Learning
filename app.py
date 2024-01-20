import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import pandas as pd
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
global op1,op2,op3,op4,op5

st.set_page_config(layout='wide')





html_temp = """ 
<h1 style='font-family:sans-serif;color:white;position:relative;top:-50px;font-size:65px;opacity:0.8'>
AUTISM CLASSIFIER</h1><br/>
<h1 style='font-family:helvetica;color:white;position:relative;top:-100px;font-size:25px;opacity:0.6;margin-left:7.25%;'>Tool To detect Autism in Children!</h1>
""" 
st.markdown(html_temp, unsafe_allow_html=True) 
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('Autism_Classification.h5')
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()

file_head="""
<h1 style='font-family:helvetica;color:white;position:relative;top:-25px;margin-bottom:-50px;font-size:20px;opacity:0.5;'>Please Upload Image with the Person's Face to Detect Autism.</h1>
"""
st.markdown(file_head,unsafe_allow_html=True)
file = st.file_uploader("", type=["jpg", "png","jpeg"])
import cv2
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

if file is None:
  st.markdown("<h1 style='font-family:helvetica;color:white;position:relative;top:-10px;margin-bottom:-50px;font-size:20px;opacity:0.5;'>Please upload an image file within the allotted file size</h1>",unsafe_allow_html=True)
# else:
#   img = Image.open(file)
#   st.image(img, use_column_width=False)
#   size = (224,224)    
#   image = ImageOps.fit(img, size, Image.ANTIALIAS)
#   imag = np.asarray(image)
#   imaga = np.expand_dims(imag,axis=0) 
#   predictions = model.predict(imaga)
#   a=predictions[0][0]
#   if a==0:
#     st.error('AUTISM.')
#     #st.error('The subject under observation may have the symptoms of AUTISM. Please ensure that you consult with a professional before pursuing any kinds of treatment.')
#     #op1= 1
#   elif a==1:
#     st.success('Non AUTISM.') 
#     #op1= 0
#     #st.warning('the model is only 85% accurate. this is the beta version of the model. Futher enhancements has to made to get the best results.')
else:
  img = Image.open(file)
  col1, col2, col3 = st.columns(3)

  with col1:
    st.write(' ')

  with col2:
    st.image(img,width=400,use_column_width=False)

  with col3:
    st.write(' ')
  
  size = (224,224)    
  image = ImageOps.fit(img, size, Image.ANTIALIAS)
  imag = tf.keras.utils.img_to_array(image)
  imaga = np.expand_dims(imag,axis=0)
  predictions = model.predict(imaga)
  a=predictions[0][0]
  if a==0:
    st.error('The person in Image suffers from Autism.')
    #st.error('The subject under observation may have the symptoms of AUTISM. Please ensure that you consult with a professional before pursuing any kinds of treatment.')
    #op1= 1
  elif a==1:
    st.success('The person in Image does Not suffer from Autism.') 
    #op1= 0
    #st.warning('the model is only 85% accurate. this is the beta version of the model. Futher enhancements has to made to get the best results.')
