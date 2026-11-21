from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
model=load_model('Digit Classifier.keras')
st.title('Handwritten Digit Classifier')
st.info("**Note :** This model is trained on black and white image data so can make a mistake.")
image=st.file_uploader("Upload Image",type=['png','jpg','jpeg'])
st.divider('---')
import streamlit as st
cols=st.columns(2)
with cols[0]:
  background=st.selectbox("Background Color :",["Black","White"])
with cols[1]:
  type=st.selectbox("Image Type",["RGB","GrayScale"])
def identify(file,background_color,image_type):
  img=Image.open(file)
  if image_type=="RGB":
    img=img.convert('L')
  img=img.resize((28,28))
  img=np.array(img)
  if background_color=="White":
    img=255-img
  img=img/255
  pred=model.predict(img.reshape(1,28,28,1))
  pred=np.argmax(pred)
  return pred
if image is not None:
  pred=identify(image,background,type)
  st.image(image,"Identify Digit")
  st.success(f"Predicted Value for digit : {pred}")
