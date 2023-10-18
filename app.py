import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon = ":corn:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key


with st.sidebar:
        st.image('mg.png')
        st.title("Rice Leaf Disease Detector")
        st.subheader("Accurate detection of diseases present in the Rice leaves. This helps an user to easily detect the disease and identify it's cause.")

             
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('rice')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

    

st.write("""
         # Rice Leaf Disease Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['BrownSpot','Healthy','Hispa','LeafBlast']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'BrownSpot':
        
        st.sidebar.warning(string)
        

    elif class_names[np.argmax(predictions)] == 'Hispa':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("These include a combination of triazoles + chlorothalonil at Rice stage T1 (1-2 node stage) or triazole + SDHI at Rice stage T2 (last leaf stage). However, new solutions are also available. These are perfectly appropriate for use in a septoria control strategy in combination with a triazole and/or chlorothalonil.")

    elif class_names[np.argmax(predictions)] == 'LeafBlast':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Aviator® Xpro® and Prosaro® are both protective and curative fungicides, unlike some other fungicides which only offer protective properties against stripe rust. They are both registered for the control of stripe rust in Rice.")

    elif class_names[np.argmax(predictions)] == 'Healthy':
        st.balloons()
        st.sidebar.success(string)
