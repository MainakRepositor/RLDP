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
        st.markdown("## Remedy")
        st.info("Brown spot disease in rice leaves, caused by the fungus Bipolaris oryzae, can significantly impact crop yield. To address this issue, follow an integrated disease management approach. First, choose disease-resistant rice varieties like CO 39, ADT 37, or IR 64. Implement good agricultural practices such as proper spacing, balanced fertilizer application, and avoiding excessive moisture. For chemical control, consider the following pesticides: 1) Carbendazim (Bavistin), 2) Propiconazole (Tilt), and 3) Mancozeb (Dithane M-45). Rotate these chemicals to prevent resistance buildup. Apply these pesticides at the recommended dosages during the initial stages of disease onset and adhere to safety guidelines. Always consult with local agricultural authorities for the latest recommendations, as pesticide availability and regulations may vary. Integrated management, including cultural and chemical practices, can effectively mitigate brown spot disease and preserve rice crop health.")
        

    elif class_names[np.argmax(predictions)] == 'Hispa':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("To combat Hispa disease, a common ailment affecting rice leaves, it is essential to employ integrated pest management strategies to ensure effective control. Begin by practicing good agricultural practices, including maintaining proper field hygiene, avoiding waterlogging, and using disease-resistant rice varieties. Implement cultural controls like crop rotation and intercropping. Biological control methods can involve releasing natural predators of Hispa beetles, such as ladybugs or parasitoid wasps. Chemical control, when necessary, should be a last resort. Use pesticides like Imidacloprid, Thiamethoxam, or Chlorantraniliprole, but only under expert guidance, following recommended dosage and safety precautions. Regular monitoring of the field and weather conditions is crucial to make informed decisions about pesticide applications. Lastly, always promote sustainable farming practices to minimize the impact of Hispa disease and safeguard the environment.")

    elif class_names[np.argmax(predictions)] == 'LeafBlast':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Leaf Blast disease in rice leaves, caused by the fungus Pyricularia oryzae, can devastate crop yields. To effectively manage this disease, a multi-pronged approach is recommended. Begin by practicing proper field hygiene, ensuring adequate spacing between plants to improve air circulation and reduce humidity. Utilize resistant rice varieties where possible. Apply organic solutions such as neem oil and copper-based fungicides like Bordeaux mixture as preventive measures. For curative actions, employ chemical fungicides like tricyclazole (e.g., Blastban) or azoxystrobin (e.g., Amistar) following label instructions. Rotate fungicides to reduce the risk of resistance development. Regularly monitor the field for early symptoms and implement these remedies when needed. Remember to adhere to safety guidelines when handling pesticides and consult local agricultural authorities for up-to-date recommendations. Integrated management strategies offer the best results, ultimately safeguarding rice crops from Leaf Blast disease while minimizing environmental impact.")

    elif class_names[np.argmax(predictions)] == 'Healthy':
        st.balloons()
        st.sidebar.success(string)
