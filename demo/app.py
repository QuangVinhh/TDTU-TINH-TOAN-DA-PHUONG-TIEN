import streamlit as st
import io
import os
import torch

from PIL import Image
from io import BytesIO

#---page
st.set_page_config(
    page_title="DEMO",
    layout="centered",
)

#---model
@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def instantiate_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path = "model/last.pt", force_reload=True)
    model.eval()
    model.conf = 0.5
    model.iou = 0.45
    return model

#---download img
def download_success():
    st.success('Download Successful !')

upload_path = "uploads/"
download_path = "downloads/"
model = instantiate_model()

st.title('License Plate Detected')
uploaded_file = st.file_uploader("Upload Image of car's number plate", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"..."):
        uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
        downloaded_image = os.path.abspath(os.path.join(download_path,str("output_"+uploaded_file.name)))

        with open(uploaded_image,'rb') as imge:
            img_bytes = imge.read()

        img = Image.open(io.BytesIO(img_bytes))

        results = model(img, size=640)
        
        results.render()
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save(downloaded_image, format="JPEG")

        final_image = Image.open(downloaded_image)
        st.markdown("---")
        st.image(final_image)
        with open(downloaded_image, "rb") as file:
            
            if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
                if st.download_button(
                                        label="Download Output Image",
                                        data=file,
                                        file_name=str("output_"+uploaded_file.name),
                                        mime='image/jpg'
                                        ):
                    download_success()

            if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
                if st.download_button(
                                        label="Download Output Image",
                                        data=file,
                                        file_name=str("output_"+uploaded_file.name),
                                        mime='image/jpeg'
                                        ):
                    download_success()

            if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                if st.download_button(
                                        label="Download Output Image",
                                        data=file,
                                        file_name=str("output_"+uploaded_file.name),
                                        mime='image/png'
                                        ):
                    download_success()

else:
    st.warning('Please upload your Image')
