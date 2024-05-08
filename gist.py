import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import io
import numpy as np
import torch
import torchvision
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

st.title('Gist image segmentation')
st.write('Upload image, draw on it, and get the segmentation image')

# Specify canvas parameters in application

bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg", "jpeg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("desired selection: ", "#8AE485", "#E67272" )
bg_color = st.sidebar.color_picker("NOT desired selection: ", "#E67272")

realtime_update = st.sidebar.checkbox("Update in realtime", True)


if bg_image is not None:
    image = Image.open(bg_image)
    h = image.height
    w = image.width
    image_RGB = np.array(Image.open(bg_image).convert('RGB'))
    predictor.set_image(image_RGB)


# Create a canvas component
if bg_image is not None:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=h,
        width=w,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)

    canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    buf = io.BytesIO()
    canvas_image.save(buf, format='PNG')
    byte_im = buf.getvalue()

    st.download_button(label="Download the painted picture",
                                   data=byte_im,
                                   file_name="painted_picture.png",
                                   mime="image/png")
