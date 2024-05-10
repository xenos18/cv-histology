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
from weights import predict, plot
from streamlit_javascript import st_javascript
#predictor = weights.predictor


st.sidebar.image('Biocad_Logo.svg',  width=100)
st.title('Hist image segmentation')
st.write('Upload image, draw on it, and get the segmentation image')

# Specify canvas parameters in application

bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg", "jpeg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw")
)


stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)


#stroke_color = st.sidebar.color_picker("desired selection: ", "#8AE485")
#bg_color = st.sidebar.color_picker("NOT desired selection: ", "#E67272")

realtime_update = st.sidebar.checkbox("Update in realtime", True)

if bg_image is not None:
    image = Image.open(bg_image)
    h = image.height
    w = image.width
    image_RGB = np.array(image.convert('RGB'))

# Create a canvas component
if bg_image is not None:
    inner_width = st_javascript("""await fetch("http://localhost:8501/").then(function(response) {
            return window.innerWidth;
    }) """)

    canvas_width = inner_width
    canvas_height = h * (canvas_width / w)

select_event = st.sidebar.selectbox('Какую точку поставить?',
                                        ['green', 'red'])
if select_event == 'red':
    canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="#E67272",
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )
    st.sidebar.color_picker("desired selection: ", "#E67272")

else:
    canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="#8AE485",
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
            )
    st.sidebar.color_picker("desired selection: ", "#8AE485")


    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        buf = io.BytesIO()
        canvas_image.save(buf, format='PNG')
        byte_im = buf.getvalue()

        st.download_button(label="Download the painted picture",
                           data=byte_im,
                           file_name="painted_picture.png",
                           mime="image/png")
    #st.button('Распознать')
    if st.button('Распознать'):
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        objects = objects[objects.type == 'circle']
        cords = pd.DataFrame({'x': objects['left'].apply(lambda x: int(x * w / inner_width)).astype("str"),
                              'y': objects['top'].apply(lambda x: int(x * w / inner_width)).astype("str")})

        st.dataframe(objects)
        unique_colors = objects['stroke'].unique()
        #objects['labels'] = objects['stroke'].apply(lambda a:  )
        objects['stroke'].replace('#8AE485', 1, inplace=True)
        objects['stroke'].replace('#E67272', 0, inplace=True)
        st.dataframe(objects)


        points = cords.to_numpy()
        labels = objects['stroke'].to_numpy()
        predict(image_RGB, points, labels)
        masks, scores, logits = predict(image_RGB, points, labels)
        maska = plot(image_RGB, masks, scores, points, labels)
        segm_im = np.array(Image.open('result.png'))
        st.image(segm_im)

        maska = (maska != 0).astype(int)
        im = Image.fromarray(np.uint8(maska * 255))

        # Save image to a bytes buffer instead of a file
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        buf.seek(0)

        # Use the buffered bytes data for the download button
        st.image(im)
        st.download_button(label="Download the mask",
                           data=buf,
                           file_name="mask.png",
                           mime="image/png")



    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        objects = objects[objects.type == 'circle']


        num_points = len(objects)


    arr = cords.to_numpy()
    input_point = arr
    input_label = np.array([1])

    #st.write(w, h, inner_width)

