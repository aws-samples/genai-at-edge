import streamlit as st
import base64
import requests
import json
import time
from PIL import Image
from io import BytesIO

# Set the API Gateway URL
api_url = "API GATEWAY URL PLACEHOLDER"

st.set_page_config(layout="wide")

st.title("GenAI@Edge: FasterSAM")

st.write("Live Feed:")

# Load placeholder image
placeholder_image = Image.open('placeholder.jpg')

# Function to get the image from API Gateway
def get_image():
    response = requests.get(api_url + "/video")
    response = json.loads(response.content.decode('utf8'))
    if response["statusCode"] == 200:
        image_data = base64.b64decode(response["body"])
        return Image.open(BytesIO(image_data))
    else:
        return placeholder_image

# Function to send a POST request
def send_post_request(data):
    response = requests.post(api_url + "/mqtt", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to send POST request"}

# Live Feed Section
live_image_placeholder = st.image(get_image(), width=400)

# Initialize session state
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = placeholder_image
if 'segmented_image' not in st.session_state:
    st.session_state.segmented_image = placeholder_image

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to update the image
def update_image():
    while True:
        live_image_placeholder.image(get_image(), width=400)

with st.sidebar:
    capture_button = st.button("Capture")
    reset_button = st.button("Reset")


# Capture Section
col11, col12, col13 = st.columns(3, gap="small")
with col11:
    st.write("Captured Image:")
    captured_image_placeholder = st.image(st.session_state.captured_image, width=420)
    if capture_button:
        data = {
            "event": "capture",
            "type": "fastersam"
        }
        st.session_state.captured_image = placeholder_image
        response = send_post_request(data)
        if "error" not in response:
            st.session_state.captured_image = Image.open(BytesIO(base64.b64decode(response["body"])))
        captured_image_placeholder.image(st.session_state.captured_image, width=420)
    if reset_button:
        data = {
            "event": "reset",
            "type": "fastersam"
        }
        response = send_post_request(data)
        st.session_state.captured_image = placeholder_image
        captured_image_placeholder.image(st.session_state.captured_image, width=420)
# Segmented Image Section
with col12:
    st.write("Segmented Image:")
    segmented_image_placeholder = st.image(st.session_state.segmented_image, width=420)
    if capture_button:
        st.session_state.segmented_image = placeholder_image
        segmented_image_placeholder.image(st.session_state.segmented_image, width=420)
    if reset_button:
        st.session_state.segmented_image = placeholder_image
        segmented_image_placeholder.image(st.session_state.segmented_image, width=420)

# Chat Section
with col13:
    st.write("Prompt Box:")
    messages = st.container(height=320)

    if reset_button:
        st.session_state.messages = []
    if prompt := st.chat_input("Say something", key="fastersamchat"):
        messages.chat_message("user").write(prompt)
        data = {
            "event": "text",
            "text": prompt,
            "type": "fastersam"
        }
        response = send_post_request(data)
        if response["statusCode"] == 200:
            ai_response = Image.open(BytesIO(base64.b64decode(response["body"])))
            st.session_state.segmented_image = ai_response
            segmented_image_placeholder.image(st.session_state.segmented_image, width=420)
        else:
            st.session_state.segmented_image = placeholder_image
            segmented_image_placeholder.image(st.session_state.segmented_image, width=420)

# Start updating the live image in a separate thread
update_image()
