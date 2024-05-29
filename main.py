#!/usr/bin/env python3
import os
import time
import socket
import pickle
import cv2
from models import FasterSamModel
from models import NanoVlmModel

# Function to receive image over a socket
def receive_image_from_socket(conn):
    # Receive image data
    data = b""
    while True:
        packet = conn.recv(4096)
        data += packet
        if len(packet)<4096:
            break
    # Deserialize image with pickle
    img_encoded = pickle.loads(data)
    # Decode image
    img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    # Save image
    img_path = "/opt/genai-jetson/images/tmp.png"
    cv2.imwrite(img_path, img)
    return img_path

# Function to receive text over a socket
def receive_text_from_socket(conn):
    # Receive text data
    data = conn.recv(4096)
    text = data.decode()
    return text

# Server configuration
HOST = '0.0.0.0'
PORT = 65432

is_reset = True
edgegenai_type = None

fasterSamModel = FasterSamModel()
nanoVlmModel = NanoVlmModel()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    # Start listening for incoming connections
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # Receive header to determine type of data
                header = conn.recv(3)

                if header == b"IMG":
                    print(f"{header}: Receiving image ...")
                    edgegenai_type = conn.recv(3)
                    image_path = receive_image_from_socket(conn)
                    is_reset = False
                    print(f"{edgegenai_type}: Received image: {image_path}")
                    if edgegenai_type == b"VLM":
                        nanoVlmModel.image_init(image_path)
                    elif edgegenai_type == b"SAM":
                        fasterSamModel.image_init(image_path)
                    continue
                elif header == b"TXT":
                    print(f"{header}: Receiving text ...")
                    user_prompt = receive_text_from_socket(conn)
                    print(f"{edgegenai_type}: Received text: {user_prompt}")
                    if edgegenai_type == b"VLM":
                        results = nanoVlmModel.inference(user_prompt)
                    elif edgegenai_type == b"SAM":
                        results = fasterSamModel.inference(user_prompt)
                    conn.sendall(results)
                elif header == b"RST":
                    print("{edgegenai_type}: Resetting chat history")
                    is_reset = True
                    if edgegenai_type == b"VLM":
                        nanoVlmModel.reset()
                    elif edgegenai_type == b"SAM":
                        fasterSamModel.reset()
                    edgegenai_type = None
                    continue
                else:
                    time.sleep(0.05)
                    continue
