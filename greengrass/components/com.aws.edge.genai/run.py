from utils import GreengrassMqttClient, CameraClient, s3_save_fastersam
import socket
import time
import argparse, json
import sys, cv2, pickle, base64, subprocess, psutil, re, time, os
from datetime import datetime


# Server configuration
SERVER_IP = '0.0.0.0'  # IP address of the server Docker container
PORT = 65432           # Exposed port of the server Docker container


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
    try:
        img_encoded = pickle.loads(data)
    except Exception as e:
        print(f"Error deserializing image: {e}")
        return None
    return img_encoded
# Function to receive text over a socket
def receive_text_from_socket(conn):
    # Receive text data
    data = conn.recv(4096)
    text = data.decode()
    return text


###
# MQTT messages:
# 1. Take a capture: {"event": "capture", "type": "nanovlm/fastersam"}
# 2. Run a query: {"event": "text", "text": "query text", "type": "nanovlm/fastersam"}
# 3. Reset: {"event": "reset", "type": "nanovlm/fastersam"}
###


EDGEGENAI_TYPE = "nanovlm"


def main(config : dict):
    ggClient = GreengrassMqttClient(config = config)
    cameraClient = CameraClient(config = config)

    event_status = None

    # Create a socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socketClient:

        # Connect to the server
        num_try = 0
        while True:
            try:
                socketClient.connect((SERVER_IP, PORT))
                print(f"Connected: {SERVER_IP}:{PORT}")
                break
            except Exception as e:
                print(f"[{num_try:2d}] Trying to connect ...")
                num_try += 1
                time.sleep(10)

        print(f"Running Processes ...")

        # Keep the main thread alive, or the process will exit
        while True:
            message_event = ggClient.queue.get()
            print(f"[Main] Received message {message_event}")
            if 'event' in message_event:
                # tegrastats_file = open( '/tmp/tegrastats.txt', mode='a')
                # tegrastats_file.write(f"START_{message_event['type']}_{message_event['event']}\n")
                # tegrastats_file.close()
                start_time = time.time()
                print(f"START {message_event['type']}-{message_event['event']} = {datetime.now().strftime('%H:%M:%S')} : {time.time()-start_time}")
                if message_event['event'].lower() == 'reset':
                    event_status = 'RESET'
                    print(f"[MAIN]: EdgeGenAI event: {event_status}")
                    socketClient.sendall(b"RST")
                elif message_event['event'].lower() == 'capture':
                    event_status = 'CAPTURE'
                    print(f"[MAIN]: EdgeGenAI event: {event_status}")
                    socketClient.sendall(b"IMG")
                    EDGEGENAI_TYPE = message_event['type'].lower()
                    if EDGEGENAI_TYPE!="nanovlm" and EDGEGENAI_TYPE!="fastersam": EDGEGENAI_TYPE = "nanovlm"
                    print(f"[MAIN]: EdgeGenAI event type: {EDGEGENAI_TYPE}")
                    if EDGEGENAI_TYPE=="nanovlm": socketClient.sendall(b"VLM")
                    elif EDGEGENAI_TYPE=="fastersam": socketClient.sendall(b"SAM")
                    pickle_data, img_encoded = cameraClient.get_frame_data()
                    socketClient.sendall(pickle_data)
                    if img_encoded is not None: s3_save_fastersam(img_encoded, is_capture=True)
                    # ggClient.publish_message(
                    #     message={
                    #         "data": -1,
                    #         "type": EDGEGENAI_TYPE
                    #     },
                    #     output_type="capture"
                    # )
                elif message_event['event'].lower() == 'text':
                    event_status = 'TEXT'
                    print(f"[MAIN]: EdgeGenAI event: {event_status}")
                    socketClient.sendall(b"TXT")
                    socketClient.sendall(message_event['text'].encode())
                    print(f"[MAIN]: EdgeGenAI event type: {EDGEGENAI_TYPE}")
                    if EDGEGENAI_TYPE=="nanovlm":
                        text_data = receive_text_from_socket(socketClient)
                        print("[NanoVLM] Received Text: ", text_data)
                        ggClient.publish_message(
                            message={
                                "data": text_data.split('</s>')[0],
                                "type": EDGEGENAI_TYPE
                            },
                            output_type=EDGEGENAI_TYPE
                        )
                    elif EDGEGENAI_TYPE=="fastersam":
                        img_encoded = receive_image_from_socket(socketClient)
                        print("[FasterSAM] Received Image Data")
                        if img_encoded is not None: s3_save_fastersam(img_encoded)
                        # ggClient.publish_message(
                        #     message={
                        #         "data": image_data,
                        #         "type": EDGEGENAI_TYPE
                        #     },
                        #     output_type=EDGEGENAI_TYPE
                        # )
                elif message_event['event'].lower() == 'stop':
                    event_status = 'STOP'
                    print(f"[MAIN]: EdgeGenAI event: {event_status}")
                    socketClient.close()
                    break
                # tegrastats_file = open( '/tmp/tegrastats.txt', mode='a')
                # tegrastats_file.write(f"STOP_{message_event['type']}_{message_event['event']}\n")
                # tegrastats_file.close()
                print(f"STOP  {message_event['type']}-{message_event['event']} = {datetime.now().strftime('%H:%M:%S')} : {time.time()-start_time}")

    ggClient.operation.close()
    ggClient.ipc_client.close()
    del ggClient
    del cameraClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print("[Main] Arguments - ", sys.argv[1:])
    parser.add_argument("--config", type=str,required = True)
    args = parser.parse_args()
    print("[Main] ConfigRaw - ", args.config)
    config = json.loads(args.config)
    print("[Main] ConfigJson - ", config)
    main(config)
