import threading, traceback ,json, queue, cv2, pickle, boto3
import numpy as np
import awsiot.greengrasscoreipc.clientv2 as clientv2
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    IoTCoreMessage,
    QOS
)

BUCKET_NAME = 'jetson-projects'
s3 = boto3.client('s3')


# Class for Greengrass MQTT Connection
MQTT_TIMEOUT = 5
class StreamHandlerMQTT(client.SubscribeToIoTCoreStreamHandler):
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
    def on_stream_event(self, event: IoTCoreMessage) -> None:
        try:
            message = str(event.message.payload, "utf-8")
            topic_name = event.message.topic_name
            print(f"[StreamHandlerMQTT] Received message {message} on topic {topic_name}")
            self.queue.put(json.loads(message))
        except:
            traceback.print_exc()
    def on_stream_error(self, error: Exception) -> bool:
        # Handle error.
        return True  # Return True to close stream, False to keep stream open.
    def on_stream_closed(self) -> None:
        # Handle close.
        pass
class GreengrassMqttClient:
    def __init__(self, config: dict):
        self.ipc_client = clientv2.GreengrassCoreIPCClientV2()
        self.input_topic = config['genai_input_topic']
        self.output_topic = config['genai_output_topic']
        self.stream_handler = StreamHandlerMQTT()
        resp, self.operation = self.ipc_client.subscribe_to_iot_core(
            topic_name=self.input_topic,
            qos=QOS.AT_MOST_ONCE,
            stream_handler=self.stream_handler
        )
        self.queue = self.stream_handler.queue
    def publish_message(self, message: dict, output_type: str):
        resp = self.ipc_client.publish_to_iot_core(topic_name=self.output_topic+f"/{output_type}", qos=QOS.AT_LEAST_ONCE, payload=json.dumps(message))


# Camera Class for starting/stopping a camera
class CameraClient:
    def __init__(self, config: dict) -> None:
        self.camera_id = config['camera_id']
        self.cam = None
        if self.camera_id.isnumeric():
            self.camera_id = int(self.camera_id)
        self.cam = cv2.VideoCapture(self.camera_id)
        # Run dummy catures to reset camera
        self.frame_num = 0
        self.ret_encoded = False
        self.img_encoded = None
        self.frame = None
        self.stop_event = threading.Event()  # Event to signal when to stop
        self.frame_thread = threading.Thread(target=self._get_frame_continuously)
        self.frame_thread.start()
    def _get_frame_continuously(self):
        while not self.stop_event.is_set():
            self.get_frame()
    def get_frame(self):
        if self.camera_status:
            self.frame = self.cam.read()[1]
            if self.frame_num%2==0:
                self.ret_encoded, self.img_encoded = cv2.imencode('.jpg', self.frame)
                s3.put_object(Bucket=BUCKET_NAME, Key=f'edgegenai/frame_0.jpg', Body=self.img_encoded.tobytes())
            self.frame_num += 1
            if self.frame_num>=30:
                self.frame_num = 0
    def stop_camera(self) -> None:
        self.stop_event.set()
        self.frame_thread.join()
        self.cam.release()
    def camera_status(self):
        if not self.cam.isOpened():
            self.cam = cv2.VideoCapture(self.camera_id)
        return self.cam.isOpened()
    def get_frame_data(self, is_stream=True):
        if self.frame is None: return None, None
        if not self.ret_encoded: return None, None
        # Serialize image with pickle
        pickle_data = pickle.dumps(self.img_encoded)
        return pickle_data, self.img_encoded

def s3_save_fastersam(img_encoded, is_capture=False):
    if not is_capture:
        frame_bgr = cv2.imdecode(np.frombuffer(img_encoded, dtype=np.uint8), -1)
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Encode the image back to JPEG format
        ret_encoded, img_encoded = cv2.imencode('.jpg', frame_rgb)
        s3.put_object(Bucket=BUCKET_NAME, Key=f'fastersam/frame_0.jpg', Body=img_encoded.tobytes())
    else:
        s3.put_object(Bucket=BUCKET_NAME, Key=f'edgegenai/capture_frame_0.jpg', Body=img_encoded.tobytes())