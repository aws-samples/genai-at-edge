from huggingface_hub import snapshot_download
import requests, shutil
from subprocess import call

from faster_sam.image_utils import ImageUtils
from faster_sam.fastsam_utils import FastSAM
from faster_sam.vit_utils import VIT

import time, os, cv2, numpy as np, pickle
from termcolor import cprint
from nano_llm.models import MLCModel
from nano_llm import ChatHistory


def download_file(url, destination):
    with requests.get(url, stream=True) as r:
        with open(destination, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


ChatTemplates = {
    'vicuna-v1': {
        'system_prompt': "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        'system': '${MESSAGE}\n\n',
        'user': 'USER: ${MESSAGE}\n',
        'bot': 'ASSISTANT: ${MESSAGE}\n', # TODO: does output already end in </s> ?
    }
}


class FasterSamModel():
    def __init__(self):
        self.download_models()
        sam_model = "models/yolov8s-seg.trt"
        vit_image_model = "models/onnx32-open_clip-ViT-B-32-openai-visual.trt"
        vit_text_model = "models/onnx32-open_clip-ViT-B-32-openai-textual.trt"
        self.imageutils = ImageUtils()
        self.fastsam = FastSAM(model_path=sam_model)
        self.vit = VIT(image_model_path=vit_image_model, text_model_path=vit_text_model)
        self.image = None
    def download_models(self):
        if not os.path.exists('models/onnx32-open_clip-ViT-B-32-openai-visual.trt'):
            os.system('/usr/src/tensorrt/bin/trtexec --onnx=models/onnx32-open_clip-ViT-B-32-openai-visual.onnx --saveEngine=models/onnx32-open_clip-ViT-B-32-openai-visual.trt')
        if not os.path.exists('models/onnx32-open_clip-ViT-B-32-openai-textual.trt'):
            os.system('/usr/src/tensorrt/bin/trtexec --onnx=models/onnx32-open_clip-ViT-B-32-openai-textual.onnx --saveEngine=models/onnx32-open_clip-ViT-B-32-openai-textual.trt')
        if not os.path.exists('models/yolov8s-seg.trt'):
            os.system('/usr/src/tensorrt/bin/trtexec --onnx=models/yolov8s-seg.onnx --saveEngine=models/yolov8s-seg.trt')
        os.system('rm models/*.pt')
        os.system('rm models/*.onnx')
    def image_init(self, image_path):
        # add the latest user prompt to the chat history
        _, _, image_in_BGR, _ = self.imageutils.read_image_path(image_path)
        self.image = image_in_BGR
        self.boxes, self.masks, self.scores = self.fastsam.infer(self.image)
        print(f"[FasterSamModel]: Image Received by model")
    def inference(self, user_prompt):
        print(f"[FasterSamModel]: Text Received by model")
        self.vit.image = self.image
        print(f"[FasterSamModel]: Running SAM model")
        results = self.vit.format_results(self.boxes, self.masks, self.scores, 0)
        annotations = self.vit.prompt(results, text=True, text_prompt=user_prompt)
        annotations = np.array([annotations])
        results = self.imageutils.fast_process(annotations=annotations)
        ret_encoded, img_encoded = cv2.imencode('.jpg', results)
        pickle_data = pickle.dumps(img_encoded)
        print(f"[FasterSamModel]: Sending response back")
        return pickle_data
    def reset(self):
        print(f"[FasterSamModel]: Stopping SAM model")
        self.image = None


class NanoVlmModel():
    def __init__(self):
        model_path = "models/VILA-2.7b"
        vision_model = "models/clip-vit-large-patch14-336"
        api = "mlc"
        self.quantization = "q4f16_ft"
        self.vision_scaling = "resize"
        self.max_context_len = 768
        self.max_new_tokens = 128
        self.min_new_tokens = -1
        self.wrap_tokens = 512
        self.reply_color = 'green'
        self.prompt_color = 'blue'
        self.top_p = 0.95
        self.temperature = 0.7
        self.do_sample = True
        self.repetition_penalty = 1.0
        load_begin = time.perf_counter()
        kwargs = {}
        kwargs['name'] = os.path.basename(model_path)
        kwargs['api'] = api
        kwargs['vision_model'] = vision_model
        self.model = MLCModel(model_path, **kwargs)
        self.model.init_vision(**kwargs)
        self.model.config.load_time = time.perf_counter() - load_begin
        self.chat_template = 'vicuna-v1'
        self.system_prompt = ChatTemplates[self.chat_template]['system_prompt']
        self.chat_history = ChatHistory(self.model, self.chat_template, self.system_prompt)
    def image_init(self, image_path):
        # add the latest user prompt to the chat history
        entry = self.chat_history.append(role='user', msg=image_path)
        print(f"[NanoVlmModel]: Image Received by model")
    def inference(self, user_prompt):
        print(f"[NanoVlmModel]: Text Received by model")
        # add the latest user prompt to the chat history
        entry = self.chat_history.append(role='user', msg=user_prompt)
        # get the latest embeddings (or tokens) from the chat
        embedding, position = self.chat_history.embed_chat(
            max_tokens=self.model.config.max_length - self.max_new_tokens,
            wrap_tokens=self.wrap_tokens,
            return_tokens=False,
        )
        # generate bot reply
        print(f"[NanoVlmModel]: Running VLM model")
        reply = self.model.generate(
            embedding,
            streaming=True,
            kv_cache=self.chat_history.kv_cache,
            stop_tokens=self.chat_history.template.stop,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        for token in reply:
            cprint(token, self.reply_color, end='', flush=True)
        self.chat_history.append(role='bot', text=reply.text)
        self.chat_history.kv_cache = reply.kv_cache
        print(f"[NanoVlmModel]: Sending response back")
        return reply.text.encode()
    def reset(self):
        print(f"[NanoVlmModel]: Stopping VLM model")
        self.chat_history.reset()
        self.chat_history = ChatHistory(self.model, self.chat_template, self.system_prompt)
