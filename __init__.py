import cv2
import numpy as np
import torch
import time
import random
from PIL import Image

IMAGE_TYPE = "IMAGE"
STRING_TYPE = "STRING" 
LOGGING_KEY = "WebcamCapture: "

class WebcamCapture:
    def __init__(self):
        self.cam_on = False
        self.cam_id = -1
        self.cap = None

    RETURN_TYPES = (IMAGE_TYPE,)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def _ensure_cam_on(self, cam_id=0):
        if cam_id != self.cam_id and self.cap: # new cam id, close old one
            if self.cap.isOpened():
                self.cap.release()
            self.cam_on = False
        if not self.cam_on:
            try:
                self.cam_id = cam_id
                self.cap = cv2.VideoCapture(cam_id)
                # hack to ensure camera is awake and starting to autofocus and autoexpose
                time.sleep(0.5) 
                _, _ = self.cap.read() # throwing away the first frame seems to help with some webcams
                time.sleep(0.5) 
                self.cam_on = True
            except:
                self.cam_on = False
                raise IOError(f"{LOGGING_KEY}cannot open webcam")
            print(f"{LOGGING_KEY}started webcam")

    @classmethod
    def INPUT_TYPES(s):
        # seed is a workaround because there's no way to force comfy to rerun the node every time otherwise. https://github.com/comfyanonymous/ComfyUI/discussions/1804
        return {"required": {
                    "cam_id": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1,}),
                    "width": ("INT", {"default": 720, "min": 256, "max": 1920, "step": 8}),
                    "height": ("INT", {"default": 480, "min": 256, "max": 1080, "step": 8}),
                    "brightness": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.05}),
                    "exposure": ("INT", {"default": 0, "min": -10, "max": 10, "step": 1}),
                    "aperture": ("FLOAT", {"default": 1.8, "min": 1.6, "max": 16.0, "step": 0.2}),
                    "autoexp": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                    "autofocus": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                    "autowb": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,}),
                }
            }

    @classmethod
    def IS_CHANGED(s, **args):
        return True # always capture a new image

    def run(self, cam_id=0,width=720, height=480, brightness=1, exposure=0, aperture:float=1.8, autoexp:int=1, autofocus:int=1, autowb:int=0, seed:int=-1):
        self._ensure_cam_on(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.cap.set(cv2.CAP_PROP_APERTURE, aperture)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, autoexp)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, autowb)
        #cap.set(cv2.CALIB_FIX_FOCAL_LENGTH, focus)

        if not self.cap.isOpened():
            raise IOError(f"{LOGGING_KEY}Cannot open webcam")

        ret, frame = self.cap.read()
        if not ret:
            raise IOError(f"{LOGGING_KEY}Cannot read from webcam")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # debug: save image to file
        #pil_image = Image.fromarray(frame)
        #pil_image.save(f"{width}x{height} exp-{exposure} ap-{aperture} brght-{brightness} ae-{autoexp} awb-{autowb}, foc-{focus}.png")
        #pil_image.show()

        image = np.array(frame).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        print(f"{LOGGING_KEY}returning image shape {image_tensor.shape}")
        if height != image.shape[0] or width != image.shape[1]:
            print(f"{LOGGING_KEY}requested {height} x {width} but got {image.shape[0]}x{image.shape[1]}, you may be requesting an unsupported size for your webcam.")
        return (image_tensor,)

NODE_CLASS_MAPPINGS = {
    "WebcamCapture": WebcamCapture
}
