import base64
import os
from typing import Annotated

import cv2 as cv
import numpy as np
from litestar import Litestar, MediaType, Response, Router
from litestar import get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.config.cors import CORSConfig

from .modules.model_module import ResUNet, DoubleConv, Down, Up, OutConv, Model
from .modules.post_processing_module import ImageProcessor
from .modules.pre_processing_module import ImageCropper

# Workaround for unpickling saved PyTorch model
import __main__

setattr(__main__, 'ResUNet', ResUNet)
setattr(__main__, 'DoubleConv', DoubleConv)
setattr(__main__, 'Down', Down)
setattr(__main__, 'Up', Up)
setattr(__main__, 'OutConv', OutConv)

# Constants
MODEL_FOLDER = "./app/models"
CROPPER_HEIGHT = 120
CROPPER_WIDTH = 120

# Initialize components
img_cropper = ImageCropper(CROPPER_HEIGHT, CROPPER_WIDTH)
cors_config = CORSConfig(allow_origins=["*"])
model_dict = {}


@get("/", media_type=MediaType.JSON)
async def get_models() -> Response:
    model_files = os.listdir(MODEL_FOLDER)
    model_names = [model_file.split(".")[0] for model_file in model_files]
    obj = {
        "models": model_names
    }

    return Response(content=obj, media_type=MediaType.JSON)


@post(path="/{model_name:str}/predict", media_type=MediaType.JSON)
async def handle_model_predict(
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
        model_name: str
) -> Response:
    try:
        content = await data.read()
        if model_name not in model_dict:
            model_dict[model_name] = load_model(model_name)
        model = model_dict[model_name]

        raw_img_array = np.frombuffer(content, np.uint8)
        img = cv.imdecode(raw_img_array, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")

        cropped_imgs, cropped_cords = img_cropper.crop_image(img)
        masks = model.predict(cropped_imgs)

        img_processor = ImageProcessor(
            img.shape[0], img.shape[1], masks, cropped_cords)
        combined_img = img_processor.process_cropped_images()
        combined_img_rgb = combined_img * 255

        retval, buffer = cv.imencode('.png', combined_img_rgb)
        if not retval:
            raise ValueError("Failed to encode image")

        img_base64 = base64.b64encode(buffer).decode()
        return Response(content={"image": img_base64}, media_type=MediaType.JSON)

    except Exception as e:
        return Response(content={"error": str(e)}, status_code=400, media_type=MediaType.JSON)


def load_model(model_name: str) -> Model:
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pth")
    return Model(model_path)


models_route = Router(path="/models", route_handlers=[get_models, handle_model_predict])
app = Litestar(path="/api", route_handlers=[models_route], cors_config=cors_config)
