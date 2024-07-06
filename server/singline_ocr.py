import copy
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from utils import LayoutAnalyzer, OCRModel, fw_fill

from ultralytics import YOLO
import math
ocr_model = OCRModel(
    model_root_dir= Path("../models/") / "paddle-ocr", 
)
image = cv2.imread('Image11.png')    #'DIT_tb1.png'
cropped_img_np = np.array(image, dtype=np.uint8)
# cv2.imwrite('Image1.png',cropped_img_np )
print(cropped_img_np.shape)
ocr_results = ocr_model(cropped_img_np)[1]
print(ocr_results)
