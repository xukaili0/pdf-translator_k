import copy
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Union
import os
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
# import os
from textwrap import TextWrapper

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sys.path.append(str(Path(__file__).parent.parent))

from utils import fw_fill   #LayoutAnalyzer OCRModel, 

from ultralytics import YOLO
import math

import fitz  # PyMuPDF

# 设置当前文件所在目录作为工作目录，这样可以使用相对路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DPI = 300
FONT_SIZE = 8

# font_name = 'SerifCN'
# font_file_path = "./font/SourceHanSerifCN-Medium.ttf" 


font_name = 'Simibold'
font_file_path = "./font/SourceHanSerifCN-SemiBold.ttf" 

font_bold_name = 'SerifCNbold'
font_bold_file_path ="./font/SourceHanSerifCN-Bold.ttf" 

# font_bold_name = 'Heavy'
# font_bold_file_path ="./font/SourceHanSerifCN-Heavy.ttf" 

# from surya.ocr import run_ocr
# from surya.model.detection import model
# from surya.model.recognition.model import load_model
# from surya.model.recognition.processor import load_processor
# det_processor, det_model = model.load_processor(), model.load_model()
# rec_model, rec_processor = load_model(), load_processor()

class CustomTextWrapper(TextWrapper):
    def __init__(self, en_width=3.5, cn_width=7.9, *args, **kwargs):  # en_width=4, cn_width=7.9,
        super().__init__(*args, **kwargs)
        self.en_width = en_width
        self.cn_width = cn_width

    def _measure(self, s):
        """Calculate the total width of the string."""
        return sum(self.cn_width if '\u4e00' <= char <= '\u9fff' else self.en_width for char in s)

    def _split(self, text):
        """Split text into individual characters."""
        return list(text)


class TranslateApi:
    
    def __init__(self, model_root_dir: Path = Path("../models/")):
        self.app = FastAPI()
        self.app.add_api_route(
            "/translate_pdf/",
            self.translate_pdf,
            methods=["POST"],
            response_class=FileResponse,
        )
        self.app.add_api_route(
            "/clear_temp_dir/",
            self.clear_temp_dir,
            methods=["GET"],
        )
        self.insertbox_count = 0

        self.DPI = DPI
        self.empity_ocr_result = 0
        self.usage_tokens = 0
        self.total_tokens = 0
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = Path(self.temp_dir.name)

    def run(self):
        """Run the API server"""
        uvicorn.run(self.app, host="0.0.0.0", port=8765)

    async def translate_pdf(self, input_pdf: UploadFile = File(...)) -> FileResponse:
        input_pdf_data = await input_pdf.read()
        self._translate_pdf(input_pdf_data, self.temp_dir_name)

        return FileResponse(
            self.temp_dir_name / "translated.pdf", media_type="application/pdf"
        )

    async def clear_temp_dir(self):
        """API endpoint for clearing the temporary directory."""
        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = Path(self.temp_dir.name)
        return {"message": "temp dir cleared"}

    def _translate_pdf(
        self, pdf_name, data_dir, output_dir ,all_pages=False, specific_pages_list_1 = None
    ) -> None:
        pdf_path = data_dir + pdf_name + '.pdf'
        pdf_images = convert_from_path(pdf_path, dpi=self.DPI)

        pdfdoc = fitz.open(pdf_path)
        pdfdoc_copy = fitz.open(pdf_path)
        self.font = fitz.Font(fontname=font_name, fontfile=font_file_path)  
        # subset_font = self.font.get_subset()      
        print('pdf_===',len(pdfdoc), len(pdf_images))
        if all_pages == True:
            specific_pages_list = list(range(len(pdfdoc)))
        else:
            specific_pages_list = [n-1 for n in specific_pages_list_1]

        for i in specific_pages_list:
            pdf_page = pdfdoc[i]
            image = pdf_images[i]
            self.width_img, self.height_img = image.size
            self.img_pdf_scale = self.height_img / pdf_page.rect.height

            print(f" page {i} scale {self.img_pdf_scale} Image: {self.width_img} pixels x {self.height_img} pixels   pdf height {pdf_page.rect.height}")
            output_path = output_dir + f"{i:03}.pdf" 

            img_np = self.__translate_one_page(
                image=image,
                pdf_page = pdf_page,
                No = i,
                pdfname = pdf_name,
                pdf_page2 = pdfdoc_copy[i]
            )

        # if not os.path.exists(output_dir + 'out'):
        #     os.makedirs(output_dir + 'out')

        pdfdoc.subset_fonts(fallback=True, verbose=False)
        pdfdoc.save(output_dir + 'out_' + pdf_name + '.pdf', garbage=3, deflate=True, clean = True, deflate_fonts = True) 
        print(f' ************  {pdf_path}  total tokens  {self.total_tokens} ****** {self.insertbox_count} ********')

    def get_text_info(self, pdf_page2,rect):

        blocks_text = pdf_page2.get_textbox(rect)
        blocks_text = blocks_text.replace("\n", " ")
        blocks_text = blocks_text.replace("- ", "")
        # blocks_text = blocks_text.replace(' ', '\xa0')  # 注意'\xa0'是U+00A0的转义序列


        blocks2 = pdf_page2.get_text("dict", clip = rect)
        # print(blocks2)
        # print(blocks['blocks']) #[0]['lines']
        fontsize_ = []
        textjoin = []
        fontbold_ = fonttype_all = 0
        for block in blocks2['blocks']:  # this is a text block
            # print('+++++++++++',block)
            for l in block['lines']:  # iterate through the text lines
                for s in l["spans"]:  # iterate through the text spans 
                    # print(s['size'], fitz.sRGB_to_rgb(s['color']), s['font'],f'_{s["text"]}_')  
                    fontsize_.append(s['size'])
                    
                    if 'bold' in s['font'].lower() or 'medi' in s['font'].lower():
                        fontbold_ = fontbold_+1
                    fonttype_all = fonttype_all + 1
        # elif:
        if len(blocks_text)==0 and fonttype_all == 0:
            flag_null = True
            fonttype_all = 1 # 防止除0
            fontsize_block = 1
        else:
            flag_null = False
            fontsize_block = max(set(fontsize_), key=fontsize_.count)
                     
        return flag_null, blocks_text, fontsize_block, fontbold_/fonttype_all
    
    def translate_easyword(self, s):
        import string
        word_dict = {
            "abstract": "摘要",
            "introduction": "引言",
            "background": "背景",
            "proposed": "提出的",
            "methodology": "方法",
            "method": "方法",
            "proposed": "提出的",
            "experiment": "实验",
            "case": "例子",
            "discussion": "讨论",
            "conclusion": "结论",
            "conclusions": "结论",
            "acknowledgment": "鸣谢",
            "declarations": "声鸣",
            "reference": "参考文献",
            "references": "参考文献",
        }
        ishave = False
        if len(s) < 20:
            words = s.translate(str.maketrans('', '', string.punctuation)).split()  # 将字符串分割成单词列表
            if len(words) <= 2:
                for word in words:
                    # 检查单词是否在字典中，并替换
                    if word.lower() in word_dict:
                        ishave = True
                        s = s.replace(word, word_dict[word.lower()])   #word_dict[word]
        return s, ishave

    def __translate_one_page(
        self,
        image: Image.Image,
        pdf_page,
        No,      # 当前的序号,
        pdfname,
        pdf_page2
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        color_map = {
            "Caption": (191, 100, 21),
            "Footnote": (2, 62, 115),
            "Formula": (140, 80, 58),
            "List-item": (168, 181, 69),
            "Page-footer": (2, 69, 84),
            "Page-header": (83, 115, 106),
            "Picture": (255, 72, 88),
            "Section-header": (0, 204, 192),
            "Table": (116, 127, 127),
            "Text": (0, 153, 221),
            "Title": (196, 51, 2)
        }
        model_paths = {
            "YOLOv8x Model": "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt",
            "YOLOv8x_full Model": "best.pt",
        }
        yolomodel = YOLO(model_paths["YOLOv8x Model"])

        img_np = np.array(image, dtype=np.uint8)
        # original_img_np = copy.deepcopy(img_np)
        # result = self.layout_model(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        results = yolomodel(source=img_np, save=False, show_labels=True, show_conf=True, show_boxes=True,agnostic_nms = True, iou = 0.2)#iou = 0 ,   iou = 0.7  ,,imgsz = imgsize*32

        # from paddleocr import PaddleOCR
        # ocr = PaddleOCR(use_angle_cls=True, lang="en",det_db_box_thresh	= 0.1,use_dilation = True, det_db_score_mode='slow',det_db_unclip_ratio=1.8 ,det_limit_side_len=1600
                        # rec_model_dir='./rec_svtr_tiny_none_ctc_en_train',
                        # --rec_char_dict_path=
                        #rec_algorithm = rec_algorithm1,
                        # )  # need to run only once to download and load model into memory
        # import time
        # pdffile.save('./out/out_1' + '.pdf', garbage=3, deflate=True, clean = True, deflate_fonts = True) 

        for result in results:
            boxes = result.boxes  # 包含边界框信息

            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # 转换为列表，包含(x1, y1, x2, y2)坐标
                classsify_name = yolomodel.names[int(box.cls.item())]  # 类别名称

                pdf_pos = [(x / self.img_pdf_scale) for x in xyxy]
                rect = fitz.Rect(pdf_pos[0], pdf_pos[1], pdf_pos[2], pdf_pos[3])   
                if classsify_name in["Section-header","Text","List-item","Caption","Page-header"]:  #, 
                    #############
                    annot = pdf_page.add_redact_annot(rect)
                    annot.set_colors(stroke=(1, 0, 0), fill=(0.97, 0.99, 0.95))  # 设置边框和填充颜色
                    annot.update()  # 必须更新注释以应用更改
                    pdf_page.apply_redactions()
                    #############

        for result in results:
            boxes = result.boxes  # 包含边界框信息
            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # 转换为列表，包含(x1, y1, x2, y2)坐标
                conf = box.conf.item()  # 置信度
                cls = box.cls.item()  # 类别ID
                classsify_name = yolomodel.names[int(cls)]  # 类别名称
                # label = yolomodel.names[int(box.cls[0])]

                if (classsify_name in ["Section-header","Text","List-item","Caption"]) or ((classsify_name in ["Page-header"]) and ((xyxy[3]-xyxy[1])/(xyxy[2]-xyxy[0])) < 5 ):  #, 
                    # cropped_img = image.crop((math.floor(xyxy[0]-xy_), math.floor(xyxy[1]-xy_), math.ceil(xyxy[2]+xy_), math.ceil(xyxy[3]+xy_)))

                    # cropped_img_np = np.array(cropped_img, dtype=np.uint8)
                    # cv2.imwrite(f'./out/Image_{classsify_name}_{conf:.2f}.png',cropped_img_np )

                    ################# paddle ocr #####################
                    # ocrrs = ocr.ocr(cropped_img_np,cls=False)[0]
                    # ocr_results = [item[1][0] for item in ocrrs]

                    ################# soyar  ocr #####################

                    # predictions = run_ocr([cropped_img], [['en']], det_model, det_processor, rec_model, rec_processor)
                    # ocr_results = [box.text for box in predictions[0].text_lines]

                    ##################################################  
                    # text_join = " ".join(ocr_results)


                    image_orig = results[0].orig_img  # 获取原始图像
                    xy_ = 4                       
                    cv2.rectangle(image_orig, (math.floor(xyxy[0]-xy_), math.floor(xyxy[1]-xy_)), (math.ceil(xyxy[2]+xy_), math.ceil(xyxy[3]+xy_)), color_map[classsify_name], 3)
                    cv2.putText(image_orig, classsify_name+ f" {conf:.2f} {(pdf_pos[2]-pdf_pos[0]):.2f}", (int(xyxy[0]), int(xyxy[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = color_map[classsify_name], thickness = 2)  # 绘制标签 
                    path_img = f'./out/layout/ollama_{pdfname}'
                    if not os.path.exists(path_img):
                        os.makedirs(path_img)
                    cv2.imwrite(f'{path_img}/{pdfname}_{No+1} .png', image_orig) 
                    #############


                    # if len(ocr_results) >= 1:
                    #     text = " ".join(ocr_results)
                    #     text = re.sub(r"\n|\/|\|", " ", text)
                    #     # translated_text = text
                    #     print('\n--- ',text)

                    #     translated_text = text
                        # print('+++ ',translated_text)

                        # print(f'\n___ completion {self.usage_tokens.completion_tokens} prompt {self.usage_tokens.prompt_tokens} total {self.usage_tokens.total_tokens} \n\n\n')

                    pdf_pos = [(x / self.img_pdf_scale) for x in xyxy]
                    rect = fitz.Rect(pdf_pos[0], pdf_pos[1]+1, pdf_pos[2], pdf_pos[3]-1 )            
                    flag_null, text_block, fontsize_block, font_bold_ratio = self.get_text_info(pdf_page2,rect)
                    if flag_null == True: 
                        print('\n$$$$$$$$$$$$ skip  ')

                        continue

                    print('\n\n\n----------',text_block)

                    test_easy, flag_easyword = self.translate_easyword(text_block)
                    # print(flag_easyword)
                    if flag_easyword ==True:
                        text_block = test_easy
                    else:

                        if text_block and text_block[0].isdigit():  #类似标题 2.4
                        # 使用第一个空格分割字符串
                            parts = text_block.split(" ", 1)
                            if len(parts)>1: #类似于 2.3 apple ， 否则为纯数字 ，不进行翻译

                                trans_part1 , self.usage_tokens = self.__translate_llm(parts[1])
                                self.total_tokens = self.total_tokens + self.usage_tokens.total_tokens
                                text_block = parts[0] +'  '+ trans_part1
                        elif re.match(r'^\[(\d+)\]', text_block):  #类似reference [2] afd
                        # 使用第一个空格分割字符串

                            parts = text_block.split(" ", 1)
                            print(parts)
                            if len(parts)>1: #类似于 2.3 apple ， 否则为纯数字 ，不进行翻译

                                trans_part1 , self.usage_tokens = self.__translate_llm(parts[1])
                                self.total_tokens = self.total_tokens + self.usage_tokens.total_tokens
                                text_block = parts[0] +'  '+ trans_part1
                        else:
                            text_block , self.usage_tokens = self.__translate_llm(text_block)
                            self.total_tokens = self.total_tokens + self.usage_tokens.total_tokens
                    text_block = text_block.replace("\n", "   ")
                    text_block = text_block.replace(' ', '\xa0')  # 注意'\xa0'是U+00A0的转义序列

                    # print('\n',,'\n',)
                    
                    print(f'########### fontsize {fontsize_block} {font_bold_ratio} \n{text_block}')
                    if font_bold_ratio > 0.5:
                        insertbox_count = pdf_page.insert_textbox(rect = (pdf_pos[0]-3, pdf_pos[1]-3, pdf_pos[2]+3, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_bold_name,fontfile = font_bold_file_path ,fontsize = fontsize_block-0.4, color=[0, 0, 0], lineheight = 1.22)    
                    else:
                        if fontsize_block > 20 or fontsize_block < 8.5:
                            insertbox_count = pdf_page.insert_textbox(rect = (pdf_pos[0]-3, pdf_pos[1]-3, pdf_pos[2]+3, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_name,fontfile = font_file_path ,fontsize = fontsize_block-0.8, color=[0, 0, 0], lineheight = 1.15)    #fontname=self.font, 
                        else:    
                            insertbox_count = pdf_page.insert_textbox(rect = (pdf_pos[0]-3, pdf_pos[1]-3, pdf_pos[2]+3, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_name,fontfile = font_file_path ,fontsize = fontsize_block-0.4, color=[0, 0, 0], lineheight = 1.26)    #fontname=self.font, 
                    if insertbox_count < 0:
                        self.insertbox_count += 1

                    # print('^^^^^^^^^^^',insertbox_count)
                    # self.insertbox_count = insertbox_count + self.insertbox_counts
                    # pdf_page.insert_text((pdf_pos[0], pdf_pos[1]+0.88*FONT_SIZE), text_block, fontsize=FONT_SIZE-2,color=[0, 0, 0] ,fontfile = font_file_path, fontname=font_name , lineheight = 1.2)    

                    # page.insert_textbox(rect, modified_text, fontsize=font_size-4,fontname = 'simsunbold' , color = colr, fontfile = "../server/font/SourceHanSerifCN-SemiBold.ttf") 

                    # self.total_tokens = 1
                        # if classsify_name in ["Section-header"]:
                        #     pdf_page.insert_text((pdf_pos[0], pdf_pos[1]+0.88*FONT_SIZE), processed_text, fontsize=FONT_SIZE+1, color=[0, 0, 0] ,fontfile = font_bold_file_path, fontname= font_bold_name, lineheight = 1.7)    
                        # elif classsify_name in ["List-item"]: #and area_ch < 33.5:

                        #     processed_text = fw_fill(translated_text, width=int((pdf_pos[2] - pdf_pos[0]) / 3))

                        #     pdf_page.insert_text((pdf_pos[0], pdf_pos[1]+0.88*FONT_SIZE), processed_text, fontsize=FONT_SIZE-2,color=[0, 0, 0] ,fontfile = font_file_path, fontname=font_name , lineheight = 1.2)    
                        # else:    
                        #     pdf_page.insert_text((pdf_pos[0], pdf_pos[1]+0.88*FONT_SIZE), processed_text, fontsize=FONT_SIZE+2,color=[0, 0, 0] ,fontfile = font_file_path, fontname=font_name , lineheight = 2)    

                    # elif len(ocr_results) == 0:
                    #     self.empity_ocr_result += 1
                    # print("+++++++++ empity ",self.empity_ocr_result)
        # pdf_page
        return img_np


    def __translate_llm(self, text: str) -> str:
        from openai import OpenAI
        # client = OpenAI(
        #     api_key = "f0cbdf6cc3bc6ed11100f087a327e3e1.TFeSjBzErpykvnFh",
        #     base_url = "https://open.bigmodel.cn/api/paas/v4/"
        # )
        # model_name = "GLM-4-Air"

        # client = OpenAI(
        #     api_key = "f10e8607d7b6494c814684171d6b9c23",
        #     base_url = "https://api.lingyiwanwu.com/v1"
        # )
        # model_name = "yi-medium"

        client = OpenAI(
            api_key = "ollama",
            base_url = "http://localhost:11434/v1/"
        )
        ollama = 0

        if ollama == 0:
            model_name = "ali-qwen-8b:latest"
        if ollama == 1:
            model_name = "dol-qwen-8b:latest"
        elif ollama == 2:
            model_name = "secstate-gemma2:latest"
        elif ollama == 3:
            model_name = "hfl_llama3_chinese_ins_v3:latest"

        
        # client = OpenAI(
        #     api_key = "lm-studio",
        #     base_url = "http://localhost:1234/v1/"
        # )
        # lmstudio = 3
        # if lmstudio == 1:
        #     model_name = "cognitivecomputations/dolphin-2.9.2-qwen2-7b-gguf"
        # elif lmstudio == 2:
        #     model_name = "second-state/Gemma-2-9B-Chinese-Chat-GGUF"
        # elif lmstudio == 3:
        #     model_name = "hfl_chinese_instruct_v3/Repository"

        system_prompt = """
 你是一位高级的英汉翻译助手，您的任务是精准地将下列文本或单词翻译成中文，对于人名、项目编号、数字、数学符号、和标点符号要保留原文中的格式不要翻译。请翻译下面的文本或单词，直接显示输出翻译的结果。不要输出任何提示，注意，和警告信息。
    """
    # You are an advanced english to chinese translation assistant,  Your task is to translate the following text or word into Chinese precisely. preserve original formatting and do not translate such as people name, item numbers, maths, latex and punctuation. Please translate the following text or word ,output translation directly and do not output attention tips and warnings.         

# 翻译结果保留原来排版格式，不要添加换行，空格等
        response = client.chat.completions.create(
        model=model_name,
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        )
        translation = response.choices[0].message.content
        usage_tokens = response.usage 
        # print('+++',text,'+++')
        # print('~~~',translation,'~~~')

        return translation, usage_tokens
 

if __name__ == "__main__":
    translate_api = TranslateApi()
    # translate_api.run()

    # 定义输出目录
    out_directory ='./out/ollama_aliqwen_cn_' #data/
    data_directory ='./data/' #data/

    # 调用_translate_pdf函数，传入PDF文件路径
    pdf_name = 'CMTFNet_CNN_and_Multiscale_Transformer_Fusion_Network_for_Remote-Sensing_Image_Semantic_Segmentation'
    translate_api._translate_pdf(pdf_name, data_directory, out_directory,all_pages=0, specific_pages_list_1 = [2])