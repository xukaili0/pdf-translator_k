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
class InputPdf(BaseModel):
    """Input PDF file."""

    input_pdf: UploadFile = Field(..., title="Input PDF file")
import fitz  # PyMuPDF

# 设置当前文件所在目录作为工作目录，这样可以使用相对路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DPI = 300
FONT_SIZE = 8
font_name = 'SerifCN'
font_file_path = "./font/SourceHanSerifCN-Medium.otf"  # 或 .otf 文件

font_bold_name = 'SerifCNbold'
font_bold_file_path ="./font/SourceHanSerifCN-Bold.otf" # "./font/SourceHanSerifCN-Bold.otf"  # 或 .otf 文件

from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
det_processor, det_model = segformer.load_processor(), segformer.load_model()
rec_model, rec_processor = load_model(), load_processor()

class CustomTextWrapper(TextWrapper):
    def __init__(self, en_width=4.3, cn_width=7.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_width = en_width
        self.cn_width = cn_width

    def _measure(self, s):
        """Calculate the total width of the string."""
        return sum(self.cn_width if '\u4e00' <= char <= '\u9fff' else self.en_width for char in s)

    def _split(self, text):
        """Split text into individual characters."""
        return list(text)

    def wrap(self, text, width_rec):
        lines = []
        current_line = []
        current_width = 0

        for char in self._split(text):
            if '\u4e00' <= char <= '\u9fff' or char in ['。', '，', '、', '%', '（', '）','：','；']:
                char_width = self.cn_width

            elif '\u0041' <= char <= '\u005A': # 大写字符
                char_width = 5.8
            else :
                char_width = self.en_width

            if current_width + char_width > width_rec:
                lines.append(''.join(current_line))
                current_line = [char]
                current_width = char_width
            else:
                current_line.append(char)
                current_width += char_width

        if current_line:
            lines.append(''.join(current_line))

        return '\n'.join(lines)


class TranslateApi:
    """Translator API class.

    Attributes
    ----------
    app: FastAPI
        FastAPI instance
    temp_dir: tempfile.TemporaryDirectory
        Temporary directory for storing translated PDF files
    temp_dir_name: Path
        Path to the temporary directory
    font: ImageFont
        Font for drawing text on the image
    layout_model: PPStructure
        Layout model for detecting text blocks
    ocr_model: PaddleOCR
        OCR model for detecting text in the text blocks
    translate_model: MarianMTModel
        Translation model for translating text
    translate_tokenizer: MarianTokenizer
        Tokenizer for the translation model
    """

    
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
        self.DPI = DPI
        self.empity_ocr_result = 0
        self.__load_models(model_root_dir)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = Path(self.temp_dir.name)

    def run(self):
        """Run the API server"""
        uvicorn.run(self.app, host="0.0.0.0", port=8765)

    async def translate_pdf(self, input_pdf: UploadFile = File(...)) -> FileResponse:
        """API endpoint for translating PDF files.

        Parameters
        ----------
        input_pdf: UploadFile
            Input PDF file

        Returns
        -------
        FileResponse
            Translated PDF file
        """
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
        self, pdf_path_or_bytes: Union[Path, bytes], output_dir: Path
    ) -> None:
        """Backend function for translating PDF files.

        Translation is performed in the following steps:
            1. Convert the PDF file to images
            2. Detect text blocks in the images
            3. For each text block, detect text and translate it
            4. Draw the translated text on the image
            5. Save the image as a PDF file
            6. Merge all PDF files into one PDF file

        At 3, this function does not translate the text after
        the references section. Instead, saves the image as it is.

        Parameters
        ----------
        pdf_path_or_bytes: Union[Path, bytes]
            Path to the input PDF file or bytes of the input PDF file
        output_dir: Path
            Path to the output directory
        """

        if isinstance(pdf_path_or_bytes, Path):
            pdf_images = convert_from_path(pdf_path_or_bytes, dpi=self.DPI)
        else:
            pdf_images = convert_from_bytes(pdf_path_or_bytes, dpi=self.DPI)

        import pdfplumber

        with pdfplumber.open(pdf_path_or_bytes) as pdf:
            for i, page in enumerate(pdf.pages):
                width = page.width
                height = page.height
                print(f"PDF Page {i + 1} dimensions: {page.width} x {page.height} points")

        # 使用PIL的Image模块打开图像
        # with Image.open(pdf_images) as img_np:
            # 获取图像的宽度和高度，单位为像素
            # width, height = img_np.size
            
            # # 打印图像尺寸
            # print(f"Image size: {width} pixels x {height} pixels")
        pdfdoc = fitz.open(pdf_path_or_bytes)

        # font_id = pdfdoc.embed_font(fitz.FONT_TYPE_OPENTYPE, font_file_path)
        # pdfdoc.subset_fonts()
        # 使用 embed_font 方法加载字体
        self.font = fitz.Font(fontname=font_name, fontfile=font_file_path)  
        # subset_font = self.font.get_subset()      
        pdf_files = []
        reached_references = False
        for i, image in tqdm(enumerate(pdf_images)):
            pdf_page = pdfdoc[i]
            self.width_img, self.height_img = image.size
            self.img_pdf_scale = self.height_img / pdf_page.rect.height

            print(f" page {i} scale {self.img_pdf_scale} Image: {self.width_img} pixels x {self.height_img} pixels   pdf height {pdf_page.rect.height}")
            output_path = output_dir / f"{i:03}.pdf"
            if not reached_references:
                img_np, original_img_np, reached_references = self.__translate_one_page(
                    image=image,
                    reached_references=reached_references,
                    pdf_page = pdf_page
                )
                fig, ax = plt.subplots(1, 2, figsize=(20, 14))
                ax[0].imshow(original_img_np)
                ax[1].imshow(img_np)
                ax[0].axis("off")
                ax[1].axis("off")
                plt.tight_layout()
                plt.savefig(output_path, format="pdf", dpi=self.DPI)
                plt.close(fig)
            else:
                (
                    image.convert("RGB")
                    .resize((int(1400 / image.size[1] * image.size[0]), 1400))
                    .save(output_path, format="pdf")
                )

            pdf_files.append(str(output_path))
        pdfdoc.subset_fonts(fallback=True, verbose=False)
        pdfdoc.ez_save('out soyar '+str(pdf_path_or_bytes), garbage=3, deflate=True, clean = True, deflate_fonts = True)
        self.__merge_pdfs(pdf_files)

    def __load_models(self, model_root_dir: Path, device: str = "cuda"):
        """Backend function for loading models.

        Called in the constructor.
        Load the layout model, OCR model, translation model and font.

        Parameters
        ----------
        model_root_dir: Path
            Path to the directory containing the models.
        device: str
            Device to use for the layout model.
            Defaults to "cuda". 
        """
        self.font = ImageFont.truetype(
            str(model_root_dir / "SourceHanSerif-Light.otf"),
            size=FONT_SIZE,
        )
        self.device = device

        # self.layout_model = LayoutAnalyzer(
        #     model_root_dir=model_root_dir / "unilm", device=self.device
        # )
        # self.ocr_model = OCRModel(
        #     model_root_dir=model_root_dir / "paddle-ocr", device=self.device
        # )

        # self.translate_model = MarianMTModel.from_pretrained("staka/fugumt-en-ja").to(
        #     self.device
        # )
        # self.translate_tokenizer = MarianTokenizer.from_pretrained("staka/fugumt-en-ja")

    def __translate_one_page(
        self,
        image: Image.Image,
        pdf_page,
        reached_references: bool,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Translate one page of the PDF file.

        There are some heuristics to clean-up the results of translation:
            1. Remove newlines, tabs, brackets, slashes, and pipes
            2. Reject the result if there are few Japanese characters
            3. Skip the translation if the text block has only one line

        Parameters
        ----------
        image: Image.Image
            Image of the page
        reached_references: bool
            Whether the references section has been reached.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, bool]
            Translated image, original image,
            and whether the references section has been reached.
        """
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
            "YOLOv8m Model": "yolov8m-doclaynet.pt",
            "YOLOv8n Model": "yolov8n-doclaynet.pt",
            "YOLOv8s Model": "yolov8s-doclaynet.pt",
            "YOLOv8x_full Model": "best.pt",
        }
        yolomodel = YOLO(model_paths["YOLOv8x Model"])


        print('image',image)
        img_np = np.array(image, dtype=np.uint8)
        original_img_np = copy.deepcopy(img_np)
        # result = self.layout_model(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        imgsize = 35
        results = yolomodel(source=img_np, save=False, show_labels=True, show_conf=True, show_boxes=True,agnostic_nms = True, iou = 0.3)#iou = 0 ,   iou = 0.7  ,,imgsz = imgsize*32

        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en",det_db_box_thresh	= 0.1,use_dilation = True, det_db_score_mode='slow',det_db_unclip_ratio=1.8 ,det_limit_side_len=1600
                        # rec_model_dir='./rec_svtr_tiny_none_ctc_en_train',
                        # --rec_char_dict_path=
                        #rec_algorithm = rec_algorithm1,
                        )  # need to run only once to download and load model into memory

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
                label = yolomodel.names[int(box.cls[0])]

##########################  PDF   ########################
                 # 定义矩形区域
                # # 在指定区域填充矩形
                # pdf_page.insert_shape(rect, fill={255,255,255}, even_odd=False)

                # # 定义矩形的坐标（左下角的x和y，宽度，高度）
                # rect_coordinates = (pdf_pos[0], pdf_pos[1], pdf_pos[2], pdf_pos[3])  # 左下角的x, 左下角的y, 宽度, 高度

                # # 创建一个Shape对象，用于绘制
                # shape = pdf_page.new_shape()

                # # 设置填充颜色，例如红色
                # shape.set_fill((1, 0, 0))  # RGB颜色，红色

                # # 绘制矩形并填充
                # shape.rect(*rect_coordinates)  # *操作符用于解包坐标元组
                # shape.finish(fill=True)  # 填充矩形
                # # 将绘制的内容添加到页面
                # pdf_page.draw_shape(shape)

                pdf_pos = [(x / self.img_pdf_scale) for x in xyxy]
                rect = fitz.Rect(pdf_pos[0], pdf_pos[1], pdf_pos[2], pdf_pos[3])            

                if (classsify_name in ["Section-header","Text","List-item","Caption"]) or ((classsify_name in ["Page-header"]) and ((xyxy[3]-xyxy[1])/(xyxy[2]-xyxy[0])) < 5 ):  #, 
                    # # 绘制矩形
                    # pdf_page.draw_rect(rect, color=(1,0,0), width=1,fill=(0.4,0.2,0.3))  # 边框宽度为2
                    # print(f"____ Class: {classsify_name}, Conf: {conf:.2f}, BBox: {xyxy} label {label}")   
                    xy_ = 4    
                    cropped_img = image.crop((math.floor(xyxy[0]-xy_), math.floor(xyxy[1]-xy_), math.ceil(xyxy[2]+xy_), math.ceil(xyxy[3]+xy_)))

                    # cropped_img_np = np.array(cropped_img, dtype=np.uint8)
                    # cv2.imwrite(f'./out/Image_{classsify_name}_{conf:.2f}.png',cropped_img_np )

                    ################# paddle ocr #####################
                    # ocrrs = ocr.ocr(cropped_img_np,cls=False)[0]
                    # ocr_results = [item[1][0] for item in ocrrs]

                    ################# soyar  ocr #####################

                    predictions = run_ocr([cropped_img], [['en']], det_model, det_processor, rec_model, rec_processor)
                    ocr_results = [box.text for box in predictions[0].text_lines]

                    ##################################################  

                    text_join = " ".join(ocr_results)
                    area_ch = ((pdf_pos[2]-pdf_pos[0])*(pdf_pos[3]-pdf_pos[1])) / len(text_join)
                    # height_line = (xyxy[3]-xyxy[1])/len(ocr_results)
                    print(f'\n~~~ ocr_num {len(ocr_results)} h {(xyxy[3]-xyxy[1]):.2f}')

                    # 在图像上绘制边界框和标签
                    image_orig = results[0].orig_img  # 获取原始图像
                    
                    cv2.rectangle(image_orig, (math.floor(xyxy[0]-xy_), math.floor(xyxy[1]-xy_)), (math.ceil(xyxy[2]+xy_), math.ceil(xyxy[3]+xy_)), color_map[classsify_name], 3)
                    cv2.putText(image_orig, classsify_name+ f" {conf:.2f} {(pdf_pos[2]-pdf_pos[0]):.2f}", (int(xyxy[0]), int(xyxy[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = color_map[classsify_name], thickness = 2)  # 绘制标签 
                    cv2.imwrite(f'./out yolo .png', image_orig) #[:, :, ::-1]  imgsize_test/{DPI}_{imgsize}_3.png
                    #############


                    if len(ocr_results) >= 1:
                        text = " ".join(ocr_results)
                        text = re.sub(r"\n|\/|\|", " ", text)
                        # translated_text = text
                        translated_text = self.__translate_llm(text)
                        # print('translated_text',translated_text)

                        wrapper = CustomTextWrapper()
                        processed_text = wrapper.wrap(translated_text,(pdf_pos[2]-pdf_pos[0]))

                        # processed_text = fw_fill(
                        #     translated_text,
                        #     width=int((pdf_pos[2] - pdf_pos[0]) / 3.5)    #4.5 3.7                            
                        # )
                        print('+++ processed_text\n',processed_text,'\n\n')

                        if classsify_name in ["Section-header"]:
                            pdf_page.insert_text((pdf_pos[0], pdf_pos[1]+0.88*FONT_SIZE), processed_text, fontsize=FONT_SIZE+1, color=[0, 0, 0] ,fontfile = font_bold_file_path, fontname= font_bold_name, lineheight = 1.7)    
                        elif classsify_name in ["List-item"] and area_ch < 33.5:

                            processed_text = fw_fill(translated_text, width=int((pdf_pos[2] - pdf_pos[0]) / 3.4))

                            pdf_page.insert_text((pdf_pos[0], pdf_pos[1]+0.88*FONT_SIZE), processed_text, fontsize=FONT_SIZE-1,color=[0, 0, 0] ,fontfile = font_file_path, fontname=font_name , lineheight = 1.2)    
                        else:    
                            pdf_page.insert_text((pdf_pos[0], pdf_pos[1]+0.88*FONT_SIZE), processed_text, fontsize=FONT_SIZE,color=[0, 0, 0] ,fontfile = font_file_path, fontname=font_name , lineheight = 1.6)    

                        # pdf_page.insert_textbox(rect = rect,buffer = translated_text,  fontsize=6,color=[0, 0, 1])    #fontname=self.font, 
                    elif len(ocr_results) == 0:
                        self.empity_ocr_result += 1
                        print("+++++++++ empity ",self.empity_ocr_result)
                # elif line.type == "title":
                #     try:
                #         title = self.ocr_model(line.image)[1][0][0]
                #     except IndexError:
                #         continue
                #     if title.lower() == "references" or title.lower() == "reference":
                #         reached_references = True
        # pdf_page
        return img_np, original_img_np, reached_references

    def __translate(self, text: str) -> str:
        texts = self.__split_text(text, 448)

        translated_texts = []
        for i, t in enumerate(texts):
            inputs = self.translate_tokenizer(t, return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.translate_model.generate(inputs, max_length=512)
            res = self.translate_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # skip weird translations
            if res.startswith("「この版"):
                continue

            translated_texts.append(res)
        return "".join(translated_texts)
    
    def __translate_llm(self, text: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key = "f0cbdf6cc3bc6ed11100f087a327e3e1.TFeSjBzErpykvnFh",
            base_url = "https://open.bigmodel.cn/api/paas/v4/"
        )
        model_name = "GLM-4-Air"

        system_prompt = """
        You are an advanced chinese translation assistant,  Your task is to:

        - Translate the content as precisely as possible.
        - Preserve original formatting such as tables, spacing, punctuation, numbers.
        - If content is like reference, please translate title only
        Please translate the following text into Chinese, do not output any unrelated infomation such as tips and warnings if text is numbers,do not translate that

        """

        # print('---',model_name,api_key,base_url)        

        response = client.chat.completions.create(
        model=model_name,
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        )
        translation = response.choices[0].message.content
        print('+++',text,'+++')
        # print('~~~',translation,'~~~')

        return translation




    def __split_text(self, text: str, text_limit: int = 448) -> List[str]:
        """Split text into chunks of sentences within text_limit.

        Parameters
        ----------
        text: str
            Text to be split.
        text_limit: int
            Maximum length of each chunk. Defaults to 448.

        Returns
        -------
        List[str]
            List of text chunks,
            each of which is shorter than text_limit.
        """
        if len(text) < text_limit:
            return [text]

        sentences = text.rstrip().split(". ")
        sentences = [s + ". " for s in sentences[:-1]] + sentences[-1:]
        result = []
        current_text = ""
        for sentence in sentences:
            if len(current_text) + len(sentence) < text_limit:
                current_text += sentence
            else:
                if current_text:
                    result.append(current_text)
                while len(sentence) >= text_limit:
                    # better to look for a white space at least?
                    result.append(sentence[: text_limit - 1])
                    sentence = sentence[text_limit - 1 :].lstrip()
                current_text = sentence
        if current_text:
            result.append(current_text)
        return result

    def __merge_pdfs(self, pdf_files: List[str]) -> None:
        """Merge translated PDF files into one file.

        Merged file will be stored in the temp directory
        as "translated.pdf".

        Parameters
        ----------
        pdf_files: List[str]
            List of paths to translated PDF files stored in
            the temp directory.
        """
        pdf_merger = PyPDF2.PdfMerger()

        for pdf_file in sorted(pdf_files):
            pdf_merger.append(pdf_file)
        pdf_merger.write(self.temp_dir_name / "translated.pdf")



if __name__ == "__main__":
    translate_api = TranslateApi()
    # translate_api.run()

    # 定义输出目录
    output_directory = Path('./out')

    # 调用_translate_pdf函数，传入PDF文件路径
    pdf_file_path = Path('tb4.pdf')
    print(pdf_file_path)
    translate_api._translate_pdf(pdf_file_path, output_directory)