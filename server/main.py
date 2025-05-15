from pathlib import Path
from typing import List, Tuple, Union
import os, io, re, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image, ImageDraw, ImageFont
# import os
from datetime import datetime, timedelta

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
import math, time
import fitz  # PyMuPDF
import concurrent.futures
# 设置当前文件所在目录作为工作目录，这样可以使用相对路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))
font_name = 'Simibold'
font_file_path = "./font/SourceHanSerifCN-SemiBold.ttf" 

font_bold_name = 'SerifCNbold'
font_bold_file_path ="./font/SourceHanSerifCN-Bold.ttf" 

model_paths = {
    "YOLOv8x Model": "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt",
    "YOLOv8x_full Model": "best.pt",
}

# yolomodel = YOLO(model_paths["YOLOv8x Model"])
yolomodel = YOLO("yolov10x_best.pt")  # YOLOV10

# font_bold_name = 'Heavy'
# font_bold_file_path ="./font/SourceHanSerifCN-Heavy.ttf" 

# from surya.ocr import run_ocr
# from surya.model.detection import model
# from surya.model.recognition.model import load_model
# from surya.model.recognition.processor import load_processor
# det_processor, det_model = model.load_processor(), model.load_model()
# rec_model, rec_processor = load_model(), load_processor()
insertbox_count = 0
picture_figure = True  # 在有的pdf阅读器上，图片中的线条形状会错乱，所以直接保存为图片。
DPI = 200
usage_tokens = 0
total_tokens = 0
model_select = 5
def convert_math_alpha_to_ascii(text):
  """
  将 Unicode 数学字母符号转换为对应的 ASCII 字母。
  """
  result = []
  for char in text:
    codepoint = ord(char)
    # print(hex(ord(char)))
    if 0x1D434 <= codepoint <= 0x1D44D:
      ascii_codepoint = codepoint - 0x1D3F3 
      result.append(chr(ascii_codepoint))
    elif 0x1D44E <= codepoint <= 0x1D467:
        ascii_codepoint = codepoint - 0x1D3ED 
        result.append(chr(ascii_codepoint))
    else:
      result.append(char)
  return ''.join(result)

def _translate_pdf(
    pdf_name, data_dir, output_dir ,all_pages=False, specific_pages_list_1 = None
) -> None:
    start_time = time.time()
    pdf_path = data_dir + pdf_name + '.pdf'
    pdf_images = convert_from_path(pdf_path, dpi=DPI)
    pdf_images_2_dpi = convert_from_path(pdf_path, dpi=2*DPI)

    pdfdoc = fitz.open(pdf_path)        #用来写入中文的pdf
    pdfdoc_fig = fitz.open(pdf_path)    #用来写入中文的pdf,含有图片的pdf
    pdfdoc_copy = fitz.open(pdf_path)   #提取英文的原版pdf
    print('pdf_===',len(pdfdoc), len(pdf_images))
    if all_pages == True:
        specific_pages_list = list(range(len(pdfdoc)))
    else:
        specific_pages_list = [n-1 for n in specific_pages_list_1]

    for i in specific_pages_list:
        pdf_page = pdfdoc[i]
        pdf_page_fig = pdfdoc_fig[i]
        image = pdf_images[i]
        width_img, height_img = image.size
        img_pdf_scale = height_img / pdf_page.rect.height

        print(f"###### page {i} scale {img_pdf_scale} Image: {width_img} pixels x {height_img} pixels   pdf height {pdf_page.rect.height}")

        __translate_one_page(
            image=image,
            image_2_dpi = pdf_images_2_dpi[i],
            pdf_page = pdf_page,
            pdf_page_fig = pdf_page_fig,
            No = i,
            pdfname = pdf_name,
            pdf_page2 = pdfdoc_copy[i],
            img_pdf_scale=img_pdf_scale
        )

    output_path = output_dir + f'\\{pdf_name[:40]}_A_{str(model_select)} .pdf'
    pdfdoc.subset_fonts(fallback=True, verbose=False)
    pdfdoc.ez_save(output_path, expand = 255, deflate=True, clean = True, deflate_images = True,  deflate_fonts = True) # , garbage=0,linear =True pretty = True
    pdfdoc.close()    #用来写入中文的pdf
    
    output_path_fig = output_dir + f'\\{pdf_name[:40]}_C_{str(model_select)} .pdf'
    pdfdoc_fig.subset_fonts(fallback=True, verbose=False)
    pdfdoc_fig.ez_save(output_path_fig, expand = 255, deflate=True, clean = True, deflate_images = True,  deflate_fonts = True) # , garbage=0,linear =True pretty = True
    pdfdoc_fig.close()    #用来写入中文的pdf
    
    pdfdoc_copy.close()    

    if all_pages == True and False:                
        mergepdf_pypdf2(pdf_path, output_path, output_dir + f'\\{pdf_name[:40]}_B_{str(model_select)} .pdf')

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = time.time() - start_time
    formatted_time = str(timedelta(seconds=total_time))
    print_str = f"\n{time_now}  {pdf_name}\n        {insertbox_count}   {model_select}   {total_tokens}   {formatted_time} \n"

    print(print_str)
    with open(data_dir + "pdf_info.log", "a", encoding="utf-8") as log_file:
        log_file.write(print_str)    # f"{formatted_date}: \n

def mergepdf_pypdf2( pdf1_path, pdf2_path, output_path):
    # 打开PDF文件
    pdf_a = PyPDF2.PdfReader(open(pdf1_path, 'rb'))
    pdf_b = PyPDF2.PdfReader(open(pdf2_path, 'rb'))

    # 创建一个PDF writer对象
    pdf_writer = PyPDF2.PdfWriter()

    # 获取PDF文件的页数
    num_pages_a = len(pdf_a.pages)
    num_pages_b = len(pdf_b.pages)

    # 确保两个PDF文件页数相同
    assert num_pages_a == num_pages_b, "The PDF files must have the same number of pages."

    # 交替插入页面
    for i in range(num_pages_a):
        pdf_writer.add_page(pdf_a.pages[i])
        pdf_writer.add_page(pdf_b.pages[i])

    # 写入新的PDF文件
    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)

    print("PDF files merged successfully.")


## 问题： 两个2m的文件，合成之后16m
def interleave_pdfs( doc1, doc2, output_path):

    merged_doc = fitz.open()

    for i in range(0, len(doc1)):
        merged_doc.insert_pdf(doc1, from_page=i, to_page=i)
        merged_doc.insert_pdf(doc2, from_page=i, to_page=(i))

    merged_doc.subset_fonts(fallback=True, verbose=False)
    merged_doc.save(output_path, garbage=3, deflate=True, clean = True, deflate_fonts = True)

    # 关闭文档
    doc1.close()
    doc2.close()
    merged_doc.close()

def get_text_info( pdf_page2,rect):

    blocks_text = pdf_page2.get_textbox(rect)
    blocks_text = blocks_text.replace("\n", " ")
    blocks_text = blocks_text.replace("- ", "")
    # blocks_text = blocks_text.replace(' ', '\xa0')  # 注意'\xa0'是U+00A0的转义序列
    # print('++++++++++', blocks_text)

    blocks2 = pdf_page2.get_text("dict", clip = rect, flags = 195)
    # print(blocks2)
    # print(blocks['blocks']) #[0]['lines']
    textjoin = []
    fontsize_counts = {}
    fontbold_ = fonttype_all = 0
    max_font = max_count = 0
    
    for block in blocks2['blocks']:  # this is a text block
        # print('\n*************',block)
        for l in block['lines']:  # iterate through the text lines
            for s in l["spans"]:  # iterate through the text spans 
                font_size =  round(s['size'], 1)
                # print(font_size)

                if font_size in fontsize_counts:
                    fontsize_counts[font_size] += len(s["text"])
                else:
                    fontsize_counts[font_size] = len(s["text"])
                

                if 'bold' in s['font'].lower() or 'medi' in s['font'].lower():
                    fontbold_ = fontbold_+1
                
                fonttype_all = fonttype_all + 1
                # print(s['size'], fitz.sRGB_to_rgb(s['color']),  s['font'],f'_  {len(s["text"])}  {s["text"]}_')  

    if len(blocks_text)==0 or fonttype_all == 0:
        flag_null = True
        fonttype_all = 1 # 防止除0
        max_font = 1
    else:
        flag_null = False
        #######################  find max ########################
        for font, count in fontsize_counts.items():
            if count > max_count:
                max_font = font
                max_count = count
        #######################  find max ########################
        # fontsize_block = max(set(fontsize_), key=fontsize_.count)

    # print(fontsize_counts)
    # print(max_font, max_count)
                    
    return flag_null, blocks_text, max_font, fontbold_/fonttype_all

def translate_easyword( s):
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
        "result": "结果",
        "results": "结果",
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
    
    image: Image.Image,
    image_2_dpi,
    pdf_page,
    pdf_page_fig,
    No,      # 当前的序号,
    pdfname,
    pdf_page2,
    img_pdf_scale
) -> Tuple[np.ndarray, np.ndarray, bool]:
    global total_tokens
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

    img_np = np.array(image, dtype=np.uint8)
    # original_img_np = copy.deepcopy(img_np)
    # result = layout_model(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    imgsize = imagesize
    print('&&&&& imagesize', imgsize)

    # results = yolomodel(source=img_np, save=False, show_labels=True, show_conf=True, show_boxes=True,agnostic_nms = True, iou = 0.2 )#iou = 0 ,  imgsz = int(imgsize*32) iou = 0.7 
    results = yolomodel.predict(source=image, conf=0.2, iou=0.8, nms = True) #imgsz = int(imgsize*32)

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

            pdf_pos = [(x / img_pdf_scale) for x in xyxy]
            rect = fitz.Rect(pdf_pos[0], pdf_pos[1], pdf_pos[2], pdf_pos[3])  

            if picture_figure == True:
                if (classsify_name in ["Formula","Picture","Table"]):

                    rect_p1 = fitz.Rect(pdf_pos[0]-0.5, pdf_pos[1]-0.5, pdf_pos[2]+0.5, pdf_pos[3]+0.5)  
                    #############
                    annot = pdf_page_fig.add_redact_annot(rect_p1)
                    annot.set_colors(stroke=(1, 0, 0), fill=(0.88, 0.92, 0.81))  # 设置边框和填充颜色
                    annot.update()  # 必须更新注释以应用更改
                    pdf_page_fig.apply_redactions(images = fitz.PDF_REDACT_IMAGE_REMOVE_UNLESS_INVISIBLE ) #graphics  = 3, text = 2
                    #############

            if classsify_name in["Section-header","Text","List-item","Caption","Page-header"]:  #, 
                #############
                annot = pdf_page.add_redact_annot(rect)
                annot.set_colors(stroke=(1, 0, 0), fill=(0.98, 0.99, 0.96))  # 设置边框和填充颜色
                annot.update()  # 必须更新注释以应用更改
                pdf_page.apply_redactions(images = fitz.PDF_REDACT_IMAGE_REMOVE_UNLESS_INVISIBLE ) #graphics  = 3, text = 2

                #############
                annot_fig = pdf_page_fig.add_redact_annot(rect)
                annot_fig.set_colors(stroke=(1, 0, 0), fill=(0.98, 0.99, 0.96))  # 设置边框和填充颜色
                annot_fig.update()  # 必须更新注释以应用更改
                pdf_page_fig.apply_redactions(images = fitz.PDF_REDACT_IMAGE_REMOVE_UNLESS_INVISIBLE ) #graphics  = 3, text = 2

                #############

    for result in results:
        boxes = result.boxes  # 包含边界框信息
        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # 转换为列表，包含(x1, y1, x2, y2)坐标
            conf = box.conf.item()  # 置信度
            cls = box.cls.item()  # 类别ID
            classsify_name = yolomodel.names[int(cls)]  # 类别名称
            # label = yolomodel.names[int(box.cls[0])]

            ##################################################  

            pdf_pos = [(x / img_pdf_scale) for x in xyxy]

            image_orig = results[0].orig_img  # 获取原始图像
            xy_ = 4                       
            cv2.rectangle(image_orig, (math.floor(xyxy[0]-xy_), math.floor(xyxy[1]-xy_)), (math.ceil(xyxy[2]+xy_), math.ceil(xyxy[3]+xy_)), color_map[classsify_name], 3)
            cv2.putText(image_orig, classsify_name+ f" {conf:.2f} {(pdf_pos[2]-pdf_pos[0]):.2f}", (int(xyxy[0]), int(xyxy[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = color_map[classsify_name], thickness = 2)  # 绘制标签 
            path_img = out_directory + f'\\layout\\{pdfname}' #f'./data1/layout/o_{pdfname}_'
            if not os.path.exists(path_img):
                os.makedirs(path_img)
            cv2.imwrite(f'{path_img}\\{No+1} .png', image_orig) #{pdfname}
            #############

            if picture_figure == True:
                if (classsify_name in ["Formula","Picture","Table"]):
                    rect = fitz.Rect(pdf_pos[0], pdf_pos[1], pdf_pos[2], pdf_pos[3])  
                    xy_ = 0                 
                    cropped_img = image_2_dpi.crop(((2*xyxy[0]-xy_), (2*xyxy[1]-xy_), (2*xyxy[2]+xy_), (2*xyxy[3]+xy_)))
                    # 将PIL图像转换为字节流
                    img_byte_arr = io.BytesIO()
                    cropped_img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    pdf_page_fig.insert_image(rect = rect, stream = img_byte_arr)
                                    
            if (classsify_name in ["Section-header","Text","List-item","Caption"]) or ((classsify_name in ["Page-header"]) and ((xyxy[3]-xyxy[1])/(xyxy[2]-xyxy[0])) < 5 ):  #, 

                # cropped_img_np = np.array(cropped_img, dtype=np.uint8)
                # cv2.imwrite(f'./out/Image_{classsify_name}_{conf:.2f}.png',cropped_img_np )

                ################# paddle ocr #####################
                # ocrrs = ocr.ocr(cropped_img_np,cls=False)[0]
                # ocr_results = [item[1][0] for item in ocrrs]

                ################# soyar  ocr #####################

                # predictions = run_ocr([cropped_img], [['en']], det_model, det_processor, rec_model, rec_processor)
                # ocr_results = [box.text for box in predictions[0].text_lines]

                    # print(f'\n___ completion {usage_tokens.completion_tokens} prompt {usage_tokens.prompt_tokens} total {usage_tokens.total_tokens} \n\n\n')

                rect = fitz.Rect(pdf_pos[0], pdf_pos[1]+1, pdf_pos[2], pdf_pos[3]-1 )            
                flag_null, text_block, fontsize_block, font_bold_ratio = get_text_info(pdf_page2,rect)
                if flag_null == True: 
                    print('\n$$$$$$$$$$$$ skip  ')

                    continue

                test_easy, flag_easyword = translate_easyword(text_block)
                # print(flag_easyword)
                if flag_easyword ==True:
                    text_block = test_easy
                    print('flag_easyword', text_block)
                else:

                    if text_block and text_block[0].isdigit():  #类似标题 2.4
                    # 使用第一个空格分割字符串
                        parts = text_block.split(" ", 1)    # parts 类似于 ['3.3.2', ' How can I ']
                        if len(parts)>1: #类似于 2.3 apple ， 否则为纯数字 ，不进行翻译
                            if parts[1] == '':  # 类似于['3.3.2', '']
                                print('only numbers, part[1] no contents')
                            else:
                                trans_part1 , usage_tokens = translate_with_timeout(parts[1])
                                total_tokens = total_tokens + usage_tokens
                                text_block = parts[0] +'  '+ trans_part1
                    elif re.match(r'^\[(\d+)\]', text_block):  #类似reference [2] afd
                    # 使用第一个空格分割字符串

                        parts = text_block.split(" ", 1)
                        if len(parts)>1: #类似于 2.3 apple ， 否则为纯数字 ，不进行翻译

                            trans_part1 , usage_tokens = translate_with_timeout(parts[1])
                            total_tokens = total_tokens + usage_tokens
                            text_block = parts[0] +'  '+ trans_part1
                    else:
                        text_block = convert_math_alpha_to_ascii(text_block)
                        text_block , usage_tokens = translate_with_timeout(text_block)
                        total_tokens = total_tokens + usage_tokens
                text_block = text_block.replace("\n", "   ")
                text_block = text_block.replace('<|im_end|>', '')  # gemma2 结尾有时会有
                text_block = text_block.replace('|im_end|>', '')  # gemma2 结尾有时会有
                text_block = text_block.replace('<|eotid|>', '')  # gemma2 结尾有时会有
                text_block = text_block.replace(' ', '\xa0')  # 注意'\xa0'是U+00A0的转义序列

                # print('\n',,'\n',)
                
                # print(f'******** fontsize {fontsize_block} {font_bold_ratio} \n{text_block}')
                margin_textbox = 2
                if font_bold_ratio > 0.5:
                    insertbox_count = pdf_page.insert_textbox(rect = (pdf_pos[0]-margin_textbox, pdf_pos[1]-margin_textbox, pdf_pos[2]+margin_textbox, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_bold_name,fontfile = font_bold_file_path ,fontsize = fontsize_block-0.4, color=[0, 0, 0], lineheight = 1.22)   

                    insertbox_count = pdf_page_fig.insert_textbox(rect = (pdf_pos[0]-margin_textbox, pdf_pos[1]-margin_textbox, pdf_pos[2]+margin_textbox, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_bold_name,fontfile = font_bold_file_path ,fontsize = fontsize_block-0.4, color=[0, 0, 0], lineheight = 1.22)    
                else:
                    if fontsize_block > 20 or fontsize_block < 8:
                        insertbox_count = pdf_page.insert_textbox(rect = (pdf_pos[0]-margin_textbox, pdf_pos[1]-margin_textbox, pdf_pos[2]+margin_textbox, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_name,fontfile = font_file_path ,fontsize = fontsize_block-0.8, color=[0, 0, 0], lineheight = 1.12)   

                        insertbox_count = pdf_page_fig.insert_textbox(rect = (pdf_pos[0]-margin_textbox, pdf_pos[1]-margin_textbox, pdf_pos[2]+margin_textbox, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_name,fontfile = font_file_path ,fontsize = fontsize_block-0.8, color=[0, 0, 0], lineheight = 1.12)    
                    else:    
                        insertbox_count = pdf_page.insert_textbox(rect = (pdf_pos[0]-margin_textbox, pdf_pos[1]-margin_textbox, pdf_pos[2]+margin_textbox, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_name,fontfile = font_file_path ,fontsize = fontsize_block-0.6, color=[0, 0, 0], lineheight = 1.23) 
                    
                        insertbox_count = pdf_page_fig.insert_textbox(rect = (pdf_pos[0]-margin_textbox, pdf_pos[1]-margin_textbox, pdf_pos[2]+margin_textbox, 3*pdf_pos[3]-pdf_pos[1]), buffer = text_block, fontname = font_name,fontfile = font_file_path ,fontsize = fontsize_block-0.6, color=[0, 0, 0], lineheight = 1.23)    
                if insertbox_count < 0:
                    insertbox_count += 1

                # page.insert_textbox(rect, modified_text, fontsize=font_size-4,fontname = 'simsunbold' , color = colr, fontfile = "../server/font/SourceHanSerifCN-SemiBold.ttf") 

    return img_np


def __translate_llm(text: str, temper, seed, top_p) -> str:
    

    # return text, 1  # No translate
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
        api_key = "lm-studio",
        base_url = "http://localhost:1234/v1/"
    )
    if model_select == 0:
        model_name = "Qwen/Qwen2-7B-Instruct-GGUF"
    elif model_select == 1:
        model_name = "cognitivecomputations/dolphin-2.9.2-qwen2-7b-gguf"
    elif model_select == 2:
        model_name = "second-state/Gemma-2-9B-Chinese-Chat-GGUF"
    elif model_select == 3:
        model_name = "hfl_chinese_instruct_v3/llama-3-chinese-8b-instruct-v3-gguf"
    elif model_select == 4:
        model_name = "chatpdflocal/llama3.1-8b-gguf"
    elif model_select == 5:
        model_name = "lmstudio-community/gemma2-2b-q8"
    elif model_select == 6:
        model_name = "lmstudio-community/gemma-2-9b-it-Q6_K"
    # # 5 

    system_prompt = """
请将英文翻译成中文，保持英文原文中人名，专有名词，数学字符符号和公式的格式保持原始不变，不要将数学符号和公式转换成 LaTeX 公式形式。请直接将英文翻译成中文，只输出翻译结果，不要输出其他的注释note、注意Note、提示tips 和解释explanation，直接输出翻译后的中文结果。
"""
# Translate into Chinese, don't output notation, attention, tips, warning. keeping the format of mathematical symbols and equations as they are in the original english text. don't convert mathematical symbols and equations into math Latex $$ formulas. Translate english into Chinese directly. only output english translation results, and don't output other notation, attention, tips, warning, interpret. output english translation results

    response = client.chat.completions.create(
    model=model_name,
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    stream = False,
    temperature = temper,
    seed = seed,
    top_p = top_p
    )
    translation = response.choices[0].message.content
    usage_tokens = response.usage 
    # print('+++',text,'+++')
    # print('~~~',translation,'~~~')

    return translation, usage_tokens.total_tokens

def translate_with_timeout(text, temper = 0.5, seed = 33, top_p = 0.95, timeout=30):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            future = executor.submit(__translate_llm, text, temper, seed, top_p)
            result  = future.result(timeout)
            # print(result,'&&&&&&&&')
            return result
        except concurrent.futures.TimeoutError:
            print("\n##################Translation timed out. Retrying...   " ,{temper},   {seed})
            # 在这里可以重新尝试，或者根据需要处理超时
            return f"# {text} #", 10
#  translate_with_timeout(text, temper + 0.1, seed+10, timeout, )  # 递归调用自身以重新尝试

if __name__ == "__main__":

    imagesizes = [32]   #10,15,20,25,30,35, 45,50  17.5, 22.5, 27.5, 32.5, 37.5
    for imagesize in imagesizes:
        data_directory ='C:\\Users\\18420\\Desktop\\AIpaper\\translate\\' #data/  weekly_paper

        out_directory = data_directory+ 'layout_\\DPI_'+str(DPI)
        print(out_directory)
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)

        for filename in os.listdir(data_directory):
            if filename.endswith('.pdf'):
                # 提取PDF文件名
                pdf_name = os.path.splitext(filename)[0]
                print(f"PDF name:  {pdf_name}")
                _translate_pdf(pdf_name, data_directory, out_directory,all_pages=1, specific_pages_list_1 = [9])            

    
