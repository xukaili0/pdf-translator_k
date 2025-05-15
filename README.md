# PDF 翻译工具

本项目是基于 [pdf-translator](https://github.com/discus0434/pdf-translator.git) 的个人改进版本，旨在提升 PDF 文件翻译的灵活性和功能性。以下是本项目的主要改进内容：

## 主要改进

1. **灵活的文本提取方式**  
   支持两种 PDF 文本提取方式：  
   - 使用 `PyMuPDF` 直接提取 PDF 中的文本内容。  
   - 对图片形式的文本内容，使用 `PaddleOCR` 进行光学字符识别（OCR）提取。

2. **多样化的翻译接口**  
   - 新增支持通过 `LLM API` 进行文本翻译。  
   - 同时支持使用本地部署的大型语言模型（LLM）进行翻译。

3. **优化输出格式**  
   - 新增了翻译中文显示的支持。
   - 原项目仅支持图片格式输出，现新增 PDF 输出格式,可以指定翻译的页数。  
   - 新输出的 PDF 格式支持选中翻译后的文本，并优化了显示效果。
   

4. **非文本内容的处理**  
   - 对于无法直接翻译的内容（如公式、表格、图片等），自动将其转化为图片形式嵌入输出文件，保留原始信息。

5. **文本布局检测**  
   - 集成 `YOLO`布局检测模型 和 `Paddle`OCR识别模型 ，用于检测 PDF 中的文本布局，提升复杂文档的处理精度。
## 翻译效果
见``example``文件夹

## 运行方式

1. **克隆代码**  
   ```bash
   git clonehttps://github.com/xukaili0/pdf-translator_k.git
   ```

2. **下载模型**  
   从 Hugging Face 平台下载 [YOLO] (https://huggingface.co/spaces/omoured/YOLOv10-Document-Layout-Analysis/tree/main) 布局检测模型，从[Paddle](https://www.paddlepaddle.org.cn/)下载OCR识别模型，并将其放置在项目指定目录下。


3. **配置翻译设置**  
   在配置文件中填写以下内容：  
   - 选择使用的翻译接口（LLM API 或本地 LLM）。  
   - 如果使用本地 LLM，需提供模型路径或相关配置。  
   - 指定待翻译 PDF 文件所在的文件夹路径。

4. **运行项目**  
   确保依赖安装完成后，将pdf文件夹放入main的路径中即可。  


## 感谢

[pdf-translator](https://github.com/discus0434/pdf-translator.git)

