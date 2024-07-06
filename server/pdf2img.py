from pdf2image import convert_from_path

# 指定PDF文件路径
pdf_file_path = 'tb4.pdf'

# 将PDF转换为图像列表
images = convert_from_path(pdf_file_path)

# 遍历图像列表，保存每个图像
for i, image in enumerate(images):
    # 保存图像到文件
    image.save(f'tb4_output_page_{i+1}.png', 'PNG')
    # image.save(f'tb1_output_page_{i+1}.jpeg', 'JPEG')

print("PDF pages have been converted to images.")