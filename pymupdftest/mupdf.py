import fitz  # PyMuPDF

# 打开PDF文件
doc = fitz.open("ta1.pdf")

# 初始化一个空字符串来保存所有页面的文本
text = ""

# 创建一个新的PDF文档
doc_new = fitz.open()

# 添加一个新的空白页
doc_new.new_page()
print(len(doc))
# 获取包含布局信息的文本块
for i in range(0,len(doc)):
    print('--',i)
    blocks = doc[i].get_text("blocks")
    blocks1 = doc[i].get_textbox((300.83823389049405, 706.9585543775181, 544.2511199330115, 720.3709799357295))
    print('blocks',blocks1)
    # for block in blocks:
    #     print('--',block)
        # doc_new[i].insert_text((block[0], block[1]), block[4], fontsize=8+1, color=[0, 0, 0] , lineheight = 1.5)    
        # ,fontfile = font_bold_file_path,font_bold_name
# doc_new[0].insert_text((5,9), 'terewrjoiwejr', fontsize = 18)
# doc_new.save('out_t1.pdf', garbage=3, deflate=True, clean = True, deflate_fonts = True) 

# 打印提取的文本
print(text)

# 关闭PDF文档
doc.close()