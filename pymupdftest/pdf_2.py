import fitz

doc = fitz.open("ta1.pdf")
page = doc[0]  # load first page(37.587398529052734, 251.53643798828125, 291.03155517578125, 316.9003601074219)


# 获取包含布局信息的文本块
for i in range(0,len(doc)):
    print('--',i)
    blocks = doc[i].get_text("blocks")
    blocks1 = doc[i].get_textbox((300.83823389049405, 706.9585543775181, 544.2511199330115, 720.3709799357295))
    print('blocks__',blocks1)

blocks = page.get_text("dict", clip =(300.83823389049405, 706.9585543775181, 544.2511199330115, 720.3709799357295))
print(blocks['blocks']) #[0]['lines']
for block in blocks['blocks']:  # this is a text block
    print('+++++++')
    for l in block['lines']:  # iterate through the text lines
        for s in l["spans"]:  # iterate through the text spans
            print(s['size'], s['font'],'   ',s['text'])                
# for b in blocks:  # iterate through the text blocks
#     print(b)
#     if b['type'] in 'blocks':  # this is a text block
#         for l in b["lines"]:  # iterate through the text lines
#             for s in l["spans"]:  # iterate through the text spans
#                 print(s)                
                # print(s["text"], s["size"])
                # 
