import fitz

doc = fitz.open('tb1.pdf')
pdf_page = doc[0]
rect = fitz.Rect(30,300,60,310)   
   
#############
annot = pdf_page.add_redact_annot(rect, cross_out  = False)
annot.set_colors(stroke=(1, 0, 0), fill=(0.97, 0.99, 0.95))  # 设置边框和填充颜色
annot.update()  # 必须更新注释以应用更改
pdf_page.apply_redactions(graphics = 0,text = 1 )
#############
doc.ez_save('./annot1' + '.pdf', )# garbage=3, deflate=True, clean = True, deflate_fonts = True
