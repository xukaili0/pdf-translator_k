import fitz, pymupdf

# 字体文件的路径
# font_path = "./font/SourceHanSerifCN-Medium.otf" 
# font_path = "./font/simsun.ttf" 
# font_name = 'simsun'
# font_bold_path = "./font/simsunbold.ttf" 
# font_bold_name = 'simsunbold'
# font_path = "./font/simsun.ttf" 
# font_name = 'simsun'
font_name = 'SerifCN'
font_path = "../server/font/SourceHanSerifCN-Medium.ttf"  # 或 .otf 文件

font_bold_name = 'SerifCNbold'
font_bold_path ="../server/font/SourceHanSerifCN-Bold.ttf" # "./font/
#C:\\pdf-translator\\server\\font\\SourceHanSerifCN-Medium.otf
fontfile = open(font_path, "rb").read()
font = fitz.Font(fontbuffer=fontfile)
# 创建一个新的PDF文档
doc = fitz.open()

# 添加一页
page = doc.new_page()

# 加载字体
font = fitz.Font(fontfile=font_path)
print(font)
# 或者，如果字体是嵌入式的，你可以这样加载：
# font = fitz.Font(fontfile=fitz.open(font_path), embedded=True)

# 设置字体和字号
font_size = 16

# 在页面上写入文本
text = """abcdehigh <span style="color: #ab00;">reprehenderit</span>zd ipQWDVHIP,._， <b>Hello</b> 你好，世界 于 讲 赵 雷电 挑 
<p style="color: #ff0000;">这也是一段gfsgs红色的文字，使用十六进制表示。</p>运 东 、\n数据说明：包括正常轴承数据 ！"""
text2 = """a<b>bcd解放军偶ehi,ghz</b>dip- QWDV HI P, ._，。Hello你好，世界 于 讲 赵 雷电 挑 运 东 、hthfh/，。75747252 36，，/，数据说明：包括正常轴承数据 ！jas,fjidjal jfl d.ijfiod .keknso  ji. ojeflj e jwqq  尔就就是房价 窘境哦加哦就就 基金理解。， 集i哦窘境哦i就。士。。 <b>.基金】【 就，gsgh</b>u ijg  及飞机公司2i哦i经济oig_就 akdshjkashshfhskhahsajkhskjhdkszh k解放军 偶尔就就 是发大水 就嗲家覅的时 间佛加哦就 的撒了解 就打死哦啊睡觉哦的发 掘我的教案 课件案件平时覅安 件 覅欧吉安按 军法案件覅哦 安吉尔啊案件按 揭贷款链接发"""
#
# body {
#     font-family: 'simsun-m';
#     src: url('../server/font/SourceHanSerifCN-Medium.ttf') format('ttf');
# }

text3 = """
<style>
@font-face {
  font-family: 'simsun-m';
  src: url('../server/font/SourceHanSerifCN-Medium.ttf') format('ttf');
}
</style>
Lorem ipsum dolor排时间发案ambvceta, consectetur adipisici elit, sed
    minim veniam, quis nostrud exercitation <b>ul集发动lamco <i>lab民间oris</i></b>hthfh/，。75747252 36，，/，数据说明：包括正常轴承数据 ！jas,fjidjal jfl d.ijfiod .keknso  ji. ojeflj e jwqq  尔就就是房价 窘境哦加哦就就 基金理解。， 集i哦窘境哦i就faddasfdsa。。 <b>.基金】【 就gsfgfdgfdgddfgs gh</b>u ijg  及飞机公司2i哦i经济oig_就 akdshjkashshfhskhahsajkhskjhdkszh k解放军 偶尔就就 是发大水 就嗲家覅的时 间佛加哦就 的撒了解 就打死哦啊睡觉哦的发 掘我的教案 课件案件平时覅安 件 覅欧吉安按
    nisi ut aliquid ex ea commodi 间发 consequat. Quis aute iure
    <span style="color: #f00;">repre大夫henderit</span>
    in <span style="color: #0f0;font-weight:bold;">volu房价ptate</span> velit
    <a href="https://www.artifex.com">officia</a> deserunt mollit anim id
    est laborum."""
font = fitz.Font("cjk")
page.insert_text((20, 50), text, fontsize=font_size,fontname=font_name, fontfile = font_path) #'unicode', 'china-s'   "china-s"

page.insert_text((20, 150), text, fontsize=font_size-1,fontname = 'semibold', fontfile = "../server/font/SourceHanSerifCN-SemiBold.ttf") #'unicode', 'china-s'   "china-s"

page.insert_text((20, 200), text, fontsize=font_size-2,fontname=font_bold_name, fontfile = font_bold_path) #'unicode', 'china-s'   "china-s"


page.insert_text((20, 250), text, fontsize=font_size-3,fontname = 'simsunbold' ,fontfile = "../server/font/SourceHanSerifCN-Heavy.ttf") #'unicode', 'china-s'   "china-s"
rect = fitz.Rect(20, 300, 240, 900)
# 原始文本

# 替换U+0020空格为U+00A0不间断空格
modified_text = text3.replace(' ', '\xa0')  # 注意'\xa0'是U+00A0的转义序列

colr = (19/255,137/255,218/255)
print(colr)
# page.insert_textbox(rect, modified_text, fontsize=font_size-4,fontname = 'simsunbold' , color = colr, fontfile = "../server/font/SourceHanSerifCN-SemiBold.ttf") 

page.insert_htmlbox(rect, text3, css="* {font-family: simsun-m;font-size:16px;}") 

# doc.ez_save("x.pdf")
# # 嵌入字体到PDF  
# for i in range(len(doc)):
#     page = doc[i]
#     for j, img in enumerate(page.get_images(full=True)):
#         xref = img[0]
#         smask = fitz.Pixmap(fitz.csRGB, img[0])
#         pix = fitz.Pixmap(img[0])
#         pix0 = fitz.Pixmap(pix)
#         if pix0.n < 5:          # this is GRAY or RGB
#             pix = pix0
#         else:                   # CMYK: convert to RGB first
#             pix = fitz.Pixmap(fitz.csRGB, pix0)
#         pix.writePNG("img-%s-%i.png" % (i, j))
#         img[0].set_filter(fitz.PDF_FILTER_FLATE)

# 保存PDF
doc.subset_fonts(fallback=True, verbose=False)
doc.save("./out_t1.pdf", garbage=4, deflate=True, clean = True,deflate_fonts = True)
# doc.ez_save("output_with_embedded_font.pdf", garbage=4, deflate=True, clean = True, deflate_fonts = True) 
# 关闭文档
doc.close()