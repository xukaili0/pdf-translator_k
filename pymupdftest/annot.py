import fitz

# 打开PDF文档
doc = fitz.open('out_t1.pdf')

# 遍历文档中的每一页
for page in doc:
    # 搜索特定的字符串
    matches = page.search_for('colo')
    
    # 遍历所有找到的匹配项
    for match in matches:
        # 创建高亮注释
        highlight = page.add_highlight_annot(match)
        # highlight.set_colors((0.4,0.2,0.6))
        # 设置高亮注释的颜色为红色fill=(1, 0, 0)
        
        # 可选：设置高亮注释的透明度（0.0 到 1.0）
        highlight.set_colors(stroke=(0.7, 0.9, 0.3), fill=(1, 0.6, 1))
        highlight.update()

# 保存修改后的文档
doc.save('out_t2.pdf')