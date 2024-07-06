from PIL import Image, ImageDraw, ImageFont
new_block = Image.new(
    "RGB",
    (
1,2    ),
    color=(255, 255, 255),
)
draw = ImageDraw.Draw(new_block)
