
from combine_picturev2 import *
## combine_mode = 2
description = "工地上有一群人，一群羊，一群马，一群骆驼" #输入文字
# base_dir = 'picture_hub'
# # bg_path = 'picture_hub/bg/grasslands/bg1.jpeg'
# bg_path = 'picture_hub/bg/sand/bg2.jpeg'
output_path, bg_path = description2pix(description, combine_mode=2)
print(output_path)
print(bg_path)

