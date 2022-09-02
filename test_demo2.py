
from combine_picturev2 import *
## combine_mode = 2
description = "沙漠上有一群牛，一群羊，一群人" #输入文字
base_dir = 'picture_hub'
# bg_path = 'picture_hub/bg/grasslands/bg1.jpeg'
bg_path = 'picture_hub/bg/sand/bg2.jpeg'
output_path = description2pix(description, base_dir, bg_path, combine_mode=2)
print(output_path)

