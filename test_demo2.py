
from combine_picturev2 import *
# seg_image_all()
## combine_mode = 2
description = "工地上有一群人" #输入文字
# base_dir = 'picture_hub'
bg_path = 'picture_hub/bg/construction/bg2.jpeg'
# bg_path = 'picture_hub/bg/sand/bg2.jpeg'
output_path, bg_path = description2pix(description, combine_mode=2)
# for ii in range(1,6):
#     output_path, fg_out_path_list = description2pix(description, base_dir='picture_hub', bg_path=bg_path, 
#         fg_path_list=['picture_hub/fg/people/fg'+str(ii)+'.jpeg'], combine_mode=1)
# # print(output_path)
# # print(bg_path)

