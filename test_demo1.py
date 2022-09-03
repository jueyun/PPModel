
from combine_picturev2 import *
## combine_mode = 1
description = "沙漠上有一群马" #输入文字
base_dir = 'picture_hub'
bg_path = 'picture_hub/bg/sand/bg1.jpeg'
fg_path_list = ['picture_hub/fg/ox/fg5.jpeg']
output_path, fg_out_path_list = description2pix(description, base_dir, bg_path, fg_path_list, combine_mode=1)
print(output_path, fg_out_path_list)
# for ii in range(1,11):
#     # bg_path = 'picture_hub/bg/sand/bg' + str(ii) + '.jpeg'
#     # fg_path_list = ['picture_hub/fg/sheep/fg1.jpeg','picture_hub/fg/sheep/fg2.jpeg',
#     # 'picture_hub/fg/ox/fg1.jpeg','picture_hub/fg/ox/fg2.jpeg']
#     fg_path_list = ['picture_hub/fg/horse/fg' + str(ii)+'.jpeg']
#     print(fg_path_list)
#     output_path, fg_out_path_list = description2pix(description, base_dir, bg_path, fg_path_list, combine_mode=1)
#     print(output_path, fg_out_path_list)