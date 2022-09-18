
from combine_picturev2 import *
## combine_mode = 1
description = "沙漠上有一群马" #输入文字
base_dir = 'picture_hub'

# 1.搜索背景接口：search_bg(description, n_picture=4, base_dir='picture_hub')
bg_path_list = search_bg(description)
print('bg_path_list', bg_path_list)

# 2.搜索前景接口-限制一种前景：search_fg(description, n_picture=4, base_dir='picture_hub'):
fg_path_list = search_fg(description, n_picture=4, base_dir='picture_hub')
print('fg_path_list', fg_path_list)

# 3.前景提取接口: get_fg(fg_path_list, func_mode = 1):
fg_out_path_list = get_fg(fg_path_list)
print('fg_out_path_list', fg_out_path_list)

# 4.图片合成接口-全部拼接:combine_pic_testv(description, bg_path_list, fg_rm_path_list, func_mode = 1, n_picture = 4, base_dir='picture_hub'):
fg_rm_path_list = fg_out_path_list
output_path = combine_pic_testv(description, bg_path_list, fg_rm_path_list)
print('output_path',output_path)