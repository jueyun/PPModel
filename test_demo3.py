
from combine_picturev2 import *
## combine_mode = 2
bg_list = ['工地','沙漠','草原']
# fg_list = ['马','羊','人','牛']
fg_list = ['人']
description_list = []
for bg in bg_list:
    for fg in fg_list:
        description_list.append(bg + fg)
for description in description_list:
    # bg
    if '草原' in description:
        color = 'green'
        bg = 'grasslands'
        bg_path = 'picture_hub/bg/grasslands'
        n_bg = len(os.listdir('picture_hub/bg/grasslands'))
    elif '沙' in description:
        color = 'orange'
        bg = 'sand'
        bg_path = 'picture_hub/bg/sand'
        n_bg = len(os.listdir('picture_hub/bg/sand'))
    elif '工地' in description:
        color = 'orange'
        bg = 'construction'
        bg_path = 'picture_hub/bg/construction'
        n_bg = len(os.listdir('picture_hub/bg/construction'))
    #fg
    if '羊' in description:
        fg = 'sheep'
        fg_path = 'picture_hub/fg/sheep'
        n_fg = len(os.listdir('picture_hub/fg/sheep'))
    if '牛' in description:
        fg = 'ox'
        fg_path = 'picture_hub/fg/ox'
        n_fg = len(os.listdir('picture_hub/fg/ox'))
    if '人' in description:
        fg = 'people'
        fg_path = 'picture_hub/fg/people'
        n_fg = len(os.listdir('picture_hub/fg/people'))
    if '马' in description:
        fg = 'horse'
        fg_path = 'picture_hub/fg/horse'
        n_fg = len(os.listdir('picture_hub/fg/horse'))

    for bg_idx in range(1,n_bg+1):
        for fg_idx in range(1, n_fg+1):
            print(bg,fg,bg_idx,fg_idx)
            bg_path_new = bg_path + '/bg' + str(bg_idx) + '.jpeg'
            fg_path_new = fg_path + '/fg' + str(fg_idx) + '.jpeg'
            print(bg_path,fg_path)
            img = clear_and_combine_test(base_dir='picture_hub', bg_path=bg_path_new, fg_path_list=[fg_path_new], color=color)
            output_path = 'picture_hub/test_combine/' + bg + '_' + str(bg_idx) + '_' + fg + '_' + str(fg_idx) + '.jpeg' 
            cv2.imwrite(output_path, img)
# print(output_path)
# print(bg_path)

