# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
from removebg import RemoveBg
import cv2
import numpy as np
import random
import time
# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))
from paddleseg.utils import get_sys_env, logger, get_image_list
from infer import Predictor

import numpy as np
import collections

## 全局badcase
badcase = {"construction1":{"sheep":[],"ox":[5],"people":[3],"horse":[]},
           "construction2":{"sheep":[1],"ox":[3,5],"people":[2,3],"horse":[]},
           "construction3":{"sheep":[],"ox":[3],"people":[2,3],"horse":[]},
           "construction4":{"sheep":[],"ox":[3,5],"people":[2,3],"horse":[]},
           "construction5":{"sheep":[],"ox":[5],"people":[2,3],"horse":[]},
           "grasslands1":{"sheep":[],"ox":[3,5],"people":[2,3],"horse":[]},
           "grasslands2":{"sheep":[],"ox":[3],"people":[2,3],"horse":[]},
           "grasslands3":{"sheep":[],"ox":[3],"people":[2,3],"horse":[]},
           "grasslands4":{"sheep":[],"ox":[3,5],"people":[2,3],"horse":[]},
           "grasslands5":{"sheep":[],"ox":[3],"people":[2,3],"horse":[]},
           "grasslands6":{"sheep":[],"ox":[3],"people":[2,3],"horse":[]},
           "grasslands7":{"sheep":[],"ox":[3,5],"people":[2,3],"horse":[]},
           "sand1":{"sheep":[],"ox":[3],"people":[2,3],"horse":[]},
           "sand2":{"sheep":[],"ox":[5],"people":[2,3],"horse":[]},
           "sand3":{"sheep":[],"ox":[3,5],"people":[2,3],"horse":[]},
           "sand4":{"sheep":[],"ox":[5],"people":[2,3],"horse":[]},
           "sand5":{"sheep":[],"ox":[3],"people":[2,3],"horse":[]},
           "sand6":{"sheep":[6],"ox":[3,5],"people":[2,3],"horse":[]},
           "sand7":{"sheep":[],"ox":[5],"people":[2,3],"horse":[]},
}

def getColorList():
    color_dict = collections.defaultdict(list)

    # black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_dict['black'] = [lower_black, upper_black]

    # gray
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    color_dict['gray'] = [lower_gray, upper_gray]

    # white
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_dict['white'] = [lower_white, upper_white]

    # red
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_dict['red'] = [lower_red, upper_red]

    # red2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_dict['red2'] = [lower_red, upper_red]

    # orange
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_dict['orange'] = [lower_orange, upper_orange]

    # yellow
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_dict['yellow'] = [lower_yellow, upper_yellow]

    # green
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_dict['green'] = [lower_green, upper_green]

    # cyan
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_dict['cyan'] = [lower_cyan, upper_cyan]

    # blue
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_dict['blue'] = [lower_blue, upper_blue]

    # purple
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_dict['purple'] = [lower_purple,upper_purple]

    return color_dict


def parse_args(config, img_path):
    parser = argparse.ArgumentParser(
        description='PP-HumanSeg inference for video')
    
    parser.add_argument(
        "--config",
        help="The config file of the inference model.",
        type=str,default=config)
    parser.add_argument(
        '--img_path', help='The image that to be predicted.', type=str,default=img_path)
    parser.add_argument(
        '--video_path', help='Video path for inference', type=str)
    parser.add_argument(
        '--bg_img_path',
        help='Background image path for replacing. If not specified, a white background is used',
        type=str)
    parser.add_argument(
        '--bg_video_path', help='Background video path for replacing', type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')

    parser.add_argument(
        '--vertical_screen',
        help='The input image is generated by vertical screen, i.e. height is bigger than width.'
        'For the input image, we assume the width is bigger than the height by default.',
        action='store_true')
    parser.add_argument(
        '--use_post_process', help='Use post process.', action='store_true')
    parser.add_argument(
        '--use_optic_flow', help='Use optical flow.', action='store_true')
    parser.add_argument(
        '--test_speed',
        help='Whether to test inference speed',
        action='store_true')

    return parser.parse_args()


def get_bg_img(bg_img_path, img_shape):
    if bg_img_path is None:
        bg = 255 * np.ones(img_shape)
    elif not os.path.exists(bg_img_path):
        raise Exception('The --bg_img_path is not existed: {}'.format(
            bg_img_path))
    else:
        bg = cv2.imread(bg_img_path)
    return bg


################ 抠图方法，可继续在此添加 ################
## 1.paddle
def seg_image(config, img_path, save = False):
    '''
    parameter:
        -config:模型参数，写死
        -img_path:抠图图片的本地存储路径
        -save:是否存储中间结果
    return:
        -out_img:抠图结果的图片数组
    '''
    args = parse_args(config, img_path)
    args.use_gpu = False
    predictor = Predictor(args)
    img = cv2.imread(args.img_path)
    bg_img = get_bg_img(args.bg_img_path, img.shape)
    out_img = predictor.run(img, bg_img)
    if not save:
        return out_img
    img_path_split = img_path.split('/')
    img_path_split[-3] = 'fg_temp'
    img_path_split[-2] = img_path_split[-1] + '_' + img_path_split[-2] + '_removebg.jpeg'
    img_path_split = img_path_split[:-1]
    img_path_new = '/'.join(img_path_split)
    cv2.imwrite(img_path_new , out_img)
    return out_img, img_path_new
    

################ 合成方法，可继续在此添加 ################
## 1.按位拼接
def combine_picture(bg_img, fg_img, fg_shape, offset):
    '''
    parameter:
        -bg_img:bg图片数组
        -fg_img:fg图片数组
        -fg_shape: fg形状
        -offset:fg相对bg位置
    return:
        -bg_img:合成后的图片数组
    '''
    fg_img = cv2.resize(fg_img, fg_shape)
    rows,cols,_ = fg_img.shape
    roi = bg_img[offset[1]:offset[1]+rows, offset[0]:offset[0]+cols]
    img2gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    img2_fg = cv2.bitwise_and(fg_img,fg_img,mask = mask_inv)
    dst = cv2.add(img1_bg,img2_fg)
    bg_img[offset[1]:offset[1]+rows, offset[0]:offset[0]+cols] = dst
    return bg_img

## 2.直接相加
def combine_picture_v2(bg_img, fg_img, fg_shape, offset):
    '''
    parameter:
        -bg_img:bg图片数组
        -fg_img:fg图片数组
        -fg_shape: fg形状
        -offset:fg相对bg位置
    return:
        -bg_img:合成后的图片数组
    '''
    fg_img = cv2.resize(fg_img, fg_shape)
    rows,cols,_ = fg_img.shape
    roi = bg_img[offset[1]:offset[1]+rows, offset[0]:offset[0]+cols]
    bg_img[offset[1]:offset[1]+rows, offset[0]:offset[0]+cols] = cv2.add(roi,fg_img)
    return bg_img


## 背景检测
def detect_bg(bg_img, color):
    '''
    parameter:
        -bg_img:bg图片数组
        -color:检测颜色，str
    return:
        -point:检测位置，（topx,topy, width,height)
    '''
    bg_hsv = cv2.cvtColor(bg_img,cv2.COLOR_BGR2HSV)
    color_dict = getColorList()
    lower_green,higher_green = color_dict[color]
    mask_green = cv2.inRange(bg_hsv,lower_green,higher_green)#获得绿色部分掩膜
    mask_green = cv2.medianBlur(mask_green, 7) # 中值滤波
    cnts,hierarchy = cv2.findContours(mask_green,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    max_area = 0
    point = (0,0,0,0)
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)#该函数返回矩阵四个点
        if max_area < w * h:
            point = (x, y, w, h)
            max_area = w * h 
    return point

## 生成前景位置
def generate_fg_wh(w_range, fg_wh_ratio, before_x, point, bg_w, bg_h, i, max_num=10):
    '''
     parameter:
        -w_range:fg-width范围
        -fg_wh_ratio: fg width/height
        -before_x:前一前景终止位置
        -point:背景检测区域
        -bg_w:bg-width
        -bg_h:bg-height
        -max_num:最大随机生成次数
    return:
        fg位置，是否合格
    '''
    ## fg_location
    if i == 0:
        max_num = int(1e3)
    for _ in range(max_num):
        w = random.randint(w_range[0], w_range[1])
        h = int(fg_wh_ratio * w)
        loc_x = random.randint(before_x + 10, before_x + 50)
        loc_y = random.randint(point[1], point[1] + h)
        if loc_x + w < bg_w and loc_y + h < bg_h:
            return (loc_x, loc_y, w, h), True
    return (0,0,0,0), False

def remove_badcase(base_dir, bg_key, fg_name, fg_n, max_num = int(1e3)):
    #维护一个全局badcase:{bg_name+bg_n:{fg_name:[fg_n]}}
    if bg_key not in badcase:
        fg_path = base_dir + '/fg/' + fg_name + '/fg' + str(random.randint(1,fg_n)) + '.jpeg'
        return fg_path
    bad_fg_list = badcase[bg_key][fg_name]
    for _ in range(max_num):
        fg_number = random.randint(1,fg_n)
        if fg_number not in bad_fg_list:
            fg_path = base_dir + '/fg/' + fg_name + '/fg' + str(fg_number) + '.jpeg'
            return fg_path
    return fg_path

## 抠图+合成    
def clear_and_combine(base_dir, bg_name, fg_name, bg_max_n, fg_max_n, color):
    '''
    parameter:
        -base_dir:图片路径
        -bg_name:bg库
        -fg_name:fg库
        -bg_max_n:bg库中背景图数
        -fg_max_n:fg库中前景图数
        -color:需要检测的颜色
    return:
        output_path:合成后图片路径
    '''
    fg_seg_config = 'inference_model/deploy.yaml'
    ## bg处理
    bg_path = base_dir + '/' + bg_name + '/bg' + str(random.randint(1,bg_max_n)) + '.jpeg'
    bg_img = cv2.imread(bg_path)
    bg_w, bg_h = bg_img.shape[1], bg_img.shape[0]
    bg_wh_ratio = bg_h / bg_w
    bg_w = max(800, bg_w)
    bg_h = int(bg_w * bg_wh_ratio)
    bg_img = cv2.resize(bg_img, (bg_w, bg_h))
    ## bg area
    point = detect_bg(bg_img, color)
    before_x = point[0]
    ## fg-width范围根据 bg-width确定
    w_range = [bg_w//10, bg_w//3]
    ## num_fg
    n_fg = random.randint(1,8)
    for i in range(n_fg):
        fg_path = base_dir + '/' + fg_name + '/fg' + str(random.randint(1,fg_max_n)) + '.jpeg'
        # fg_img = cv2.imread(fg_path)
        fg_img = seg_image(fg_seg_config, fg_path)
        fg_w, fg_h = fg_img.shape[1], fg_img.shape[0]
        fg_wh_ratio = fg_h / fg_w
        (loc_x, loc_y, w, h), continue_generate = generate_fg_wh(w_range, fg_wh_ratio, before_x, point, bg_w, bg_h)
        if not continue_generate:
            break
        before_x  = loc_x + w
        bg_img = combine_picture(bg_img, fg_img, (w, h), (loc_x, loc_y))

    output_path = base_dir + '/' + 'output/' + str(int(time.time())) + '_' + str(random.randint(0,1000000)) + '.jpeg'
    cv2.imwrite(output_path, bg_img)
    return output_path


## 抠图+合成-v1   
def clear_and_combine_v1(base_dir, bg_path, fg_path_list, color, is_remove_badcase=True, use_removebg=False):
    '''
    parameter:
        -base_dir:图片路径
        -bg_path:bg路径
        -fg_path_list:fg路径list
        -color:需要检测的颜色
    return:
        output_path:合成后图片路径
        fg_out_path_list：去除背景后的前景存储路径
    '''
    fg_seg_config = 'inference_model/deploy.yaml'
    ## bg处理
    bg_img = cv2.imread(bg_path)
    bg_w, bg_h = bg_img.shape[1], bg_img.shape[0]
    bg_wh_ratio = bg_h / bg_w
    bg_w = max(800, bg_w)
    bg_h = int(bg_w * bg_wh_ratio)
    bg_img = cv2.resize(bg_img, (bg_w, bg_h))
    ## bg area
    point = detect_bg(bg_img, color)
    before_x = point[0]
    ## fg-width范围根据 bg-width确定
    w_range = [bg_w//10, bg_w//3]
    ## num_fg
    n_fg = len(fg_path_list)
    fg_out_path_list = []
    for i in range(n_fg):
        fg_path = fg_path_list[i]
        if use_removebg:
            fg_img, fg_out_path = seg_image(fg_seg_config, fg_path, save=True)
        else:
            fg_img = cv2.imread(fg_path)
        fg_out_path_list.append(fg_out_path)
        fg_w, fg_h = fg_img.shape[1], fg_img.shape[0]
        fg_wh_ratio = fg_h / fg_w
        (loc_x, loc_y, w, h), continue_generate = generate_fg_wh(w_range, fg_wh_ratio, before_x, point, bg_w, bg_h,i)
        if not continue_generate:
            break
        before_x  = loc_x + w
        bg_img = combine_picture(bg_img, fg_img, (w, h), (loc_x, loc_y))

    output_path = base_dir + '/' + 'output/' + str(int(time.time())) + '_' + str(random.randint(0,1000000)) + '.jpeg'
    cv2.imwrite(output_path, bg_img)
    return output_path, fg_out_path_list



#############################以下是demo2################################
def clear_and_combine_v2(base_dir, bg_name, fg_list, fg_n_dict, color, is_remove_badcase=True, use_removebg=False):
    '''
    parameter:
        -base_dir:图片路径
        -bg_name:bg名
        -fg_list:fg-name-list
        -fg_n_dict:每个fg库中前景图片数,key=fg_name, value=fg_n
        -color:需要检测的颜色
    return:
        output_path:合成后图片路径
        bg_path:背景图路径
    '''
    fg_seg_config = 'inference_model/deploy.yaml'
    ## bg处理
    bg_n = len(os.listdir(base_dir + '/bg/' + bg_name))
    bg_number = str(random.randint(1,bg_n))
    bg_path = base_dir + '/bg/' + bg_name + '/bg' + bg_number + '.jpeg'
    bg_key = bg_name +  bg_number
    bg_img = cv2.imread(bg_path)
    bg_w, bg_h = bg_img.shape[1], bg_img.shape[0]
    bg_wh_ratio = bg_h / bg_w
    bg_w = max(800, bg_w)
    bg_h = int(bg_w * bg_wh_ratio)
    bg_img = cv2.resize(bg_img, (bg_w, bg_h))
    ## bg area
    point = detect_bg(bg_img, color)
    before_x = point[0]
    ## fg-width范围根据 bg-width确定
    w_range = [bg_w//10, bg_w//3]
    ## num_fg:[len(fg_list),8]
    n_base_fg = len(fg_list)
    n_fg = random.randint(n_base_fg,8)
    for i in range(n_fg):
        if i < n_base_fg:
            idx = i
        else:
            idx = random.randint(0, n_base_fg-1)
        fg_name = fg_list[idx]
        fg_n = fg_n_dict[fg_name]
        if is_remove_badcase:
            fg_path = remove_badcase(base_dir, bg_key, fg_name, fg_n)
        else:
            fg_path = base_dir + '/fg/' + fg_name + '/fg' + str(random.randint(1,fg_n)) + '.jpeg'
        if use_removebg:
            fg_img = seg_image(fg_seg_config, fg_path)
        else:
            fg_img = cv2.imread(fg_path)
            print('not use remove-bg model')
        fg_w, fg_h = fg_img.shape[1], fg_img.shape[0]
        fg_wh_ratio = fg_h / fg_w
        (loc_x, loc_y, w, h), continue_generate = generate_fg_wh(w_range, fg_wh_ratio, before_x, point, bg_w, bg_h, i)
        if not continue_generate:
            break
        before_x  = loc_x + w
        bg_img = combine_picture(bg_img, fg_img, (w, h), (loc_x, loc_y))

    output_path = base_dir + '/' + 'output/' + str(int(time.time())) + '_' + str(random.randint(0,1000000)) + '.jpeg'
    cv2.imwrite(output_path, bg_img)
    return output_path, bg_path


def description2pix(description, base_dir='picture_hub', bg_path=None, fg_path_list=None, combine_mode=1,is_remove_badcase=True, use_removebg=False):
    '''
    parameter:
        -description:用户输入语料
        -base_dir:图片路径
        -bg_path:bg路径
        -fg_path_list：fg路径list
        -combine_mode：合成方式1，1或者2
    return:
        output_path:合成后图片路径
        fg_out_path_list：fg提取前景后存储路径
    '''
    ## 1.根据description获取检测颜色
    if '草原' in description:
        color = 'green'
        bg_name = 'grasslands'
    elif '沙' in description:
        color = 'orange'
        bg_name = 'sand'
    elif '工地' in description:
        color = 'orange'
        bg_name = 'construction'
    ## combine_mode-1
    if combine_mode == 1:
        output_path, fg_out_path_list = clear_and_combine_v1(base_dir, bg_path, fg_path_list, color)
        return output_path, fg_out_path_list
    ## combine_mode-2
    elif combine_mode == 2:
        ## 1.获取各个fg库中的图片数
        fg_list = []
        fg_n_dict = {}
        if '羊' in description:
            fg_list.append('sheep')
            fg_n_dict['sheep'] = len(os.listdir(base_dir + '/fg/sheep'))
        if '牛' in description:
            fg_list.append('ox')
            fg_n_dict['ox'] = len(os.listdir(base_dir + '/fg/ox'))
        if '人' in description:
            fg_list.append('people')
            fg_n_dict['people'] = len(os.listdir(base_dir + '/fg/people'))
        if '马' in description:
            fg_list.append('horse')
            fg_n_dict['horse'] = len(os.listdir(base_dir + '/fg/horse'))
        output_path, bg_path = clear_and_combine_v2(base_dir, bg_name, fg_list, fg_n_dict, color, is_remove_badcase, use_removebg)
        return output_path, bg_path

def clear_and_combine_test(base_dir, bg_path, fg_path_list, color):
    '''
    parameter:
        -base_dir:图片路径
        -bg_path:bg路径
        -fg_path_list:fg路径list
        -color:需要检测的颜色
    return:
        output_path:合成后图片路径
        fg_out_path_list：去除背景后的前景存储路径
    '''
    fg_seg_config = 'inference_model/deploy.yaml'
    ## bg处理
    bg_img = cv2.imread(bg_path)
    bg_w, bg_h = bg_img.shape[1], bg_img.shape[0]
    bg_wh_ratio = bg_h / bg_w
    bg_w = max(800, bg_w)
    bg_h = int(bg_w * bg_wh_ratio)
    bg_img = cv2.resize(bg_img, (bg_w, bg_h))
    ## bg area
    point = detect_bg(bg_img, color)
    before_x = point[0]
    ## fg-width范围根据 bg-width确定
    w_range = [bg_w//10, bg_w//3]
    ## num_fg
    n_fg = len(fg_path_list)
    fg_out_path_list = []
    for i in range(n_fg):
        fg_path = fg_path_list[i]
        fg_img, fg_out_path = seg_image(fg_seg_config, fg_path, save=True)
        fg_out_path_list.append(fg_out_path)
        fg_w, fg_h = fg_img.shape[1], fg_img.shape[0]
        fg_wh_ratio = fg_h / fg_w
        (loc_x, loc_y, w, h), continue_generate = generate_fg_wh(w_range, fg_wh_ratio, before_x, point, bg_w, bg_h,i)
        if not continue_generate:
            break
        before_x  = loc_x + w
        bg_img = combine_picture(bg_img, fg_img, (w, h), (loc_x, loc_y))

    # output_path = base_dir + '/' + 'output/' + str(int(time.time())) + '_' + str(random.randint(0,1000000)) + '.jpeg'
    # cv2.imwrite(output_path, bg_img)
    return bg_img 

## 批量抠图处理：from fg_origin to fg
def seg_image_all(base_dir='picture_hub', input_dir ='fg_origin', output_dir = 'fg'):
    config = 'inference_model/deploy.yaml'
    fg_list = ['people','ox','sheep','horse']
    for fg in fg_list:
        img_dir = base_dir + '/' + input_dir + '/' + fg
        fg_n = len(os.listdir(img_dir))
        for idx in range(1, fg_n + 1):
            img_path = img_dir + '/fg' + str(idx) + '.jpeg'
            out_img = seg_image(config, img_path, save = False)
            output_path = base_dir + '/' + output_dir + '/' + fg + '/fg' + str(idx) + '.jpeg'
            cv2.imwrite(output_path,out_img)
            
   

#############################以下是demo1########################
# 1.搜索背景接口
def search_bg(description, n_picture=4, base_dir='picture_hub'):
    '''
    parameter:
        -description:用户输入语料
        -n_picture:图片张数
        -base_dir:图片路径
    return:
        bg_path_list:bg图片路径list,[bg_path1,bg_path2...]
    '''
    if '草原' in description:
        bg_name = 'grasslands'
    elif '沙' in description:
        bg_name = 'sand'
    elif '工地' in description:
        bg_name = 'construction'
    bg_n = len(os.listdir(base_dir + '/bg/' + bg_name))
    bg_choice = np.random.choice(range(1,bg_n+1), n_picture, replace=False)
    bg_path_list = [base_dir + '/bg/' + bg_name + '/bg' + str(ii) + '.jpeg' for ii in bg_choice]
    return bg_path_list 

# 2.搜索前景接口-限制一种前景
def search_fg(description, n_picture=4, base_dir='picture_hub'):
    '''
    parameter:
        -description:用户输入语料
        -n_picture:图片张数
        -base_dir:图片路径
    return:
        fg_path_list:fg图片路径list,[fg_path1,fg_path2...]
    '''
    if '羊' in description:
        fg_name = 'sheep'
    elif '牛' in description:
        fg_name = 'ox'
    elif '人' in description:
        fg_name = 'people'
    elif '马' in description:
        fg_name = 'horse'
    fg_n = len(os.listdir(base_dir + '/fg_origin/' + fg_name))
    fg_choice = np.random.choice(range(1,fg_n+1), n_picture, replace=False)
    fg_path_list = [base_dir + '/fg_origin/' + fg_name + '/fg' + str(ii) + '.jpeg' for ii in fg_choice]
    return fg_path_list 

# 3.前景提取接口
def get_fg(fg_path_list, func_mode = 1):
    '''
    parameter:
        -fg_path_list:待提取fg_path_list
        -func_mode:提取方法
        -base_dir:图片路径
    return:
        fg_out_path_list:提取后后fg图片路径list,[fg_path1,fg_path2...],固定为picture_hub/fg_temp/原名_removebg.jpeg
    '''
    fg_seg_config = 'inference_model/deploy.yaml'
    fg_out_path_list = []
    for img_path  in fg_path_list:
        if func_mode == 1:
            _, fg_out_path = seg_image(fg_seg_config, img_path, save = True)
            fg_out_path_list.append(fg_out_path)
    return fg_out_path_list


# 4.图片合成接口-全部拼接
def combine_pic_testv(description, bg_path_list, fg_rm_path_list, func_mode = 1, base_dir='picture_hub'):
    '''
    parameter:
        -description:用户输入语料
        -base_dir:图片路径
        -bg_path_list:bg路径
        -fg_rm_path_list:移除背景后的前景路径
        -func_mode:合成方法
    return:
        out_path_list：合成路径list，为picture_hub/output_test_v
    '''
    ## 1.根据description获取检测颜色
    if '草原' in description:
        color = 'green'
    elif '沙' in description:
        color = 'orange'
    elif '工地' in description:
        color = 'orange'
    ## 2.合成
    out_path_list = []
    for idx in range(len(bg_path_list)):
        bg_path = bg_path_list[idx]
        bg_img = cv2.imread(bg_path)
        bg_w, bg_h = bg_img.shape[1], bg_img.shape[0]
        bg_wh_ratio = bg_h / bg_w
        bg_w = max(800, bg_w)
        bg_h = int(bg_w * bg_wh_ratio)
        bg_img = cv2.resize(bg_img, (bg_w, bg_h))
        ## bg area
        point = detect_bg(bg_img, color)
        before_x = point[0]
        ## fg-width范围根据 bg-width确定
        w_range = [bg_w//10, bg_w//3]
        ## num_fg
        n_fg = len(fg_rm_path_list)
        for i in range(n_fg):
            fg_path = fg_rm_path_list[i]
            fg_img = cv2.imread(fg_path)
            fg_w, fg_h = fg_img.shape[1], fg_img.shape[0]
            fg_wh_ratio = fg_h / fg_w
            (loc_x, loc_y, w, h), continue_generate = generate_fg_wh(w_range, fg_wh_ratio, before_x, point, bg_w, bg_h)
            if not continue_generate:
                break
            before_x  = loc_x + w
            ## 按位拼接
            if func_mode == 1:
                bg_img = combine_picture(bg_img, fg_img, (w, h), (loc_x, loc_y))
            ## 直接相加
            elif func_mode == 2:
                bg_img = combine_picture_v2(bg_img, fg_img, (w, h), (loc_x, loc_y))
        output_path = base_dir + '/' + 'output_test_v/' + str(int(time.time())) + '_' + str(random.randint(0,1000000)) + '.jpeg'
        cv2.imwrite(output_path, bg_img)
        out_path_list.append(output_path)
    return output_path

