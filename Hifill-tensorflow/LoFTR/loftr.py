

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from .util import *
from .findJpg import combine_filenames
from .src.utils.plotting import make_matching_figure
from .src.loftr import LoFTR, default_cfg

import random

def loftrGenerate(img0_pth, img1_pth, dirName,blocksXY):
    print("----", dirName)
    image_pair = [img0_pth, img1_pth]
    matcher = LoFTR(config=default_cfg)
    image_type = 'outdoor'

    if image_type == 'indoor':
      matcher.load_state_dict(torch.load("/weights/indoor_ds.ckpt")['state_dict'])
    elif image_type == 'outdoor':
      matcher.load_state_dict(torch.load("/storage/tf22/Hifill-tensorflow/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    else:
      raise ValueError("Wrong image_type is given.")


    #matcher = matcher.eval().cuda()
    matcher = matcher.eval()
    img0_origin = cv2.imread(image_pair[0])
    img1_origin = cv2.imread(image_pair[1])
    img0_origin = cv2.resize(img0_origin, (640, 480))
    img1_origin = cv2.resize(img1_origin, (640, 480))

    '''
        tmp_pth = "/storage/LoFTR/results/originSet/"+dirName
    if not os.path.exists(tmp_pth+returnBaseName(img0_pth)):
        os.makedirs(tmp_pth, exist_ok=True)
        cv2.imwrite(tmp_pth+"/"+returnBaseName(img0_pth), img0_origin) # store resized origin_img0

    '''


    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None] / 255.
    img1 = torch.from_numpy(img1_raw)[None][None] / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # blocksXY = block_region(mkpts0)
    # blocksXY = generate_random_rectangle(640, 480)
    if (blocksXY == None):
        return
    
    newName = combine_filenames(returnBaseName(img0_pth), returnBaseName(img1_pth))
    # addWhiteBlock(bloackregion[0],bloackregion[1],bloackregion[2],bloackregion[3],"image0","whiteBlock")


    H, status = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)

    result_img0 = img0_origin.copy()
    target_image1 = img1_origin
    result_img0 = cv2.rectangle(result_img0, (blocksXY[0], blocksXY[1]), (blocksXY[2], blocksXY[3]), (255, 255, 255), -1)
    for dst_y in range(blocksXY[1], blocksXY[3]):
        for dst_x in range(blocksXY[0], blocksXY[2]):
            coordinate = np.array([[dst_x], [dst_y], [1]])
            trans_coordinate = np.dot(H, coordinate)
            # 齊次座標轉二維
            trans_coordinate = trans_coordinate / trans_coordinate[2]
            src_x, src_y = int(trans_coordinate[0]), int(trans_coordinate[1])
            if ((src_x > 639 or src_x < 0) or (src_y > 479 or src_y < 0)):
                continue
            elif ((dst_x > 639 or dst_x < 0) or (dst_y > 479 or dst_y < 0)):
                continue
            result_img0[dst_y, dst_x] = target_image1[src_y, src_x] # write pixel




    filename = dirName + "/fixed.jpg"   # 貼上後的圖= img0+img1.jpg
        # Save the image using OpenCV's imwrite function
    print("----",dirName)
    cv2.imwrite(filename, result_img0)
    print("store fixed image in \"" + filename + "\"")
    filename = dirName + "/mask.jpg"
    makeBlackImg(blocksXY[0],blocksXY[1],blocksXY[2],blocksXY[3], filename)

