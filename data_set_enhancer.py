import numpy as np
import sys
import filters
import cv2
import os
from face_detect import face_detect_viola_jones
from os import listdir, getcwd
from os.path import isfile, isdir, join
from parse import img_data_to_csv

def write_on_img(im, text):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (50, 50) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method 
    im = cv2.putText(im, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return im

C = 0.3
''' Get faces from public area and crop to folder'''
if sys.argv[1] == "get_faces":
    filter_name = sys.argv[2]
    im = cv2.imread(sys.argv[3], 0)
    out_path = sys.argv[4]
    faces_loc = face_detect_viola_jones(im, 1.1, (5, 5), 1)
    k = 0
    for (x, y, w, h) in faces_loc:
        crop_im = im[y:y+h, x:x+w]
        resized_im = cv2.resize(crop_im, (64, 64), interpolation=cv2.INTER_AREA)
        if filter_name == "laplace":
            resized_im = filters.laplace_sharp(resized_im, C)
        elif filter_name == "clhae":
            resized_im = filters.apply_clahe(resized_im)
        elif filter_name == "none":
            pass
        else:
            print("filter not found")
            quit()
         
        cv2.imwrite(out_path + '/' + str(k) + '.png', resized_im)
        k += 1

elif sys.argv[1] == "test_image":
    filter_name = sys.argv[2]
    im = cv2.imread(sys.argv[3])
    im_gray_scale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces_loc = face_detect_viola_jones(im_gray_scale, 1.1, (5, 5), 1)
    k = 1
    print(str(len(faces_loc)) + " faces found")
    for (x, y, w, h) in faces_loc:
        crop_im = im_gray_scale[y:y+h, x:x+w]
        resized_im = cv2.resize(crop_im, (64, 64), interpolation=cv2.INTER_AREA)
        if filter_name == "laplace":
            resized_im = filters.laplace_sharp(resized_im, C)
        elif filter_name == "clhae":
            resized_im = filters.apply_clahe(resized_im)
        elif filter_name == "none":
            pass
        else:
            print("filter not found")
            quit()
        print("processing face " + str(k))
        cv2.imwrite('face.png', resized_im)
        os.system("./cnn_c test face.png > results.txt")
        
        masked_prob = 0.0
        unmasked_prob = 0.0
        with open("results.txt", "r") as f:
            masked_prob = float(f.readline().split(' ')[2])
            unmasked_prob = float(f.readline().split(' ')[2])
        if masked_prob > 0.5:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #print("masked prob: " + str(masked_prob) + " unmasked prob: " + str(unmasked_prob))
        os.system("rm face.png")
        os.system("rm results.txt")
        k += 1
    cv2.imwrite('out.png', im)

elif sys.argv[1] == "make_data_set":

    filter_name = sys.argv[2]
    src_path = sys.argv[3]

    labels_list = [f for f in listdir(src_path) if isdir(join(src_path, f))]

    data = []
    labels = []
    l = 0

    for label in labels_list:
        files_list = [f for f in listdir(join(src_path, label))]
        for file in files_list:
            file_path = join(src_path, label, file)
            im = cv2.imread(file_path, 0)
            
            if filter_name == "laplace":
                im = filters.laplace_sharp(im, C)
            elif filter_name == "clhae":
                im = filters.apply_clahe(im)
            elif filter_name == "none":
                pass
            else:
                print("filter not found")
                quit()

            if im.shape == (64, 64):
                im = np.reshape(im, 64 * 64)
                data.append(im)
                labels.append(l)
            else:
                print(im.shape)
        l += 1

    print("Labels : " + ' '.join(labels_list))
    print("Samples: " + str(len(labels)))
    img_data_to_csv(data, labels, labels_list, 'data.csv')