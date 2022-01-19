# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:08:25 2021

@author: Jesús S. Alegre
"""
# use this command to install open cv2
# pip install opencv-python

# use this command to install PIL
# pip install Pillow

import cv2
from PIL import Image

def mark_region(image_path):
    
    im = cv2.imread(image_path)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    line_items_coordinates = []
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)

        if y >= 600 and x <= 1000:
            if area > 10000:
                image = cv2.rectangle(im, (x,y), (3200, y+h), color=(255,0,255), thickness=3)
                line_items_coordinates.append([(x,y), (3200, y+h)])

        if y >= 2400 and x<= 2000:
            image = cv2.rectangle(im, (x,y), (3200, y+h), color=(255,0,255), thickness=3)
            line_items_coordinates.append([(x,y), (3200, y+h)])


    return image, line_items_coordinates

image,line_items_coordinates = mark_region('page_1.jpg')

# see the results
cv2.imshow('areas', image)
cv2.waitKey(0)
cv2.imwrite('areas2.jpg', image)
cv2.destroyAllWindows()


# get co-ordinates to crop the image
c = line_items_coordinates[1]

# # cropping image img = image[y0:y1, x0:x1]
# img = image[c[0][1]:c[1][1], c[0][0]:c[1][0]]    

# plt.figure(figsize=(10,10))
# plt.imshow(img)