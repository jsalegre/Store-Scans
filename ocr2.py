# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:03:40 2021

@author: Jesús S. Alegre
"""
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import numpy as np
import cv2

texts = []
image = Image.open('page_1.jpg')
with PyTessBaseAPI() as api:
    api.SetImage(image)
    boxes = api.GetComponentImages(RIL.TEXTLINE, True) # or others values from RIL, depends on your image
    print('Found {} image components.'.format(len(boxes)))
    for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object
        # box is a dict with x, y, w and h keys
            delta = image.size[0] / 200
            api.SetRectangle(box['x'] - delta, box['y'] - delta, box['w'] + 2 * delta, box['h'] + 2 * delta)
            # widening the box with delta can greatly improve the text output
            ocrResult = api.GetUTF8Text()
            texts.append(ocrResult)
            
            if 'Serial Number' in ocrResult:
                im = np.array(im)       
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.imshow('areas-ocr', im)
                cv2.waitKey(0)
                cv2.imwrite('areas-ocr.jpg', im)
                cv2.destroyAllWindows()
                
                break
            


            
# from tesserocr import PyTessBaseAPI, RIL

# images = ['page_1.jpg']

# with PyTessBaseAPI() as api:
#     for img in images:
#         api.SetImageFile(img)
#         text = api.GetUTF8Text()
#         #print(api.AllWordConfidences())
        
# api is automatically finalized when used in a with-statement (context manager).
# otherwise api.End() should be explicitly called when it's no longer needed.