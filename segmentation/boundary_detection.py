import cv2
import numpy as np

from .preprocessing import *

def lung_boundary_detection(img, preprocessing='bcet'): 
    if(preprocessing == 'bcet'):
        img = bcet(img)
    elif(preprocessing == 'clahe'):
        img = clahe(img)
    elif(preprocessing == 'clahe_lab'):
        img = clahe_lab(img)
    else:
        raise Exception('Invalid preprocessing method')

    # 1. Otsu thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)

    # 2. Convex Hull formation
    # 2.1. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 2.2. Convex hull
    hull_list = []
    hull_areas = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
        hull_area = cv2.contourArea(hull)
        hull_areas.append(hull_area)

    hull_areas = np.array(hull_areas)
    hull_areas = np.sort(hull_areas)
    top_areas = np.array(hull_areas)[-5:-1]
    # 3. Draw contours and hull result
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (np.random.randint(127, 256), np.random.randint(127, 256),
                 np.random.randint(127,256))

        # Filter contour by area
        cnt_area = cv2.contourArea(hull_list[i])
        if(cnt_area not in top_areas):
            continue

        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color, thickness=cv2.FILLED)

    return drawing, thresh
