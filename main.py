import math
import random
import time

import cv2
import numpy as np
# from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt




def get_contour_centers(contours: np.ndarray) -> np.ndarray:
    """
    Calculate the centers of the contours
    :param contours: Contours detected with find_contours
    :return: object centers as numpy array
    """

    if len(contours) == 0:
        return np.array([])

    # ((x, y), radius) = cv2.minEnclosingCircle(c)
    centers = np.zeros((len(contours), 2), dtype=np.int16)
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        centers[i] = center
    return centers


def photoMain():
    videoPath = "D:\\Facultate\\Semestrul6\\Tehnici de realizare a sistemelor inteligente\\bcfm\\img1"

    img = cv2.imread(videoPath+".jpeg", cv2.IMREAD_COLOR)
    width=int(img.shape[1]/3)
    height=int(img.shape[0]/3)

    frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    laplac=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=3)
    canny_image = cv2.Canny(blur, 100, 140)

    ret, mask = cv2.threshold(canny_image, 70, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(mask, 1, np.pi / 100, 30, np.array([]), minLineLength=5, maxLineGap=2)

    # center=get_contour_centers()

    # ret, thresh = cv2.threshold(hsv, 170, 255, 0)

    lower = np.array([20, 30, 0], dtype="uint8")
    higher = np.array([180, 250, 255], dtype="uint8")
    thresh = cv2.inRange(hsv, lower, higher)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(hsv, contours, -1, (0, 255, 0), 3)

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("Object detection", laplac)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(videoPath+"out.jpeg", frame)

import  numpy

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False


def photoMain2():
    videoPath = "D:\\Facultate\\Semestrul6\\Tehnici de realizare a sistemelor inteligente\\bcfm\\img6"

    img = cv2.imread(videoPath+".jpeg", cv2.IMREAD_COLOR)
    width=int(img.shape[1]/3)
    height=int(img.shape[0]/3)

    frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_orig=gray
    # Smoothing without removing edges.
    gray_filtered = cv2.bilateralFilter(gray, 10, 130, 250)

    # Applying the canny filter
    # edges = cv2.Canny(gray, 100, 120)
    edges_filtered = cv2.Canny(gray_filtered,25, 75)

    # kernel = np.ones((3, 3), np.uint8)
    # end_frame = cv2.dilate(edges_filtered, kernel, iterations=1)
    # for el in range(10):
    #     kernel = np.ones((3, 3), np.uint8)
    #     end_frame=cv2.dilate(end_frame, kernel, iterations=1)
    #     kernel = np.ones((10, 10), np.uint8)
    #     for el in range(1):
    #         end_frame=cv2.morphologyEx(end_frame, cv2.MORPH_CLOSE, kernel)

    # kernel = np.ones((3, 3), np.uint8)
    # end_frame = cv2.dilate(edges_filtered, kernel, iterations=1)
    # for el in range(30):
    #     kernel = np.ones((3, 3), np.uint8)
    #     end_frame=cv2.dilate(end_frame, kernel, iterations=1)
    #     kernel = np.ones((2, 2), np.uint8)
    #     end_frame=cv2.erode(end_frame, kernel,iterations=1)




    contours, hier = cv2.findContours(edges_filtered, cv2.RETR_EXTERNAL, 2)

    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    cv2.drawContours(gray, unified, -1, (0, 255, 0), 2)
    cv2.drawContours(gray, unified, -1, 255, -1)





    # Stacking the images to print them together for comparison
    images = np.hstack((gray,edges_filtered))

    # Display the resulting frame
    # cv2.imshow('Frame', edges)

    cv2.imshow("Object detection", images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(videoPath+"out.jpeg", frame)

def photoMain3():
    videoPath = "D:\\Facultate\\Semestrul6\\Tehnici de realizare a sistemelor inteligente\\bcfm\\img1"

    img = cv2.imread(videoPath+".jpeg", cv2.IMREAD_COLOR)
    width=int(img.shape[1]/3)
    height=int(img.shape[0]/3)

    frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change to grayscale

    # frame= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ret, thresh1 = cv2.threshold(frame, 100, 255,
                                 cv2.THRESH_BINARY)  # the value of 15 is chosen by trial-and-error to produce the best outline of the skull
    kernel = np.ones((5, 5), np.uint8)  # square image kernel used for erosion
    erosion = cv2.erode(thresh1, kernel, iterations=1)  # refines all edges in the binary image

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,
                               kernel)  # this is for further removing small noises and holes in the image


    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # find contours with simple approximation




    cv2.imshow("Object detection", closing)
    cv2.imshow("Object detection2", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(videoPath+"out.jpeg", frame)

def photoMain4():
    videoPath = "D:\\Facultate\\Semestrul6\\Tehnici de realizare a sistemelor inteligente\\bcfm\\img3"

    img = cv2.imread(videoPath+".jpeg", cv2.IMREAD_COLOR)
    width=int(img.shape[1]/3)
    height=int(img.shape[0]/3)

    frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0, 60, 40])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 60, 40])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0



    cv2.imshow("Object detection", output_hsv)
    cv2.imshow("Object detection2", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(videoPath+"out.jpeg", frame)

# main()

photoMain2()
