# img_viewer.py
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
from matplotlib import pyplot as plt
from PIL import Image

import PySimpleGUI as sg
import os.path


#global image
imagen = "Resources/120x2O100x2O50x2O20x4.png"
# First the window layout in 2 columns

# Gaussian Blur - kernel
def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    mask = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return mask


# Finding the intensity gradient of the image
def prewitt_operator(img):
    Gxx = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

    Gyy = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

    Gx = ndimage.filters.convolve(img, Gxx)
    Gy = ndimage.filters.convolve(img, Gyy)

    G = np.sqrt(np.square(Gx) + np.square(Gy))
    G = G / G.max() * 255
    theta = np.arctan2(Gy, Gx)

    return (G, theta)


# Tłumienie niemaksymalne
def non_max_suppression(img, D):
    M, N = img.shape
    I = np.zeros((M, N), dtype=np.uint8)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    I[i, j] = img[i, j]
                else:
                    I[i, j] = 0

            except IndexError as e:
                pass
    return I


def double_threshold(img, lowThresholdVal=0.05, highThresholdVal=0.1):
    highThreshold = img.max() * highThresholdVal
    lowThreshold = highThreshold * lowThresholdVal

    M, N = img.shape
    result = np.zeros((M, N), dtype=np.uint8)

    weak = np.uint8(25)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return (result, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img

#Funkcja wykrywająca okręgi o zadanym przedziale promienia

def detectCircles(edges, radius ,radiusrange=0):
    plt.imshow(edges, cmap='gray')
    plt.show()
    rows, columns = edges.shape
    for rad in range(radius, radius + radiusrange):
        img2buffer = np.zeros([rows, columns], dtype=np.uint8)
        for x in range(0, columns):
            for y in range(0, rows):
                if (edges[y, x] == 255):
                    for ang in range(0, 360):
                        t = (ang * np.pi) / 180
                        x0 = int(round(x + rad * np.cos(t)))
                        y0 = int(round(y + rad * np.sin(t)))
                        if (x0 < columns and x0 > 0 and y0 < rows and y0 > 0 and img2buffer[y0, x0] < 255):
                            img2buffer[y0, x0] += 1

        maxes = np.argwhere((img2buffer > 220) & (img2buffer < 255)).flatten()

        if (len(maxes) == 0):
            plt.imshow(img2buffer, cmap='gray')
            plt.show()
            print('no maxes')
            continue
        else:
            print(maxes)
            plt.imshow(img2buffer, cmap='gray')
            plt.show()
            for i in range(0, len(maxes), 2):
                cv2.circle(edges, center=(maxes[i + 1], maxes[i]), radius=rad, color=(255, 255, 255), thickness=2)

            plt.imshow(edges)
            plt.show()
            break


def CircleDetection(img):

    img_Gray = np.asarray(img, dtype=np.uint8)
    imgGray = img_Gray.astype(np.uint8)
    cv2.imshow("Grayscale IMG", imgGray)

    x, y = imgGray.shape
    result = gaussian_kernel(5, 1)
    img_gauss = np.zeros((x, y), dtype=np.uint8)
    img_gauss = convolve(imgGray, result)
    cv2.imshow("Smoothing image - Gaussian filter", img_gauss)
    cv2.waitKey(0)

    img_gauss = np.asarray(img_gauss, dtype=np.int32)
    img_prewitt = np.zeros((x, y), dtype=np.int32)
    img_prewitt, theta = prewitt_operator(img_gauss)
    img_prewitt = np.array(img_prewitt)
    img_prewitt = img_prewitt.astype(np.uint8)
    cv2.imshow("Prewitt operator - finding the intensity of the image", img_prewitt)
    cv2.waitKey(0)

    img_surp = np.zeros((x, y), dtype=np.uint8)
    img_surp = non_max_suppression(img_prewitt, theta)
    img_surp = np.array(img_surp)
    img_surp = img_surp.astype(np.uint8)
    cv2.imshow("Non-Maximum Surpression", img_surp)
    cv2.waitKey(0)

    img_threshold = np.zeros((x, y), dtype=np.uint8)
    img_threshold, weak, strong = double_threshold(img_surp)
    img_threshold = np.array(img_threshold)
    img_threshold = img_threshold.astype(np.uint8)
    cv2.imshow("Double Threshold", img_threshold)
    cv2.waitKey(0)

    img_edge = np.zeros((x, y), dtype=np.uint8)
    img_edge = hysteresis(img_threshold, weak, strong)
    cv2.imshow("Edge tracked by hysteresis", img_edge)
    cv2.waitKey(0)

    detectCircles(img_edge, 10, 4)


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
        sg.Button("Run!",enable_events=True,key="-RUN-"),
        sg.InputText("promien minimalny",key="-IN1-"),
        sg.InputText("przedzial w gore",key="-IN2-"),
        
             

    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
       
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)
            imagen = filename
            

        except:
            pass
    elif event == "-RUN-":
        try:
            print(imagen)
            img = Image.open(imagen).convert("L")
            #CircleDetection(img)
            print(values["-IN1-"])
            print(values["-IN2-"])
            r_min=float(values["-IN1-"])
            r_max=float(values["-IN2-"])
            print(r_min+r_max)
            
        except:
            pass
        
            

window.close()