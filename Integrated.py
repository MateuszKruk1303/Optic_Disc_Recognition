# img_viewer.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

import PySimpleGUI as sg
import os.path


# First the window layout in 2 columns

def detectCircles(image, edges, radius ,radiusrange=0):
    plt.imshow(image)
    plt.imshow(edges, cmap='gray')
    plt.show()
    rows, columns = edges.shape
    alternative_maxes = {}
    alternative_best = { "value" : 0, "rad": 0 }


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
        maxes = np.argwhere((img2buffer > 150) & (img2buffer < 255)).flatten()
        if (len(maxes) == 0):

            print('no maxes')
            potential_pixels = np.argwhere((img2buffer > 70) & (img2buffer < 255)).flatten()
            if(len(potential_pixels) > 0 or rad == radius+radiusrange - 1):
                if(len(potential_pixels) > 0):
                    highest_value_of_pixel = { "value" : 0, "x": 0, "y": 0 }
                    for i in range(0, len(potential_pixels), 2):
                        if(img2buffer[potential_pixels[i], potential_pixels[i+1]] > highest_value_of_pixel["value"]):
                            highest_value_of_pixel = { "value" : img2buffer[potential_pixels[i], potential_pixels[i+1]], "x": potential_pixels[i+1], "y": potential_pixels[i] }

                    alternative_maxes[rad] = highest_value_of_pixel
                    print(alternative_maxes)
                if(rad == radius+radiusrange - 1):
                    for key in alternative_maxes:
                        print(alternative_maxes)
                        if(alternative_maxes[key]['value'] > alternative_best['value']):
                            alternative_best['value'] = alternative_maxes[key]['value']
                            alternative_best['rad'] = key

                    # plt.imshow(img2buffer, cmap='gray')
                    # plt.show()

                    print(alternative_maxes[alternative_best["rad"]]["x"])
                    print(alternative_maxes[alternative_best["rad"]]["y"])
                    plt.imshow(img2buffer, cmap='gray')
                    plt.show()
                    cv2.circle(image, center=(alternative_maxes[alternative_best["rad"]]["x"], alternative_maxes[alternative_best["rad"]]["y"]), radius=int(alternative_best["rad"]), color=(255, 255, 255), thickness=2)
                    plt.imshow(image)
                    plt.show()
                    print(alternative_best)
                    print('most possible circle place')
                    print(alternative_maxes) 

            continue            
        else:
            print(maxes)
            plt.imshow(img2buffer, cmap='gray')
            plt.show()
            for i in range(0, len(maxes), 2):
                cv2.circle(image, center=(maxes[i + 1], maxes[i]), radius=rad, color=(255, 255, 255), thickness=2)

            plt.imshow(edges)
            plt.show()
            break


def CircleDetection(img_url, img, r_min, r_max):

    width, height = img.size
    denominator = 5
    new_size = (int(np.ceil(width/denominator)), int(np.ceil(height/denominator)))
    img = img.resize(new_size)
    plt.imshow(img)
    plt.show()



    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.1)
    plt.imshow(img)
    plt.show()

    img = img.filter(ImageFilter.GaussianBlur(5))
    plt.imshow(img)
    plt.show()

    img_Gray = np.asarray(img, dtype=np.uint8)
    imgGray = img_Gray.astype(np.uint8)

    for i in range(0, new_size[0]):
        for j in range(0, new_size[1]):
            if (imgGray[j,i] >= 210):
                imgGray[j,i] = 255
            else:
                imgGray[j,i] = 0

    edges = cv2.Canny(imgGray, 240, 250)
    base_image = cv2.imread(img_url)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    base_image_resized = cv2.resize(base_image, new_size)
    detectCircles(base_image_resized, edges, r_min, r_max)


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
            and f.lower().endswith((".png", ".jpg", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            print(filename)
            print(filename.index("Resources"))
            # filename = filename[filename.index("Resources"):len(filename):1]
            #window["-TOUT-"].update(filename)
            #window["-IMAGE-"].update(filename=filename)
            imagen = filename
        except Exception as ex:
            print(ex)
            pass
    elif event == "-RUN-":
        try:
            print(imagen)
            Image.open(imagen)
            img = Image.open(imagen).convert("RGB")
            print(values["-IN1-"])
            print(values["-IN2-"])
            r_min=int(values["-IN1-"])
            r_max=int(values["-IN2-"])
            CircleDetection(imagen, img, r_min, r_max)
            print(r_min+r_max)
        except:
            pass
        
            

window.close()