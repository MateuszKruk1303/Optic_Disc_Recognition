#!/usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PySimpleGUI as sg
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageTk
import io

filename="Resources/eye.jpg"

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


# Get the folder containin:g the images from the user
folder = sg.popup_get_folder('Image folder to open', default_path='')
if not folder:
    sg.popup_cancel('Cancelling')
    raise SystemExit()

# PIL supported image types
img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")

# get list of files in folder
flist0 = os.listdir(folder)

# create sub list of image files (no sub folders, no wrong file types)
fnames = [f for f in flist0 if os.path.isfile(
    os.path.join(folder, f)) and f.lower().endswith(img_types)]

num_files = len(fnames)                # number of iamges found
if num_files == 0:
    sg.popup('No files in folder')
    raise SystemExit()

del flist0                             # no longer needed

# ------------------------------------------------------------------------------
# use PIL to read data of one image
# ------------------------------------------------------------------------------


def get_img_data(f, maxsize=(1200, 850), first=False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)
# ------------------------------------------------------------------------------


# make these 2 elements outside the layout as we want to "update" them later
# initialize to the first file in the list
filename = os.path.join(folder, fnames[0])  # name of first file in list
image_elem = sg.Image(data=get_img_data(filename, first=True))
filename_display_elem = sg.Text(filename, size=(80, 3))
file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15, 1))

# define layout, show and read the form
col = [[filename_display_elem],
       [image_elem]]

col_files = [[sg.Listbox(values=fnames, change_submits=True, size=(60, 30), key='listbox')],
             [sg.Button('Next', size=(8, 2)), sg.Button('Prev', size=(8, 2)), file_num_display_elem],
             [sg.InputText("promien minimalny",key="-IN1-"),sg.InputText("przedzial w gore",key="-IN2-")],
             [sg.Button('RUN!',size=(8,2)),sg.Text("enter values and run",key="output_text")]]

layout = [[sg.Column(col_files), sg.Column(col)]]

window = sg.Window('Image Browser', layout, return_keyboard_events=True,
                   location=(0, 0), use_default_focus=False)

# loop reading the user input and displaying image, filename
i = 0
while True:
    # read the form
    event, values = window.read()
    print(event, values)
    # perform button and keyboard operations
    if event == sg.WIN_CLOSED:
        break
    elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34'):
        i += 1
        if i >= num_files:
            i -= num_files
        filename = os.path.join(folder, fnames[i])
    elif event in ('RUN!'):
        print(filename)
        #run script with parameters-------------------------------------------------
        img = Image.open(filename).convert("L")
      
        r_min=values["-IN1-"]
        r_max=values["-IN2-"]

        if((r_min.isnumeric()) and (r_max.isnumeric())):
                r_min=int(values["-IN1-"])
                r_max=int(values["-IN2-"])

                
                CircleDetection(filename, img, r_min, r_max)
              

                print(r_min)
                print(r_max)
        else:
            print("values should be an integer")
        
        

    elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
        i -= 1
        if i < 0:
            i = num_files + i
        filename = os.path.join(folder, fnames[i])
    elif event == 'listbox':            # something from the listbox
        f = values["listbox"][0]            # selected filename
        filename = os.path.join(folder, f)  # read this file
        i = fnames.index(f)                 # update running index
    else:
        filename = os.path.join(folder, fnames[i])

    # update window with new image
    image_elem.update(data=get_img_data(filename, first=True))
    # update window with filename
    filename_display_elem.update(filename)
    # update page display
    file_num_display_elem.update('File {} of {}'.format(i+1, num_files))

window.close()
