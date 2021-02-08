import cv2
import numpy as np
import glob
import random
import csv
import requests
import os, shutil


# Reads all the image paths and shuffles them inside a list
def find_image_paths(limit = None):

    all_paths = glob.glob("./Data/lfw_funneled/**/*.jpg")

    random.shuffle(all_paths)

    if limit:
        return all_paths[0:limit]
    else:
        return all_paths

# Labels the images using a haarcascade and stores the labels in the "filename"
def label_images(paths,filename):

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    
    f = open(filename,"w")

    for path in paths:
        image = cv2.imread(path)
        eyes = eye_cascade.detectMultiScale(image,1.1,3)
        for (x,y,w,h) in eyes:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.imshow("image",image)
        
        if (cv2.waitKey(0) & 0xFF) == ord('s'): # Saves the image
            print("save")
            for eye in eyes:
                (x1,y1,width,height) = eye
                f.write(f"{path},{x1},{y1},{x1+width},{y1+height},eye\n")
                print(eye)
            print(path)
        elif (cv2.waitKey(0) & 0xFF) == ord('d'): # Discards the image
            print("don't save")
        elif (cv2.waitKey(0) & 0xFF) == ord('q'): # Stops labeling
            f.close()
            return
        
    f.close()

# Creates a directory of all the chosen images and creates a new csv file with paths to it
def copy_files(filename):
    data_list = []
    os.makedirs("FilteredData", exist_ok=True)
    with open(filename,"r") as f:
        all_lines = f.readlines()
        for line in all_lines:
            data = line.split(",",1)
            path = data[0]
            if path != '':
                new_path = shutil.copy(path,"FilteredData")
                print(new_path)
                data[0] = os.path.join(new_path)
                data_list.append(data)

    with open("data.csv","w") as f:
        for line in data_list:
            f.write(",".join(line))


if __name__ == "__main__":
    # Downloads the haarcascade
    if not os.path.exists("haarcascade_eye_tree_eyeglasses.xml"):
        r = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
        with open("haarcascade_eye_tree_eyeglasses.xml","w") as f:
            f.write(r.text)
        print("Done")

    paths = find_image_paths(limit = 1000)
    label_images(paths,"unprepared.csv")
    copy_files("unprepared.csv")
