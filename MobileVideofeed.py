#To run this script install IP WEBCAM on your Android Device (your computer and mobile has to be on the same network!)
#Once its downloaded go to select server and click on 'How do I connect?' on the top left corner
#Click on 'connect directly'
#If you are using Wifi-Router click on that option
#Go to the ip address suggested
#On video renderer select javascript and right click on the video feed that appears
#Copy the link till shot.jpg and truncate the rest
#Follow the code below


import urllib
import cv2
import numpy as np

url = "http://192.168.0.3:8080/shot.jpg" #replace with your url

def canny_edge(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur =  cv2.GaussianBlur(gray,(kernel,kernel),0)
    canny_img = cv2.Canny(blur,10,200)
    return canny_img

while True:
    #get the repsonse from url
    img_resp = urllib.urlopen(url)
    #convert it into a numpy image
    imagenp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
    #convert it to a cv2 image
    img = cv2.imdecode(imagenp,-1)
    # canny_edge detection
    img = canny_edge(img)
    #display image
    cv2.imshow('frame_window',img)
    if ord('q')==cv2.waitKey(10):
        exit(0)
