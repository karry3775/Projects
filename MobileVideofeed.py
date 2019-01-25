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

url = "http://192.168.0.3:8080/shot.jpg"

def canny_edge(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur =  cv2.GaussianBlur(gray,(kernel,kernel),0)
    canny_img = cv2.Canny(blur,10,250,apertureSize=5,L2gradient=True) #10 and 100 are the lower and upper limit to be considered as an edge
    return canny_img


while True:
    #get the repsonse from url
    img_resp = urllib.urlopen(url)
    #convert it into a numpy image
    imagenp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
    #convert it to a cv2 image
    frame = cv2.imdecode(imagenp,-1)
    # canny_edge detection
    edges = canny_edge(frame)

    # Dilating the image
    dilated = cv2.dilate(edges, np.ones((3,3), dtype=np.uint8))
    # line detection using Normal HOUGH Transform
    # lines = cv2.HoughLines(dilated,1,np.pi/180,250)
    # # from the above line of code we get rho and theta values we need cartesian values
    # if lines is not None:
    #     for rho,theta in lines[0]:
    #         x0 = rho*(np.cos(theta))
    #         y0 = rho*(np.sin(theta))
    #         x1 = int(x0 - 1000*(np.sin(theta)))
    #         y1 = int(y0 + 1000*(np.cos(theta)))
    #         x2 = int(x0 + 1000*(np.sin(theta)))
    #         y2 = int(y0 - 1000*(np.cos(theta)))
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)

    # lets use probabilistic HoughLinesP

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(dilated,1,np.pi/180,100,minLineLength,maxLineGap)
    if lines is not None:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)


    #display image
    cv2.imshow('frame_window',frame)
    if ord('q')==cv2.waitKey(10):
        exit(0)

