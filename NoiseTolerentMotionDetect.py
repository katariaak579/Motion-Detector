import cv2
import numpy as np
from cv2 import COLOR_BGR2GRAY
from cv2 import THRESH_BINARY
from collections import deque

class BgExtract:
    def __init__(self,width,height,scale,maxlen=10): 
        self.maxlen=maxlen
        self.width=width//scale
        self.height=height//scale
        self.buffer=deque(maxlen=maxlen)
        self.bg=None
    
    def cal_if_notfull(self):
        self.bg=np.zeros((self.height,self.width,),dtype='float32')
        for i in self.buffer:
            self.bg+=i
        self.bg//=len(self.buffer)

    def cal_if_full(self,old,new):
        self.bg-=old/self.maxlen
        self.bg+=new/self.maxlen


    def add_frame(self,frame):
        if self.maxlen>len(self.buffer):
            self.buffer.append(frame)
            self.cal_if_notfull()
        else:
            old=self.buffer.popleft()
            self.buffer.append(frame)
            self.cal_if_full(old,frame)


    def output_frame(self):       
        return self.bg.astype('uint8')

width=640
height=480
scale_down=2

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

bg_buffer=BgExtract(width,height,scale_down)
cv2.imshow("background",bg)

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    down_scale=cv2.resize(frame,(width//scale_down,height//scale_down))
    # Creating the gray frame to act as background 
    gray=cv2.cvtColor(down_scale,COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    
    bg_buffer.add_frame(gray)
    absdifference=cv2.absdiff(gray,bg_buffer.output_frame())
    _,maskabs=cv2.threshold(absdifference,15,255,THRESH_BINARY)

    contours,_=cv2.findContours(maskabs,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        if cv2.contourArea(i) < 150:
            continue

        x,y,w,h=cv2.boundingRect(i)
        x,y,w,h=x*scale_down,y*scale_down,w*scale_down,h*scale_down
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))

    cv2.imshow("abs",maskabs)
    cv2.imshow("Actual",frame)
    # cv2.imshow("dilated",dilated_mask)
    if cv2.waitKey(1) == ord('q'):
        break

