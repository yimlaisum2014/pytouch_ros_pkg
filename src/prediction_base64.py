#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray , Byte, String
from cv_bridge import CvBridge
from pytouch.tasks import TouchDetect
from pytouch.sensors import DigitSensor
import numpy as np
from PIL import Image as PILImage
import cv2
import base64

import pickle

import time

class DetectectValue:
    SCALES = [240, 320]
    MEANS = [0, 0, 0]
    STDS = [1, 1, 1]
    CLASSES = 2

class Prediction(object):
    def __init__(self,model=""):
        # Defin Model
        self.model = TouchDetect(model_path=model, defaults=DetectectValue, sensor=DigitSensor)
        print('init model' + '~' * 100)
        print(type(self.model))
        self.results = [0,0]
        print("1")
        self.bridge = CvBridge()
        # Subscriber
        print("2")
        self.sub_left_digit_image = rospy.Subscriber("/finger_left_byte",String,self.left_digit_image_callback)
        self.sub_right_digit_image = rospy.Subscriber("/finger_right_byte",String,self.right_digit_image_callback)
        
        # Publisher 
        # self.pub_result = rospy.Publisher('predict_result', Int16MultiArray  , queue_size=10)
        # self.pub_result.publish(self.results)

    def left_digit_image_callback(self,msg):
        print("Enter")
        # left = msg.data
        left = msg
        predict = self.prediction(left)
        self.results[0] = predict
        print(f"left:{self.results[0]}")
        print()
        
    
    def right_digit_image_callback(self,msg):
        # print("~"*100)
        # print(type(msg))
        # right = msg.data
        # print(type(right))
        # print("~"*100)
        right = msg
        # print(type(right), right.data)
        predict = self.prediction(right)
        self.results[1] = predict
        print(f"right:{self.results[1]}")
        print()
    
    def prepare_image(self, image_data, to_gray=False):
        # string
        decoded = base64.b64decode(image_data.data.encode('utf-8'))
        img_arr = np.frombuffer(decoded, dtype=np.uint8)
        # to cv2 image
        img = cv2.imdecode(img_arr, flags=1)

        cv2.imwrite(f"img_{time.ctime()}.png",img)
        if to_gray:
        # gray
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(f"grayimg_{time.ctime()}.png",gray)

            # make it be 3-channels
            img = np.zeros((gray.shape[0], gray.shape[1], 3))
            img[:,:,0] = gray
            img[:,:,1] = gray
            img[:,:,2] = gray
        # to PIL image
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')
        return img
    
    def prediction(self,image_data):
        predict_image = self.prepare_image(image_data, to_gray=True)
        # predict_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        # predict_image = PILImage.fromarray(predict_image.astype('uint8'), 'RGB')
        # predict_image = ImageOps.grayscale(predict_image)
        # print(predict_image.size)
        result,_ = self.model(predict_image)
        return result

if __name__ == "__main__":
    rospy.init_node('prediction', anonymous=True)
    # model_path = "/home/sis/WFH-locobot/Weight/default-epoch=9_val_loss=0.075_val_acc=0.988.ckpt"
    model_path = "/home/sis/WFH-locobot/Weight/default-epoch=9_val_loss=0.075_val_acc=0.988.ckpt"
    import os
    assert os.path.exists(model_path), "Failed to find model file"
    prediction = Prediction(model=model_path)
    rospy.spin()

