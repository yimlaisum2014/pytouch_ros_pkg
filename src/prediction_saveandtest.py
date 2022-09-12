#!/usr/bin/env python3

import os
import sys

from torchmetrics import Accuracy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), '../../PyTouch')))

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray ,Bool, String
from cv_bridge import CvBridge
from pytouch.tasks.touch_detect import TouchDetect
from pytouch.sensors import DigitSensor
import numpy as np
from PIL import Image as PILImage
import cv2

import time

class DetectectValue:
    SCALES = [240, 320]
    # SCALES = [128, 128]
    # SCALES = [64, 64]
    MEANS = [0, 0, 0]
    STDS = [1, 1, 1]
    CLASSES = 2

class Prediction(object):

    def __init__(self,model="",save=False,check_static_data=False,check_time_4_pred=False,
                check_realtime_data=False,check_realtime_samples = 500):
        # Defin Model
        self.model = TouchDetect(model_path=model, defaults=DetectectValue, sensor=DigitSensor)
        self.save = save
        self.check_static_data = check_static_data 
        self.check_time_4_pred = check_time_4_pred
        self.check_real_time_data = check_realtime_data
        self.check_real_time_samples = check_realtime_samples
        # u
        self.count = 0 
        self.time_comsume = []
        self.result_buffer = []

        self.savepath = "/home/sis/Dataset/come_f_ROS"
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.bridge = CvBridge()
        # Subscriber
        self.sub_left_digit_image = rospy.Subscriber("/finger_left",Image,self.left_digit_image_callback,queue_size=1)
        # self.sub_right_digit_image = rospy.Subscriber("/finger_right",Image,self.right_digit_image_callback,queue_size=1)

        # Publisher
        self.pub_result_left = rospy.Publisher("/predicted_left", Float32MultiArray, queue_size=1)
        # self.pub_result_right = rospy.Publisher("/predicted_right", Float32MultiArray, queue_size=1)

        if self.check_static_data == True:
            self.check_static()

    def check_static(self):
        file = self.savepath 
        count = {1: 0, 0: 0}

        for root, dirs, names in os.walk(file):
            for name in names:
                filename = os.path.join(root, name)
                img = cv2.imread(filename)
                frame = PILImage.fromarray(img)
                is_touching, certainty = self.model(frame)
                value = is_touching
                count[value] += 1

        print(count)
        
    def save_stream(self,ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        path = self.savepath
        name = f'img_{self.frame_id}.jpg'
        print(os.path.join(path, name))
        cv2.imwrite(os.path.join(path, name), cv_image)

    def cal_time_comsume():
        pass

    def left_digit_image_callback(self,msg):
        if self.save == True :
            self.save_stream(msg)   
        
        before = rospy.get_rostime() # get before prediction time
        self.count += 1

        result,accuracy = self.prediction(msg)
        predict_left = Float32MultiArray()
        frame_id = int(msg.header.frame_id)

        predict_time = rospy.get_rostime()-before # get time consuming for prediction
        # print(type(frame_id),type(result),type(accuracy),type(self.count))
        predict_left.data = [frame_id , result, self.count]
        print(predict_left.data)
        self.pub_result_left.publish(predict_left)

        # get avg of time consuming for prediction
        if self.check_time_4_pred == True :
            self.time_comsume.append(predict_time.to_sec()) # from ros.duration to float
            Sum = sum(self.time_comsume)
            Avg = Sum/self.count
            print(self.count,predict_time.to_sec(),Avg)

        # test real-time
        if self.check_real_time_data == True :
            self.result_buffer.append(result)

            if (len(self.result_buffer) == self.check_real_time_samples):
                count_result = self.result_buffer.count(0) # 0:touch
                print(count_result)
                # self.result_buffer.clear()
            


    
    def prepare_image(self, image_data, to_gray=False):
        # to cv2 image
        img = self.bridge.imgmsg_to_cv2(image_data, desired_encoding='passthrough')
        # cv2.imwrite(f"img_{time.ctime()}.png",img)
        if to_gray:
        # gray
            # print("Enter gray")
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # cv2.imwrite(f"grayimg_{time.ctime()}.png",gray)
            # make it be 3-channels
            img = np.zeros((gray.shape[0], gray.shape[1], 3))
            img[:,:,0] = gray
            img[:,:,1] = gray
            img[:,:,2] = gray
        # cv2.imwrite(f"after_{self.frame_id}.png",img)

        # to PIL image
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')
        return img
    
    def prediction(self,image_data):
        
        predict_image = self.prepare_image(image_data, to_gray=True)
        result,accuracy = self.model(predict_image)
        return result,accuracy

if __name__ == "__main__":
    rospy.init_node('prediction', anonymous=True)
    # model_path = "/home/sis/WFH-locobot/Weight/default-epoch=14_val_loss=0.018_val_acc=0.997.ckpt"
    # model_path = "/home/sis/WFH-locobot/Weight/default-epoch=9_val_loss=0.075_val_acc=0.988.ckpt"
    # model_path = "/home/sis/WFH-locobot/Weight/default-epoch=48_val_loss=0.000_val_acc=1.000.ckpt"
    # model_path = "/home/sis/WFH-locobot/Weight/default-epoch=45_val_loss=0.002_val_acc=1.000.ckpt"
    # default-epoch=33_val_loss=0.211_val_acc=0.913.ckpt gray resnet18
    # "default-epoch=46_val_loss=0.014_val_acc=0.998.ckpt" rgb

    model_folder = "/home/sis/WFH-locobot/Weight/"
    # model_file = "default-epoch=33_val_loss=0.211_val_acc=0.913.ckpt"

    # 20220711 test

    # model_file = "20220711-LeNet5/default-epoch=11_val_loss=0.005_val_acc=0.999.ckpt" # LeNet5-64-without bottlecap gray-after SDG 
    # model_file = "20220711-LeNet5/default-epoch=74_val_loss=0.001_val_acc=1.000.ckpt" # LeNet5-128-without bottlecap gray-after SDG 
    # model_file = "20220711-LeNet5/default-epoch=75_val_loss=0.010_val_acc=0.996.ckpt" # LeNet5-ori-without bottlecap gray-after SDG 

    # model_file = "20220711-VGG16/default-epoch=12_val_loss=0.132_val_acc=0.945.ckpt" # VGG16-64-without bottlecap gray-after SDG 
    # model_file = "20220711-VGG16/default-epoch=25_val_loss=0.147_val_acc=0.963.ckpt" # VGG16-128-without bottlecap gray-after SDG
    ### model_file = "20220711-VGG16/default-epoch=28_val_loss=0.156_val_acc=0.947.ckpt" # VGG16-128-anotherepoch-without bottlecap gray-after SDG
    model_file = "20220711-VGG16/default-epoch=78_val_loss=0.128_val_acc=0.963.ckpt" # VGG16-ori-without bottlecap gray-after SDG ** Best

    # model_file = "20220711-ResNet18/default-epoch=12_val_loss=0.080_val_acc=0.974.ckpt" # ResNet18-64-without bottlecap gray-after SDG
    # model_file = "20220711-ResNet18/default-epoch=45_val_loss=0.094_val_acc=0.969.ckpt" # ResNet18-128-without bottlecap gray-after SDG
    # model_file = "20220711-ResNet18/default-epoch=46_val_loss=0.138_val_acc=0.954.ckpt" # ResNet18-ori-without bottlecap gray-after SDG

    
    
    model_path = f"{model_folder}{model_file}"
    assert os.path.exists(model_path), "Failed to find model file"

    prediction = Prediction(model=model_path,save=False,check_static_data=False,check_time_4_pred=False,check_realtime_data=False)

    rospy.spin()

