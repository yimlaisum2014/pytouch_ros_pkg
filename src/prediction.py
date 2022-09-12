#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int16MultiArray ,Bool
from cv_bridge import CvBridge
from pytouch.tasks import TouchDetect
from pytouch.sensors import DigitSensor
import numpy as np
from PIL import Image as PILImage
import cv2

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

        self.predict_left = 0
        self.predict_right = 0

        self.bridge = CvBridge()
        # Subscriber
        self.sub_left_digit_image = rospy.Subscriber("/finger_left",Image,self.left_digit_image_callback,queue_size=1)
        self.sub_right_digit_image = rospy.Subscriber("/finger_right",Image,self.right_digit_image_callback,queue_size=1)
        # self.sub_test_image = rospy.Subscriber("/test",Image,self.test_image_callback)

        # Publisher
        # self.pub_result = rospy.Publisher("/predicted_touch", Bool, queue_size=1)
        self.pub_result_left = rospy.Publisher("/predicted_left", Bool, queue_size=1)
        self.pub_result_right = rospy.Publisher("/predicted_right", Bool, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1/60), self.predict_callback)

    # def test_image_callback(self,msg):
    #     img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #     cv2.imwrite(f"img_{time.ctime()}.png",img)

    def predict_callback(self, *args): # args is to fix positional argument error
        touch_result = self.get_touch_result()
        print(f'Predict result: {touch_result}')
        self.pub_result.publish(touch_result)

    
    def get_touch_result(self):
        return self.predict_left == 1 and self.predict_right == 1
    
    def left_digit_image_callback(self,msg):
        print(rospy.get_time())
        self.predict_left = self.prediction(msg)
        self.pub_result_left.publish(self.predict_left)

    
    def right_digit_image_callback(self,msg):
        self.predict_right = self.prediction(msg)
        self.pub_result_right.publish(self.predict_right)
    
    def prepare_image(self, image_data, to_gray=False):
        # to cv2 image
        img = self.bridge.imgmsg_to_cv2(image_data, desired_encoding="passthrough")
        # cv2.imwrite(f"img_{time.ctime()}.png",img)
        if to_gray:
        # gray
            print("Enter gray")
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # cv2.imwrite(f"grayimg_{time.ctime()}.png",gray)
            # make it be 3-channels
            img = np.zeros((gray.shape[0], gray.shape[1], 3))
            img[:,:,0] = gray
            img[:,:,1] = gray
            img[:,:,2] = gray
        # cv2.imwrite(f"after_{time.ctime()}.png",img)
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
    model_path = "/home/sis/WFH-locobot/Weight/default-epoch=14_val_loss=0.018_val_acc=0.997.ckpt"
    # model_path = "/home/sis/WFH-locobot/Weight/default-epoch=9_val_loss=0.075_val_acc=0.988.ckpt"
    import os
    assert os.path.exists(model_path), "Failed to find model file"
    prediction = Prediction(model=model_path)
    rospy.spin()

