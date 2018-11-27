from styx_msgs.msg import TrafficLight
# import necessary packages for prediction
import rospy
import os
import yaml

import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

IMG_H = 600   # image height in pixels
IMG_W = 800  # image width in pixels
IMG_C = 3     # num of channels

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model_dir_path = None
        self.model = None
        self.graph = None
        
        # load configuration string
        conf_str = rospy.get_param ("/traffic_light_config")
        #rospy.loginfo ("config string {}".format (conf_str))
        self.configuration = yaml.load(conf_str)
        #rospy.loginfo ("configuration {}".format (self.configuration))		
        #select & load appropriate model based environment configuration
        if (self.configuration ['is_site']):
            self.model_dir_path = './models/site_model.h5'
        else:
            self.model_dir_path = './models/sim_model.h5'
   
        rospy.loginfo ("model directory path: {} ".format(self.model_dir_path))
		
        #load the model
        if  not (os.path.exists(self.model_dir_path)):
            rospy.logerr ("model directory path {} does not exist".format (self.model_dir_path))
        else:
            self.model = load_model(self.model_dir_path)
            # why is this needed?
            self.model._make_predict_function()
            self.graph = K.tf.get_default_graph()
            rospy.loginfo ("model loaded successfully from {}".format (self.model_dir_path))
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        try:
            with self.graph.as_default ():
                # check if graph is None
                if self.graph == None:
                    rospy.logerr ("Graph is None")
                    return TrafficLight.UNKNOWN
            
                # check if model is Empty
                if self.model == None:
                    rospy.logerr ("Model is None")
                    return TrafficLight.UNKNOWN
                
                #resize the image as per model acceptance
                img = np.reshape (image,  (1, IMG_H, IMG_W, IMG_C))
                score_list = self.model.predict (img)
            
                #check score_list is empty
                if (type (score_list) == None or len(score_list) == 0):
                    rospy.loginfo ("Prediction score list empty")
                    return TrafficLight.UNKNOWN
                    
                #non empty score_list
                light_type = np.argmax (score_list)
                if (light_type == 0):
                    return TrafficLight.RED
                elif (light_type == 1):
                    return TrafficLight.GREEN
                else:
                    return TrafficLight.UNKNOWN
                    
        except Exception as e:
            rospy.logerr ("Traffic Classifier raised exception")
            rospy.logerr (e)
            return TrafficLight.UNKNOWN
