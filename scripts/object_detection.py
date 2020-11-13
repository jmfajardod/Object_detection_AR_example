#!/usr/bin/env python

import rospy

from rospy.numpy_msg import numpy_msg
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

import skimage

class Camera_ORB:

    #------------------------------------------------------#
    # Callback function for image
    def cam_image_cb(self,data):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, self.encoding)
            #cv_image = self.bridge.imgmsg_to_cv2(data), self.encoding)
        except CvBridgeError as e:
            print(e)
            return

        # Create a copy of the image to publish
        cv_image2 = cv_image.copy()
        
        # Change the image to Grayscale
        cv_image_gray = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2GRAY)

        # find the keypoints with ORB
        kp_Scene, des_Scene = self.orb.detectAndCompute(cv_image_gray,None)

        # Match features
        if(len(kp_Scene)>0):
            matches = self.matcher.match(self.des_Object, des_Scene) 
            matches_sorted = sorted(matches, key = lambda x:x.distance)
        else:
            return

        # If number of matches is greater than threshold find the location of the object in the scene
        if len(matches_sorted) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([ self.kp_Object[m.queryIdx].pt for m in matches_sorted ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_Scene[m.trainIdx].pt for m in matches_sorted ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Corners of object
            h,w = self.imgObject.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

            # Corners of detected object
            dst = cv2.perspectiveTransform(pts,M).reshape(4, 2)

            # Corners of GOAL image
            h,w = self.imgGoal.shape[:2]
            pts = np.array([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ], dtype = "float32")

            # Get perspective transform
            transf = cv2.getPerspectiveTransform(pts, dst)
            h,w = cv_image2.shape[:2]
            warp = cv2.warpPerspective(self.imgGoal, transf, (w, h))

            cv_image2 = cv2.fillPoly(cv_image2, [np.int32(dst).reshape((-1, 1, 2))], (0,0,0)) # Remove object in scene
            cv_image2 = cv2.add(cv_image2, warp) # Draw new image

            #-- Draw Bounding box
            #cv_image2 = cv2.polylines(cv_image2, [np.int32(dst).reshape((-1, 1, 2))] , True, (0,0,255),3, cv2.LINE_AA)


        try:
            #self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image2))
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image2, "bgr8"))
        except CvBridgeError as e:
            print(e)

    #------------------------------------------------------#
    #------------------------------------------------------#
    #------------------------------------------------------#

    def __init__(self):

        self.cameraTopic = None
        self.encoding = None
        self.url_object = None

        # Load parameters from parameter server
        self.getParameters()

        # Check that the parameters where correctly loaded
        if(self.cameraTopic is None or self.encoding is None or self.url_object is None):
            rospy.signal_shutdown("Parameters not read")
        else:
            rospy.loginfo("Parameters found")

        # Create CV bridge
        self.bridge = CvBridge()

        # Create ORB detector
        self.orb = cv2.ORB_create()

        # Create Matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Read object to detect
        self.imgGoal = cv2.imread(self.url_goal)   
        imgObject = cv2.imread(self.url_object)   
        imgObject = cv2.resize(imgObject, None, fx=0.1, fy=0.1)
        self.imgObject = cv2.cvtColor(imgObject,cv2.COLOR_BGR2GRAY)
        self.kp_Object, self.des_Object = self.orb.detectAndCompute(self.imgObject,None)
        self.MIN_MATCH_COUNT = len(self.kp_Object)/3

        # Create image subscriber
        self.image_sub = rospy.Subscriber(self.cameraTopic, CompressedImage, self.cam_image_cb , queue_size=1)

        # Create image publisher
        self.image_pub = rospy.Publisher("image_orb_features", Image, queue_size=1)
        

    #------------------------------------------------------#
    # Function to get parameters
    def getParameters(self):

        if rospy.has_param('~cam_topic'):   self.cameraTopic = rospy.get_param('~cam_topic')
        if rospy.has_param("~encoding"):    self.encoding = rospy.get_param("~encoding")
        if rospy.has_param("~img_object"):  self.url_object = rospy.get_param("~img_object")
        if rospy.has_param("~img_goal"):    self.url_goal = rospy.get_param("~img_goal")

#-----------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------#

if __name__ == '__main__':

    # Firt init the node and then the object to correctly find the parameters
    rospy.init_node('image_features', anonymous=True)
    Camera_ORB()
    
        
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()