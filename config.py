# base path to YOLO directory
MODEL_PATH = "yolo"
MODEL = "yolov4"
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.6
NMS_THRESH = 0.6

#===============================================================================
#=================================\CONFIG./=====================================
""" Below are your desired config. options to set for real-time inference """
# To count the total number of people (True/False).
People_Counter = False
# Threading ON/OFF. Please refer 'mylib>thread.py'.
Thread = True

# Set the threshold value for total violations limit.
Threshold = 30
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video');
# Set url = 0 for webcam.
'''
url1 = "http://151.192.128.131:18888/video"
url2 = "http://151.192.128.131:18888/video"
'''

#url = ""
url = 1
#url = "IMG_9117.mov"


#url = 1
# Turn ON/OFF the email alert feature.
ALERT = False
# Set mail to receive the real-time alerts. E.g., 'xxx@gmail.com'.
MAIL = 'spaev2020@gmail.com'
# Set if GPU should be used for computations; Otherwise uses the CPU by default.
USE_GPU = False
# Define the max/min safe distance limits (in pixels) between 2 people.
MAX_DISTANCE = 200
MIN_DISTANCE = 100
#===============================================================================
#===============================================================================
