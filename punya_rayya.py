from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import io
import threading
import cv2
import time
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import os

GPIO.setwarnings(False)
GPIO.setmode (GPIO.BCM)
GPIO.setup(24,GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.OUT)

camera = PiCamera()
res = 1280
camera.resolution = (2560,1936)
camera.framerate = 64
camera.start_preview()
sleep(5)
rawCapture = PiRGBArray(camera, size = (res,res))
camera.stop_preview()
state = 0

lock=0
interval =2
i=0
waktuS=0
take=0

# while (1):
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors (image, number_of_colors):
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)


def empty(a):
    pass

# create new window with trackbar HSV Color

cv2.namedWindow("Range HSV")
cv2.resizeWindow("Range HSV", 500, 350)
cv2.createTrackbar("HUE Min","Range HSV",0,180,empty)
cv2.createTrackbar("HUE Max","Range HSV",180,180,empty)
cv2.createTrackbar("SAT Min", "Range HSV", 0,255,empty)
cv2.createTrackbar("SAT Max", "Range HSV", 255,255,empty)
cv2.createTrackbar("VALUE Min", "Range HSV", 0,255,empty)
cv2.createTrackbar("VALUE Max", "Range HSV", 255,255,empty)

# read image
image = cv2.imread(image_path)

while True:

        # get value from trackbar
        h_min = cv2.getTrackbarPos("HUE Min", "Range HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "Range HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "Range HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "Range HSV")
        v_min = cv2.getTrackbarPos("VALUE Min", "Range HSV")
        v_max = cv2.getTrackbarPos("VALUE Max", "Range HSV")

        # define range of some color in HSV

        lower_range = np.array([h_min,s_min,v_min])
        upper_range = np.array([h_max, s_max, v_max])

        # convert image to HSV

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # threshold the hsv image to get some color

        thresh = cv2.inRange(hsv, lower_range, upper_range)

        # bitwise AND mask and original image

        bitwise = cv2.bitwise_and(image, image, mask=thresh)

        cv2.imshow("Original Image", image)
        cv2.imshow("Thresholded", thresh)
        cv2.imshow("Bitwise", bitwise)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
                mode = not mode
        elif k == 27:
                break

# for threading
class ImageProcessor(threading.Thread):
    """Processor Thread processing the images from Pi Camera (like      inference in Deep Learning) """
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    #Image.open(self.stream)
                    #...
                    #...
                    # Set done to True if you want the script to terminate
                    # at some point
                    #self.owner.done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)

class ProcessOutput(object):
    """Image Input Thread maintaining streaming of the images from Pi Camera and handling ImageProcessor"""
    def __init__(self):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        # no of ImageProcessor is a parameter here, depends on used hardware.
        # This help us to scale the capturing and benchmark the processing limit
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None

    def write(self, buf):
        """Writing Image into Processor"""
        # magic word when capturing in mjpeg format.
        # magic word: 'b\xff\xd8' helps us in getting new frames
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            proc.terminated = True
            proc.join()

# resolution can be 'VGA' or 'BGR or 'RGB'
with picamera.PiCamera(resolution='RGB') as camera:
    # start the preview of camera streaming
    camera.start_preview()

    # give sensors some time to start
    time.sleep(2)

    # make a output image processor object
    output = ProcessOutput()

    # start recording the camera
    # learn about format here: https://picamera.readthedocs.io/en/release-1.10/api_camera.html#picamera.camera.PiCamera.start_recording
    camera.start_recording(output, format='mjpeg')
    while not output.done:
        camera.wait_recording(1)
    camera.stop_recording()

cv2.destroyAllWindows()