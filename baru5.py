from random import sample
from picamera.array import PiRGBArray
import picamera
import cv2
import time
import numpy as np
import RPi.GPIO as GPIO
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.animation as animation
import threading as thread
import os

#Parameters
x_len = 100        # Number of points to display
y_range = [10, 80]  # Range of possible Y values to display

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = list(range(0, 100))
ys = [0] * x_len
yu = [0] * x_len
ax.set_ylim(y_range)

#variabel
ct_value = []
set_array = [20, 40, 80]
interval_1 = 1000

ulang = 0
iter = 0
set_siklus = 20

GPIO.setwarnings(False)
GPIO.setmode (GPIO.BCM)
GPIO.setup(24,GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.OUT)

#bagian ini buat bloknya ga panas
#_________pin_GPIO_______
heat_pin = 19 #pwm > GPIO 19 > LPWM
cool_pin = 12 #pwm > GPIO 13 > RPWM
pwm_fan = 13 #pwm > GPIO 12 > fan cooling
low_volt = 21 #digital > GPIO 21 > 3v3
glass_pin = 16

GPIO.setwarnings(False) #disable warning
GPIO.setmode(GPIO.BCM)  #board GPIO
GPIO.setup(heat_pin,GPIO.OUT) #heat pin
GPIO.output(heat_pin,GPIO.LOW) #heat pin
GPIO.setup(cool_pin,GPIO.OUT) #cool pin
GPIO.output(cool_pin,GPIO.LOW) #cool pin 
GPIO.setup(pwm_fan,GPIO.OUT) #kontrol fan menggunakan pwm sinyal 
GPIO.output(pwm_fan,GPIO.HIGH) #kontrol fan menggunakan pwm sinyal 
GPIO.setup(low_volt,GPIO.OUT) #supply tegangan pin digital 3v3
GPIO.output(low_volt,GPIO.HIGH)
GPIO.setup(glass_pin, GPIO.OUT)
#GPIO.output(glass_pin, GPIO.HIGH)
GPIO.output(glass_pin, GPIO.LOW)
#____PWM_config____
heat = GPIO.PWM(heat_pin,490) #490Hz
cool = GPIO.PWM(cool_pin,490) #490Hz
fan = GPIO.PWM(pwm_fan,490) #490Hz
heat.start(0)
cool.start(0)
fan.start(100)
#sampai sini

camera = picamera.PiCamera()
res = 1280
camera.resolution = (res, res)
camera.framerate = 64
rawCapture = PiRGBArray(camera, size = (res,res))
state = 0

lock=0
interval =2
i=0
waktuS=0
take=0

#################
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
len(flags)
258
flags[40]
'COLOR_BGR2RGB'

sample = cv2.imread('./hasil_sample/ini.jpg')
plt.imshow(sample)
plt.show()

sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
plt.imshow(sample)
plt.show()

r, g, b = cv2.split(sample)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = sample.reshape((np.shape(sample)[0]*np.shape(sample)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

hsv_sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_sample)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
#############################


while (1):
    def get_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def get_colors (image, number_of_colors):
        modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
        
    def get_colors (image, number_of_colors):
        modified_image = cv2.resize(image, (600, 400), interpolation =cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image. shape[1], 3)
    
        clf = KMeans(n_clusters = number_of_colors)
        labels = clf.fit_predict(modified_image)

        counts = Counter(labels)

        center_colors = clf.cluster_centers_
        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]
        arr_rgb = np.asarray(rgb_colors)
        return rgb_colors

    
    if state ==1 :
        GPIO.output(23, True)
        print("hai")
        time.sleep(0.01)
        for frame in camera.capture_continuous(rawCapture, format = "rgb", use_video_port= True):
            time = time.time()
            image = frame.array
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Frame", rgb)

            if (waktu-waktuS >= interval) and take==0 :
                camera.capture('/home/pi/Desktop/punyarayy/hasilsample/image_tes%s.jpg' %i)
                i=i+1
                waktuS=waktu

                if i>2 :
                    take=1
                    i=0
                    print("done")
                    key = cv2.waitKey(1)&0xFF
                rawCapture.truncate(0)

            if take==1 :
                take=0
            break
        
        flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
        len(flags)
        258
        flags[40]
        'COLOR_BGR2RGB'

        sample = cv2.imread('./hasil_sample/ini.jpg')
        plt.imshow(sample)
        plt.show()

        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        plt.imshow(sample)
        plt.show()

        r, g, b = cv2.split(sample)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        pixel_colors = sample.reshape((np.shape(sample)[0]*np.shape(sample)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        plt.show()

        hsv_sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_sample)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()
################

mypath = '/home/pi/Desktop/punyarayy/hasilsample/'
onlyfile = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
lenght=len(onlyfile)
print(onlyfile)

for s in range (0, len(onlyfile)):
    im_cropp = cv2.imread('/home/pi/Desktop/punyarayy/hasilsample/image_tes%s.jpg' %s)
img = im_cropp
print(s)
for z in range(9):
    if z==0:
        x1=385
        x2=500
        y1=380
        y2=490
    if z==1:
        x1=640
        x2=755
        y1=380
        y2=490
    if z==2:
        x1=880
        x2=995
        y1=380
        y2=490
    if z==3:
        x1=385
        x2=500
        y1=590
        y2=700
    if z==4:
        x1=640
        x2=755
        y1=590
        y2=700
    if z==5:
        x1=880
        x2=995
        y1=590
        y2=700
    if z==6:
        x1=385
        x2=500
        y1=790
        y2=910
    if z==7:
        x1=640
        x2=755
        y1=790
        y2=910
    if z==8:
        x1=880
        x2=995
        y1=790
        y2=910

cropp = img[y1:y2, x1:x2]
if z==0:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s1/imagesample9%s.jpg' %s,cropp)
if z==1:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s2/imagesample9%s.jpg' %s,cropp)
if z==2:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s3/imagesample9%s.jpg' %s,cropp)
if z==3:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s4/imagesample9%s.jpg' %s,cropp)
if z==4:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s5/imagesample9%s.jpg' %s,cropp)
if z==5:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s6/imagesample9%s.jpg' %s,cropp)
if z==6:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s7/imagesample9%s.jpg' %s,cropp)
if z==7:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s8/imagesample9%s.jpg' %s,cropp)
if z==8:
    cv2.imwrite('/home/pi/Desktop/punyarayy/hasilsample/s9/imagesample9%s.jpg' %s,cropp)

print("done")
state=0
lock=0
print(state)

figure, axis = plt.subplots(3,3)
x=[1+l for l in range (lenght)]
for q in range (1,10):
    if q == 1:
        arr =[(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s1/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)

re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[0,0].plot(x, re[:,0], color='r', label='red')
axis[0,0].plot(x, re[:,1], color='g', label='green')
axis[0,0].plot(x, re[:,2], color='b', label='blue')
axis[0,0].set_title("Sample 1")

if q == 2:
    arr =[(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s2/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[0,1].plot(x, re[:,0], color='r', label='red')
axis[0,1].plot(x, re[:,1], color='g', label='green')
axis[0,1].plot(x, re[:,2], color='b', label='blue')
axis[0,1].set_title("Sample 2")

if q == 3:
    arr = [(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s3/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[0,2].plot(x, re[:,0], color='r', label='red')
axis[0,2].plot(x, re[:,1], color='g', label='green')
axis[0,2].plot(x, re[:,2], color='b', label='blue')
axis[0,2].set_title("Sample 3")

if q == 4:
    arr =[(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s4/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[1,0].plot(x, re[:,0], color='r', label='red')
axis[1,0].plot(x, re[:,1], color='g', label='green')
axis[1,0].plot(x, re[:,2], color='b', label='blue')

axis[1,0].set_title("Sample 4")

if q == 5:
    arr =[(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s5/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[1,1].plot(x, re[:,0], color='r', label='red')
axis[1,1].plot(x, re[:,1], color='g', label='green')
axis[1,1].plot(x, re[:,2], color='b', label='blue')
axis[1,1].set_title("Sample 5")

if q == 6:
    arr =[(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s6/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[1,2].plot(x, re[:,0], color='r', label='red')
axis[1,2].plot(x, re[:,1], color='g', label='green')
axis[1,2].plot(x, re[:,2], color='b', label='blue')
axis[1,2].set_title("Sample 6")

if q == 7:
    arr = [(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s7/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[2,0].plot(x, re[:,0], color='r', label='red')
axis[2,0].plot(x, re[:,1], color='g', label='green')
axis[2,0].plot(x, re[:,2], color='b', label='blue')
axis[2,0].set_title("Sample 7")

if q == 8:
    arr = [(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s8/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[2,1].plot(x, re[:,0], color='r', label='red')
axis[2,1].plot(x, re[:,1], color='g', label='green')
axis[2,1].plot(x, re[:,2], color='b', label='blue')
axis[2,1].set_title("Sample 8")

if q == 9:
    arr = [(get_colors(get_image('/home/pi/Desktop/punyarayy/hasilsample/s1/imagesample%s.jpg' %n),1)) for n in range(len(onlyfile))]
print(arr)
re= np.reshape(arr,(lenght,3))
print(re)
print(re[:,1])

axis[2,2].plot(x, re[:,0], color='r', label='red')
axis[2,2].plot(x, re[:,1], color='g', label='green')
axis[2,2].plot(x, re[:,2], color='b', label='blue')
axis[2,2].set_title("Sample 9")
plt.show()

cv2.destroyAllWindows()