#!/usr/bin/python
#-------GUI for use with RasPi (operating mode)-----------------------

#tkinter imports
import tkinter as tk
from tkinter import filedialog, messagebox, StringVar
from tkinter import * #this imports Label

#Python Imaging Library
from PIL import Image, ImageTk, ImageDraw

#Imports for saving data and Bluetooth readout
import pandas as pd
import csv
import sys
import os
import serial

#Numpy
import numpy as np
from numpy import sys
np.set_printoptions(threshold=sys.maxsize) #otherwise the print output is suppressed

#Scipy
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import stats

#Matplotlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

#Dimensionality Reduction and Machine Learning
from sklearn.decomposition import PCA
from sklearn import svm
import pandas as pd
import sys
import os

#CSV (for writing data to file)
import csv

#imports that are specific to hardware (not included in other version of script)
from picamera import PiCamera 
from time import sleep
import RPi.GPIO as GPIO
##print('Assets Imported')

#Turn off GPIO Warnings
GPIO.setwarnings(False)

#camera settings
camera = PiCamera()
camera.resolution = (2592,1944)
camera.framerate = 15
camera.brightness = 50
camera.rotation = 180

camera.ISO = 800
camera.shutter_speed = 6000000
camera.exposure_mode = 'verylong'

#Pi pins for LEDs (not included in other version of script)
pin365 = 22 #wavelength 365, previously called ledPin3
pin375 = 23 #wavelength 375, previously called ledPin1
pin385 = 27 #wavelength 385, previously called ledPin2
##pinWhite = 4 #white LED, previously for calibration

#Fonts used in GUI
SMALLEST_FONT = ("Verdana", 11)
SMALL_FONT= ("Verdana", 13)
LARGE_FONT= ("Verdana", 20)
LARGE_BOLD_FONT= ("Verdana", 18, 'bold')
LARGER_FONT= ("Verdana", 22)
LARGEST_FONT= ("Verdana", 24)

#File paths: change depending on where I am testing code

file_ex1 = '/home/pi/Desktop/Data/calibration_data/step1.jpg' #step 1 example image
file_ex2 = '/home/pi/Desktop/Data/calibration_data/step2.jpg' #step 2 example image
file_ex3 = '/home/pi/Desktop/Data/calibration_data/step3.jpg' #step 3 example image
file_coords = '/home/pi/Desktop/Data/calibration_data/crop_coordinates.txt' #file to read/write crop coordinates
intensity_data = '/home/pi/Desktop/Data/intensity_data' #save intensity data (for now)
file_dil = '/home/pi/Desktop/Data/calibration_data/dilutions.png'

#Value for resizing images so they fit on the GUI - might be altered
resize_width = 210

#Set intensity scaling factor for viewing overlayed greyscale images
intensity_scaling_factor = 10 #255 makes pixels black or white, 1 keeps the intensities as they are

#To save data to device
import os

class CrudeOilGUI(tk.Tk): #parentheses are for inheritances here, not parameters
    def __init__(self, *args, **kwargs):
        #print("Inititalizing")
        #this is a method, not a function; everything that we want to happen immediately upon running code
        #*args lets you pass thru as much crap as you want to (variables)
        #**kwargs for passing thru dictionaries
        tk.Tk.__init__(self, *args, **kwargs) #initialize tkinter too
        container = tk.Frame(self) #frame is basically a window
        container.pack(side="top", fill="both", expand=True) #expand will take up any white space remaining (beyond limits)
        container.grid_rowconfigure(0, weight=1) #makes 0 minimum size, weighting sets priorities if needed
        container.grid_columnconfigure(0, weight=1)

    #For now, keep structures that will allow for multiple pages in case we need that
        self.frames= {} #empty dictionary
        frame = HomePage(container, self)
        self.frames[HomePage] = frame
        frame.grid(row=0, column=0, sticky="nsew") #unfilled rows and columns will collapse, sticky stretches to given corners
        self.show_frame(HomePage) #show start page when you initialize
        
    def show_frame(self, cont):
        frame = self.frames[cont] #cont is the key that gets thrown into frame to bring that page up
        frame.tkraise() #raise the frame that you want at the time to the top/front
        frame.update()
        frame.event_generate("<<ShowFrame>>")
        
    def get_page(self, page_class):
        return self.frames[page_class]

#This is the home page.
class HomePage(tk.Frame): #inherit from tk frame so you don't have to call it
    def __init__(self, parent, controller):
        self.controller=controller
        tk.Frame.__init__(self, parent, bg='orchid4') #parent will be CrudeOilGUI
        label = tk.Label(self, text="Oil Type Analyzer", font=LARGER_FONT, bg='orchid4', fg='white') #haven't done anything with it yet
        label.grid(row=0, column=0, columnspan=10, sticky='w', padx=5, pady=10)

        #Create and place START button
        self.startButton = tk.Button(self, text="START", font=LARGEST_FONT)
        self.startButton['command']=lambda: self.startButtonClicked() 
        self.startButton.grid(row=1, column=0, sticky='w', padx=5, pady=10)
        
    def startButtonClicked(self):
        #Hide START button for now
        self.startButton.grid_forget()
        #Create and place label indicating that calibration is first required
        self.calibrationLabel = tk.Label(self, text="Please calibrate device before running samples", font=SMALL_FONT, fg='white', bg='orchid4')
        self.calibrationLabel.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=3)
        #Create and place button to show instructions
##        self.showInstructionsButton= tk.Button(self, text='Show Instructions', font=SMALL_FONT)
##        self.showInstructionsButton['command']= lambda: self.showInstructions()
##        self.showInstructionsButton.grid(row=2, column=0, sticky='w', padx=5, pady=10)
        #Create and place button to start calibration
        self.calibrationButton= tk.Button(self, text='Start Crop Calibration Now', font=LARGE_FONT,
                                     command= lambda: self.runCalibration())
        self.calibrationButton.grid(row=15, column=0, columnspan=2, sticky='w', padx=5, pady=10)

##    def showInstructions(self):
##        self.showInstructionsButton.config(state=DISABLED) #needs to be disabled and re-enabled otherwise clearing instructions doesn't work
##        #Create and place button to hide instructions
##        self.hideInstructionsButton = tk.Button(self, text='Hide Instructions', font=SMALL_FONT)
##        self.hideInstructionsButton['command'] = lambda: self.hideInstructions()
##        self.hideInstructionsButton.grid(row=5, column=0, sticky='w', padx=5, pady=10)
##        #Create and place text instruction boxes
##        self.instructions1 = tk.Message(self, text='Step 1: Insert DCM sample and press OK button', font=SMALLEST_FONT,
##                                   bg='white')
##        self.instructions1.grid(row=4, column=0, sticky='w', padx=5)
##
##        self.instructions2 = tk.Message(self, text="Step 2: After data is generated, click a point to generate left boundary line.",
##                                        font=SMALLEST_FONT,bg="white")
##        self.instructions2.grid(row=4, column=1, sticky='w', padx=5, pady=2)
##        self.instructions3 = tk.Message(self, text="Step 3: Click another point to generate right boundary line.", font=SMALLEST_FONT, bg="white")
##        self.instructions3.grid(row=4, column=2, sticky='w', padx=5)
##
##        #CREATE CANVASES FOR EXAMPLE IMAGES
##        self.ex1canvas = tk.Canvas(self, width=150, height=120, borderwidth=0, highlightthickness=0)
##        self.ex1canvas.grid(row=3, column=0, sticky='w', padx=3)
##        self.ex2canvas = tk.Canvas(self, width=150, height=120, borderwidth=0, highlightthickness=0)
##        self.ex2canvas.grid(row=3, column=1, sticky='w', padx=3)
##        self.ex3canvas = tk.Canvas(self, width=150, height=120, borderwidth=0, highlightthickness=0)
##        self.ex3canvas.grid(row=3, column=2, sticky='w', padx=3)
##
##        #OPEN AND RESIZE EXAMPLE IMAGES
##        ex1 = Image.open(file_ex1)
##        ex1_resized = ex1.resize((150, 120), Image.ANTIALIAS)
##        ex2 = Image.open(file_ex2)
##        ex2_resized = ex2.resize((150, 120), Image.ANTIALIAS)
##        ex3 = Image.open(file_ex3)
##        ex3_resized = ex3.resize((150, 120), Image.ANTIALIAS)
##        #DISPLAY EXAMPLE IMAGES
##        tk_ex1 = ImageTk.PhotoImage(ex1_resized)
##        tk_ex2 = ImageTk.PhotoImage(ex2_resized)
##        tk_ex3 = ImageTk.PhotoImage(ex3_resized)
##        
##        label1 = Label(image=tk_ex1)
##        label2 = Label(image=tk_ex2)
##        label3 = Label(image=tk_ex3)
##        label1.image = tk_ex1
##        label2.image = tk_ex2
##        label3.image = tk_ex3
##
##        self.ex1canvas.create_image(0,0, image=tk_ex1, anchor='nw')
##        self.ex2canvas.create_image(0,0, image=tk_ex2, anchor='nw')
##        self.ex3canvas.create_image(0,0, image=tk_ex3, anchor='nw')
##
##    def hideInstructions(self):
##        self.ex1canvas.destroy()
##        self.ex2canvas.destroy()
##        self.ex3canvas.destroy()
##        self.instructions1.destroy()
##        self.instructions2.destroy()
##        self.instructions3.destroy()
##        self.hideInstructionsButton.grid_forget()
##        self.showInstructionsButton.config(state=NORMAL)
##        self.calibrationButton.config(state=NORMAL)
        
    def runCalibration(self):
        self.calibrationButton.grid_forget()
        self.insertMessage = tk.Message(self, text='Insert DCM sample and press OK.', font=SMALL_FONT)
        self.insertMessage.grid(row=16, column=0, sticky='w', padx=5, pady=5)
        self.okButton1 = tk.Button(self, text='OK', font=SMALL_FONT)
        self.okButton1['command'] = lambda: self.clearCropInsertAlert()
        self.okButton1.grid(row=17, column=0, sticky='w', padx=5)
        
    def clearCropInsertAlert(self):
        self.insertMessage.destroy()
        self.okButton1.destroy()
        self.startCalibration()
        
    def startCalibration(self):
        #Create empty list for storing data
        self.data_to_append=[None]*20
        
        #Filepaths for cropped images
        crop_file1 = '/home/pi/Desktop/Data/calibration_data/cropcal_365.jpg' 
        crop_file2 = '/home/pi/Desktop/Data/calibration_data/cropcal_375.jpg' 
        crop_file3 = '/home/pi/Desktop/Data/calibration_data/cropcal_385.jpg' 

        #GPIO setup for board and UV LEDs
        GPIO.setmode(GPIO.BCM) # Broadcom pin-numbering scheme
        GPIO.setup(pin365, GPIO.OUT) # LED pins set as outputs
        GPIO.setup(pin375, GPIO.OUT)
        GPIO.setup(pin385, GPIO.OUT)

        #take images with each of the LEDs on in succession
        GPIO.output(pin365, GPIO.HIGH)
        camera.capture(crop_file1)
        GPIO.output(pin365, GPIO.LOW)

        GPIO.output(pin375, GPIO.HIGH)
        camera.capture(crop_file2)
        GPIO.output(pin375, GPIO.LOW)

        GPIO.output(pin385, GPIO.HIGH)
        camera.capture(crop_file3)
        GPIO.output(pin385, GPIO.LOW)
        
        cropimgs=[crop_file1, crop_file2, crop_file3]
        lst=list(cropimgs)
        
        j=1
        try:
            for cropcal in lst:
                #OPEN IMAGE
                original = Image.open(cropcal)
                #CONVERT ORIGINAL IMAGE TO GRAYSCALE NUMPY ARRAY
                grey=original.convert('L')
                grey_array2=np.array(grey)
                cols, rows = grey.size #because tuple returned is (width, height)
                #FIND INTENSITY VALUES AND SUM THEM OVER IMAGE SET (BEFORE RESIZING IMAGE)
                if j==1:
                    sum_of_intensities = np.zeros((rows, cols))
                sum_of_intensities = np.add(sum_of_intensities, grey_array2)
                #print(sum_of_intensities) THIS IS PROBLEMATIC DO NOT TRY IT. PRGRAM GETS OVERWHELMED
                j+=1
            #CONVERT ARRAY OF OVERLAYED INTENSITY VALUES TO IMAGE
            result_image = Image.fromarray(np.uint8(sum_of_intensities * intensity_scaling_factor), 'L')

            #RESIZE IMAGE
            width = resize_width #start with known desired width (set at top of script)
            wpercent = (width / float(result_image.size[0])) #create resizing percentage based on desired width
            height = int((float(result_image.size[1]) * float(wpercent))) #apply same percentage to height to determine resized height
            resized_result_image = result_image.resize((width, height), Image.ANTIALIAS) #resize image to new width and height
                
            #DISPLAY RESULTING OVERLAYED IMAGE
            tk_im1 = ImageTk.PhotoImage(resized_result_image)
            label = Label(image=tk_im1)
            label.image = tk_im1
            #self.clickBoundariesMessage=tk.Label(self, text='Click to generate boundries on image', font=SMALL_FONT, fg='white', bg='orchid4')
            #self.clickBoundariesMessage.grid(row=16, column=0, sticky='w', padx=5, pady=3)
            ca1 = tk.Canvas(self, width=tk_im1.width(), height=tk_im1.height(), #set canvas dimensions to size of image
                            borderwidth=0, highlightthickness=0)            
            ca1.grid(row=17, column=0, padx=1, pady=10)
            ca1.create_image(0,0, image=tk_im1, anchor='nw')
            self.reCalibrateButton = tk.Button(self, text='Re-Calibrate', font=SMALL_FONT)
            self.reCalibrateButton['command']=lambda:self.reCalibrate(done, ca1)
            self.reCalibrateButton.grid(row=17, column=1, padx=5)
            #If using clicked boundaries, comment out these two lines below
            done = tk.Button(self, text="Continue", font=SMALL_FONT, command = lambda: question())
            done.grid(row=18, column=0, sticky='w', padx=5, pady=10)

            #GET COORDINATES OF USER CLICKS AND CREATE BOUNDARY LINES ON IMAGE
            
##            def getLeft(eventL):
##                global x1
##                x1 = eventL.x
##                leftLine = ca1.create_line(x1, 0, x1, rows-1, fill='red', tag="leftLine") #I'm unsure why "rows" is out of scope of the image, but "rows-1" seems correct
##                ca1.bind("<Button 1>", getRight)
##                
##            ca1.bind("<Button 1>", getLeft)
##            done = tk.Button(self, text="Use these boundaries", font=SMALL_FONT, command = lambda: question(x1,x2))
##
##            def getRight(eventR):
##                global x2
##                x2 = eventR.x
##                rightLine = ca1.create_line(x2, 0, x2, rows-1, fill='red', tag="rightLine")
##                ca1.unbind("<Button 1>")
##                
##                done.grid(row=18, column=0, sticky='w', padx=5, pady=10)
##
            def question(): #If using clicked boundaries, x1 and x2 need to be args
                #question = messagebox.askyesno("Continue", "Are you satisfied with these boundaries?")
                question = True
                if question == True:
##                    #If left and rightl were clicked backwards, flip them
##                    if x1 > x2:
##                        old_x1=x1
##                        x1=x2
##                        x2=old_x1
                    #Open pre-existing file, "+" sign indicates file will be created if it doesn't already exist
                    f= open(file_coords, "w+")

                    #Write in coordinates (scaled back up)
                    #self.rescaled_x1 = int(x1/wpercent)
                    #self.rescaled_x2 = int(x2/wpercent)
                    
                    #NOTE: may need to adjust these hard-coded calibration boundaries
                    self.rescaled_x1 = 1000
                    self.rescaled_x2 = 1600
                    f.write(str(self.rescaled_x1) + ',' + str(self.rescaled_x2))
                    f.close()


####365 Green Channel
####__________________________________________________________________

                    green365 = 0
                    uvgreen365 = 0

                    original = Image.open(crop_file1)
                    original = original.rotate(180)
                    original = original.crop((self.rescaled_x1, 0, self.rescaled_x2, 1944))
                    rgb_array = np.array(original)
                    green_array = rgb_array[:,:,1]
                    cols, rows = original.size

                    #populate new matrix with zeros
                    average_intensities = np.zeros(1944) 
                    pixels = np.arange(0,1944)

                    #populate avg intensity matrix with average intensity across each row of pixels
                    for row in range(0,rows):
                        #nest loop from left to right
                        average_intensity = np.mean(green_array[row])
                        average_intensities[row] = average_intensity    
                        

                    # sort the data in x and rearrange y accordingly
                    sortId = np.argsort(pixels)
                    pixels = pixels[sortId]
                    average_intensities = average_intensities[sortId]

                    # this way the x-axis corresponds to the index of x
                    filtered = savgol_filter(average_intensities,51,1)

                    peaks, _ = find_peaks(filtered[0:1900], height = 0)  
                    peaks2 = peaks[0:int(len(peaks)/2)]

                    c = 0
                    a = 1
                    b = 1
                    avg = peaks2[a]
                    peaks3 = [0]
                    peaks3[0] = peaks2[0]

                    while( a < len(peaks2)):
                        
                        
                        if a == len(peaks2)-1 and c == 0:
                            avg = peaks2[a]
                            peaks3.append(avg)
                            
                        elif a == len(peaks2)-1 and c == 1:
                            a = 1000
                            peaks3.append(avg)
                            
                        else:
                            difference = abs(peaks2[a+1] - peaks2[a])
                            if difference <= 20:
                                while(difference <= 20):
                                    if a == len(peaks2)-1:
                                        difference = 100
                                    else:
                                        avg = avg + peaks2[a+1]
                                        b = b + 1
                                        a = a + 1
                                        if a == len(peaks2)-1:
                                            difference = 100
                                        else:
                                            difference = abs(peaks2[a+1] - peaks2[a])
                                            
                                avg = int(avg / b)
                                c = 1
                                peaks3.append(avg)
                                a = a+1

                            else:
                                avg = peaks2[a]
                                peaks3.append(avg)
                                a = a+1
                                c = 0
                                
                            if a == len(peaks2):
                                a = 1000
                            else:
                                avg = peaks2[a]
                                b = 1

                    intensity = [0]
                    intensity[0] = filtered[peaks3[0]]
                    z = 1

                    while(z < len(peaks3)):
                        intensity.append(filtered[peaks3[z]])
                        z = z+1

                    maxima = [0]

                    x = 0
                    max1 = 0
                    max1pixel = 0
                    max2 = 0
                    max2pixel = 0


                    while(x < int(len(intensity))):
                        if intensity[x] > max1 or intensity[x] > max2:
                            if intensity[x] > max1:
                                max2 = max1
                                max2pixel = max1pixel
                                max1 = intensity[x]
                                max1pixel = peaks3[x]
                            elif intensity[x] > max2:
                                max2 = intensity[x]
                                max2pixel = peaks3[x]
                            
                        x = x + 1

                    maxes = [max1pixel, max2pixel]
                    maxintensities = [max1, max2]

                    j = 0
                    while j < 2:
                        
                        if maxes[j] == max(maxes):
                            greenpixel = maxes[j]
                            greenintensity = maxintensities[j]
                        elif maxes[j] == min(maxes):
                            uvpixel = maxes[j]
                            uvintensity = maxintensities[j]
                        j = j+1


                    green365 = greenpixel
                    uvgreen365 = uvpixel

                    #
                    #PLOT FOR SHOWING FOUND PEAKS ON 385
#                     plt.figure(0)
#                     plt.xlabel('Pixel Number')
#                     plt.ylabel('Intensity')
#                     plt.suptitle('Intensity vs. Pixel Number with Peaks in Green channel at 365nm')
# 
#                     plt.plot(greenpixel, greenintensity, "o", c = 'green')
#                     plt.plot(uvpixel, uvintensity, "o", c = 'violet')
#                     plt.plot(pixels, filtered, color = 'red')


                    ##365 Red Channel repeat
                    ##__________________________________________________________________________________________

                    uvred365 = 0

                        
                    #Define 365 blue array and repeat steps
                    red_array = rgb_array[:,:,0]
                    cols, rows = original.size



                    #populate new matrix with zeros
                    average_intensities = np.zeros(1944) 
                    pixels = np.arange(0,1944)

                    #populate avg intensity matrix with average intensity across each row of pixels
                    for row in range(0,rows):
                        #nest loop from left to right
                        average_intensity = np.mean(red_array[row])
                        average_intensities[row] = average_intensity
                        

                    # sort the data in x and rearrange y accordingly
                    sortId = np.argsort(pixels)
                    pixels = pixels[sortId]
                    average_intensities = average_intensities[sortId]

                    # this way the x-axis corresponds to the index of x
                    filtered = savgol_filter(average_intensities,29,1)

                    peaks, _ = find_peaks(filtered[0:1900], height = 0)  
                    peaks2 = peaks[0:int(len(peaks)/2)]

                    c = 0
                    a = 1
                    b = 1
                    avg = peaks2[a]
                    peaks3 = [0]
                    peaks3[0] = peaks2[0]

                    while( a < len(peaks2)/2):
                        
                        
                        if a == len(peaks2)-1 and c == 0:
                            avg = peaks2[a]
                            peaks3.append(avg)
                            
                        elif a == len(peaks2)-1 and c == 1:
                            a = 1000
                            peaks3.append(avg)
                            
                        else:
                            difference = abs(peaks2[a+1] - peaks2[a])
                            if difference <= 20:
                                while(difference <= 20):
                                    if a == len(peaks2)-1:
                                        difference = 100
                                    else:
                                        avg = avg + peaks2[a+1]
                                        b = b + 1
                                        a = a + 1
                                        if a == len(peaks2)-1:
                                            difference = 100
                                        else:
                                            difference = abs(peaks2[a+1] - peaks2[a])
                                            
                                avg = int(avg / b)
                                c = 1
                                peaks3.append(avg)
                                a = a+1

                            else:
                                avg = peaks2[a]
                                peaks3.append(avg)
                                a = a+1
                                c = 0
                                
                            if a == len(peaks2):
                                a = 1000
                            else:
                                avg = peaks2[a]
                                b = 1

                    intensity = [0]
                    intensity[0] = filtered[peaks3[0]]
                    z = 1


                    while(z < len(peaks3)):
                        intensity.append(filtered[peaks3[z]])
                        z = z+1

                    maxima = [0]

                    x = 0
                    max1 = 0
                    max1pixel = 0
                    max2 = 0
                    max2pixel = 0


                    while(x < int(len(intensity))):
                        if intensity[x] > max1 or intensity[x] > max2:
                            if intensity[x] > max1:
                                max2 = max1
                                max2pixel = max1pixel
                                max1 = intensity[x]
                                max1pixel = peaks3[x]
                            elif intensity[x] > max2:
                                max2 = intensity[x]
                                max2pixel = peaks3[x]
                            
                        x = x + 1


                    maxes = [max1pixel, max2pixel]
                    maxintensities = [max1, max2]

                    if max1pixel > max2pixel:
                        uvpixel == max1pixel
                        uvintensity == max1
                    else:
                        uvpixel == max2pixel
                        uvintensity == max2

                        
                    uvred365 = uvpixel

                    #PLOT FOR SHOWING 365 PEAKS
#                     plt.figure(1)
#                     plt.xlabel('Pixel Number')
#                     plt.ylabel('Intensity')
#                     plt.suptitle('Intensity vs. Pixel Number with Peaks at Red365nm')
#                     plt.plot(uvpixel, uvintensity, "o", c = 'violet')
#                     plt.plot(pixels, filtered, color = 'red')



                    ####375 Green Channel
                    ####__________________________________________________________________

                    green375 = 0
                    uvgreen375 = 0

                    original = Image.open(crop_file2)
                    original = original.rotate(180)
                    ##original = original.crop((self.rescaled_x1, 0, self.rescaled_x2, 1944))
                    rgb_array = np.array(original)
                    green_array = rgb_array[:,:,1]
                    cols, rows = original.size

                    #populate new matrix with zeros
                    average_intensities = np.zeros(1944) 
                    pixels = np.arange(0,1944)

                    #populate avg intensity matrix with average intensity across each row of pixels
                    for row in range(0,rows):
                        #nest loop from left to right
                        average_intensity = np.mean(green_array[row])
                        average_intensities[row] = average_intensity    
                        

                    # sort the data in x and rearrange y accordingly
                    sortId = np.argsort(pixels)
                    pixels = pixels[sortId]
                    average_intensities = average_intensities[sortId]

                    # this way the x-axis corresponds to the index of x
                    filtered = savgol_filter(average_intensities,51,1)

                    peaks, _ = find_peaks(filtered[0:1900], height = 0)  
                    peaks2 = peaks[0:int(len(peaks)/2)]

                    c = 0
                    a = 1
                    b = 1
                    avg = peaks2[a]
                    peaks3 = [0]
                    peaks3[0] = peaks2[0]

                    while( a < len(peaks2)):
                        
                        
                        if a == len(peaks2)-1 and c == 0:
                            avg = peaks2[a]
                            peaks3.append(avg)
                            
                        elif a == len(peaks2)-1 and c == 1:
                            a = 1000
                            peaks3.append(avg)
                            
                        else:
                            difference = abs(peaks2[a+1] - peaks2[a])
                            if difference <= 20:
                                while(difference <= 20):
                                    if a == len(peaks2)-1:
                                        difference = 100
                                    else:
                                        avg = avg + peaks2[a+1]
                                        b = b + 1
                                        a = a + 1
                                        if a == len(peaks2)-1:
                                            difference = 100
                                        else:
                                            difference = abs(peaks2[a+1] - peaks2[a])
                                            
                                avg = int(avg / b)
                                c = 1
                                peaks3.append(avg)
                                a = a+1

                            else:
                                avg = peaks2[a]
                                peaks3.append(avg)
                                a = a+1
                                c = 0
                                
                            if a == len(peaks2):
                                a = 1000
                            else:
                                avg = peaks2[a]
                                b = 1

                    intensity = [0]
                    intensity[0] = filtered[peaks3[0]]
                    z = 1

                    while(z < len(peaks3)):
                        intensity.append(filtered[peaks3[z]])
                        z = z+1

                    maxima = [0]

                    x = 0
                    max1 = 0
                    max1pixel = 0
                    max2 = 0
                    max2pixel = 0


                    while(x < int(len(intensity))):
                        if intensity[x] > max1 or intensity[x] > max2:
                            if intensity[x] > max1:
                                max2 = max1
                                max2pixel = max1pixel
                                max1 = intensity[x]
                                max1pixel = peaks3[x]
                            elif intensity[x] > max2:
                                max2 = intensity[x]
                                max2pixel = peaks3[x]
                            
                        x = x + 1


                    maxes = [max1pixel, max2pixel]
                    maxintensities = [max1, max2]

                    j = 0
                    while j < 2:
                        
                        if maxes[j] == max(maxes):
                            greenpixel = maxes[j]
                            greenintensity = maxintensities[j]
                        elif maxes[j] == min(maxes):
                            uvpixel = maxes[j]
                            uvintensity = maxintensities[j]
                        j = j+1


                    green375 = greenpixel
                    uvgreen375 = uvpixel

                    #
                    #PLOT FOR SHOWING FOUND PEAKS ON 385
#                     plt.figure(2)
#                     plt.xlabel('Pixel Number')
#                     plt.ylabel('Intensity')
#                     plt.suptitle('Intensity vs. Pixel Number with Peaks in Green channel at 365nm')
# 
#                     plt.plot(greenpixel, greenintensity, "o", c = 'green')
#                     plt.plot(uvpixel, uvintensity, "o", c = 'violet')
#                     plt.plot(pixels, filtered, color = 'red')



                    ##375 Red Channel repeat
                    ##__________________________________________________________________________________________

                    uvred375 = 0

                        
                    #Define 365 blue array and repeat steps
                    red_array = rgb_array[:,:,0]
                    cols, rows = original.size



                    #populate new matrix with zeros
                    average_intensities = np.zeros(1944) 
                    pixels = np.arange(0,1944)

                    #populate avg intensity matrix with average intensity across each row of pixels
                    for row in range(0,rows):
                        #nest loop from left to right
                        average_intensity = np.mean(red_array[row])
                        average_intensities[row] = average_intensity
                        

                    # sort the data in x and rearrange y accordingly
                    sortId = np.argsort(pixels)
                    pixels = pixels[sortId]
                    average_intensities = average_intensities[sortId]

                    # this way the x-axis corresponds to the index of x
                    filtered = savgol_filter(average_intensities,29,1)

                    peaks, _ = find_peaks(filtered[0:1900], height = 0)  
                    peaks2 = peaks[0:int(len(peaks)/2)]

                    c = 0
                    a = 1
                    b = 1
                    avg = peaks2[a]
                    peaks3 = [0]
                    peaks3[0] = peaks2[0]

                    while( a < len(peaks2)/2):
                        
                        
                        if a == len(peaks2)-1 and c == 0:
                            avg = peaks2[a]
                            peaks3.append(avg)
                            
                        elif a == len(peaks2)-1 and c == 1:
                            a = 1000
                            peaks3.append(avg)
                            
                        else:
                            difference = abs(peaks2[a+1] - peaks2[a])
                            if difference <= 20:
                                while(difference <= 20):
                                    if a == len(peaks2)-1:
                                        difference = 100
                                    else:
                                        avg = avg + peaks2[a+1]
                                        b = b + 1
                                        a = a + 1
                                        if a == len(peaks2)-1:
                                            difference = 100
                                        else:
                                            difference = abs(peaks2[a+1] - peaks2[a])
                                            
                                avg = int(avg / b)
                                c = 1
                                peaks3.append(avg)
                                a = a+1

                            else:
                                avg = peaks2[a]
                                peaks3.append(avg)
                                a = a+1
                                c = 0
                                
                            if a == len(peaks2):
                                a = 1000
                            else:
                                avg = peaks2[a]
                                b = 1

                    intensity = [0]
                    intensity[0] = filtered[peaks3[0]]
                    z = 1

                    while(z < len(peaks3)):
                        intensity.append(filtered[peaks3[z]])
                        z = z+1

                    maxima = [0]

                    x = 0
                    max1 = 0
                    max1pixel = 0
                    max2 = 0
                    max2pixel = 0


                    while(x < int(len(intensity))):
                        if intensity[x] > max1 or intensity[x] > max2:
                            if intensity[x] > max1:
                                max2 = max1
                                max2pixel = max1pixel
                                max1 = intensity[x]
                                max1pixel = peaks3[x]
                            elif intensity[x] > max2:
                                max2 = intensity[x]
                                max2pixel = peaks3[x]
                            
                        x = x + 1


                    maxes = [max1pixel, max2pixel]
                    maxintensities = [max1, max2]


                    if max1pixel > max2pixel:
                        uvpixel == max1pixel
                        uvintensity == max1
                    else:
                        uvpixel == max2pixel
                        uvintensity == max2

                        
                    uvred375 = uvpixel

                    #PLOT FOR SHOWING UV-375 PEAKS
#                     plt.figure(3)
#                     plt.xlabel('Pixel Number')
#                     plt.ylabel('Intensity')
#                     plt.suptitle('Intensity vs. Pixel Number with Peaks with Red 375nm')
#                     plt.plot(uvpixel, uvintensity, "o", c = 'violet')
#                     plt.plot(pixels, filtered, color = 'red')


                    x = np.array([uvgreen365, uvred365, uvgreen375, uvred375, green365, green375]).reshape(-1,1)
                    y = np.array([390, 390, 390, 390, 515, 515])

                    x = x[:,0]

                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    print("slope: ", slope)
                    print("R^2: ", r_value)
                    self.data_to_append[9] = slope
                    self.data_to_append[10] = intercept

                    if r_value < 0.99: 
                        done.destroy()
                        ca1.destroy()
                        self.reCalibrateButton.destroy()
                        #self.clickBoundariesMessage.destroy()
                        self.dataError = tk.Message(self,
                                                    text= 'ERROR! R SQUARED BELOW ACCEPTABLE VALUE. Try rotating vial,and/or using wider boundaries.',
                                                    font=LARGE_FONT)
                        self.dataError.grid(row=20, column=0, pady=2)
                        self.okButton4=tk.Button(self, text='OK', font=LARGE_FONT)
                        self.okButton4['command']=lambda:self.clearR2ErrorAlert()
                        self.okButton4.grid(row=21, column=0, pady=2)
                        
                    
                    else:
                        #Hide everything associated with calibration and instructions
                        #self.clickBoundariesMessage.grid_forget()
                        ca1.grid_forget()
                        #self.showInstructionsButton.grid_forget()
                        self.calibrationLabel.grid_forget()
                        done.grid_forget()
                        self.reCalibrateButton.destroy()
                        
                        #PLOT SHOWING LINEAR REGRESSION BASED ON PEAKS
#                         plt.figure(4)
                        x0 = 0
                        xf = int((750 - intercept) / slope)
                        xaxis = np.linspace(x0, xf, abs(xf-x0))
#                         plt.plot(green365, 515,"o", color = 'green')
#                         plt.plot(green375, 515, "x", color = 'green')
#                         plt.plot(uvgreen365, 390, "o", color = 'violet')
#                         plt.plot(uvgreen375, 390, "x", color = 'violet')
#                         plt.plot(uvred365, 390, "o", color = 'red')
#                         plt.plot(uvred375, 390, "x", color = 'red')
#                         plt.plot(xaxis, slope*xaxis + intercept, color = 'black')
#                         equation = ('y = %f.2 x + %.5f' % (slope, intercept))
#                         plt.text(300,320, 'r^2 = %f' %r_value)
#                         plt.text(300, 350, equation)
#                         plt.suptitle('linear regression')
#                         plt.xlabel('pixel number')
#                         plt.ylabel('wavelength')
                        #plt.show()

#                         minwavelength = int(intercept)
#                         print(minwavelength)

#                         #ERROR TO ADJUST CAMERA IF MIN WAVELENGTH IS NOT CORRECT
#                         if minwavelength > 350: 
#                             self.dataError = tk.Message(self, text= 'ERROR! MINIMUM WAVELENGTH DETECTED IS ABOVE 350nm PLEASE ADJUST PI CAMERA.')
#                             self.dataError.grid(row=16, column=0, pady=2)
#                             self.okButton4=tk.Button(self, text='OK')
#                             self.okButton4['command']=lambda:self.clearDataErrorAlert()
#                             self.okButton4.grid(row=17, column=0, pady=2)
                            
#                         else:
                            
                        minwavelength = 350
                        maxwavelength = 750
                        wavelengths = list(range(minwavelength, maxwavelength+1))
                        
                        #find desired pixels based on wavelengths put through linear equation
                        u = 1
                        desiredpixels = [0]
                        desiredpixels[0] = int((wavelengths[0] - intercept) / slope)

                        while u < len(wavelengths):
                            desired= int((wavelengths[u]  - intercept) / slope)
                            desiredpixels.append(desired)
                            u = u+1

                        #find desired intensities 
                        desiredintensity = [0]
                        desiredintensity[0] = filtered[desiredpixels[0]]
                        s = 1
                        while s < len(desiredpixels):
                            desiredint = filtered[desiredpixels[s]]
                            desiredintensity.append(desiredint)
                            
                            s = s+1
                        
                        #print(desiredpixels)
                        filename = '/home/pi/Desktop/Data/calibration_data/desiredpixels.txt'
                        np.savetxt(filename, desiredpixels)

                        #PLOT SHOWING FINAL INTENSITY VS WAVELENGTH OUTPUT. NOT NECESSARY FOR THE DCM CALIBRATION
#                           plt.figure(3)
#                           plt.xlabel('Wavelength(nm)')
#                           plt.ylabel('Intensity')
#                           plt.suptitle('Wavelength vs. Intensity')
#                           plt.plot(wavelengths, desiredintensity)
#

                        #Display success message
                        self.savedMessage= tk.Label(self, text="Calibration saved!",
                             font=SMALL_FONT, fg='white', bg='orchid4')
                        self.savedMessage.grid(row=16, column=0, sticky='w', padx=5, pady=10)
                        self.startAnalyzingSamplesButton=tk.Button(self, text='Start Analyzing Samples', font=SMALL_FONT)
                        self.startAnalyzingSamplesButton['command']=lambda: self.startAnalyzingSamples()
                        self.startAnalyzingSamplesButton.grid(row=17, column=0, sticky='w', padx=5, pady=10)
                #NOTE: uncomment this if using clicked boundaries 
##                elif question == False:
##                    ca1.delete("leftLine", "rightLine")
##                    done.grid_forget() #hide done button for now
##                    ca1.bind("<Button 1>", getLeft) #begin again, capture new boundary lines
        except FileNotFoundError:
            #will occur if folder names are changed, or if an image cannot be generated by camera
            #in testing mode, this occurs if name entered is not an existing image in Data folder
            self.dataError = tk.Message(self,
                                        text= 'Error - please ensure all relevant files are located in the Data folder on this device. If that is not the issue, there may be a problem with the camera.',
                                        font=LARGE_FONT)
            self.dataError.grid(row=16, column=0, pady=2)
            self.okButton4=tk.Button(self, text='OK', font=LARGE_FONT)
            self.okButton4['command']=lambda:self.clearDataErrorAlert()
            self.okButton4.grid(row=17, column=0, pady=2)
        
        

    
    def clearDataErrorAlert(self):
        self.dataError.destroy()
        self.okButton4.destroy()
        
    def clearR2ErrorAlert(self):
        self.okButton4.destroy()
        self.dataError.destroy()
        self.runCalibration()
    
    def reCalibrate(self, done, ca1):
        #Hide everything associated with calibration
        #self.clickBoundariesMessage.grid_forget()
        ca1.grid_forget()
        done.grid_forget()
        self.reCalibrateButton.destroy()
        #Rerun calibration
        self.runCalibration()
        

        
#----------THIS IS WHERE CALIBRATION ENDS AND SAMPLE ANALYSIS BEGINS---------------------------------------

    def startAnalyzingSamples(self):
        self.savedMessage.destroy()
        self.startAnalyzingSamplesButton.grid_forget()
        #Hide instructions if the user still has them open
##        if str(self.showInstructionsButton["state"])=='disabled':
##            self.hideInstructions()
##        else:
##            self.showInstructionsButton.grid_forget()
        global x1, x2
        #open crop coordinates file
        try:
            f = open(file_coords, 'r')
            try:
                for line in f:
                        currentline = line.split(',')
                        x1 = int(currentline[0])
                        x2 = int(currentline[1])
                        #print('x1=', x1, 'x2=', x2)
            finally:
                #Insert oil message
                self.insertMessage = tk.Message(self, text='Insert oil sample and press OK.', font=SMALL_FONT)
                self.insertMessage.grid(row=2, column=0, sticky='w', padx=5, pady=5)
                #CREATE CANVAS FOR DILUTION IMAGE
                self.dilcanvas = tk.Canvas(self, width=500, height=200, borderwidth=0, highlightthickness=0)
                self.dilcanvas.grid(row=3, column=0, sticky='w', padx=10)
                #OPEN AND RESIZE DILUTION IMAGE
                dil = Image.open(file_dil)
                dil_resized = dil.resize((500, 200), Image.ANTIALIAS)
                #DISPLAY DILUTION IMAGE
                tk_dil = ImageTk.PhotoImage(dil_resized)
                labeldil = Label(image=tk_dil)
                labeldil.image = tk_dil
                self.dilcanvas.create_image(0,0, image=tk_dil, anchor='nw')
                #OK button
                self.okButton1 = tk.Button(self, text='OK', font=LARGE_FONT)
                self.okButton1['command'] = lambda: self.clearInsertAlert()
                self.okButton1.grid(row=4, column=0, sticky='w', padx=5)
                #switch to start page
                f.close()
        except FileNotFoundError:
            self.calError = tk.Message(self, text='Could not open calibration file! Please calibrate device before using.', font=LARGE_FONT)
            self.calError.grid(row=2, column=0, padx=2, pady=2)
            self.okButton2 = tk.Button(self, text='OK', font=LARGE_FONT)
            self.okButton2['command'] = lambda: self.clearFileAlert()
            self.okButton2.grid(row=3, column=0)
            return

    def clearFileAlert(self):
        self.calError.destroy()
        self.okButton2.destroy()
        
    def clearInsertAlert(self):
        self.insertMessage.destroy()
        self.okButton1.destroy()
        self.dilcanvas.destroy()
        self.enterOilInfo()

    def enterOilInfo(self):
        #Sample Name
        self.nameLabel = tk.Label(self, text="Sample Name (Optional):", font=SMALL_FONT, fg='white', bg='orchid4')
        self.nameLabel.grid(row=2, column=0, columnspan=4, sticky='w', padx=5)
        self.name = StringVar()
        self.nameEntry = tk.Entry(self, textvariable=self.name)
        self.nameEntry.grid(row=3, column=0, columnspan=4, sticky='w', padx=5, pady=10)
        #Replicate Dropdown
        self.replicateLabel=tk.Label(self, text='Replicate #:',font=SMALL_FONT, fg='white', bg='orchid4')
        self.replicateLabel.grid(row=2, column=6, columnspan=2, sticky='w')
        self.replicate=IntVar()
        self.replicate.set("1") #default value
        self.replicateDropdown = OptionMenu(self, self.replicate, "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20")
        self.replicateDropdown.grid(row=3, column=6, columnspan=2, sticky='w', pady=10)
        #Manufacturer Info
        self.manufacturerLabel = tk.Label(self, text='Manufacturer:',font=SMALL_FONT, fg='white', bg='orchid4')
        self.manufacturerLabel.grid(row=2, column=4, padx=3, sticky='w', columnspan=2)
        self.manufacturer=StringVar()
        self.manufacturer.set("None") #default value
        self.manufacturerDropdown=tk.OptionMenu(self, self.manufacturer,"None", "Other (see comments)", "GS", "HD", "S-Oil", "SK")
        self.manufacturerDropdown.grid(row=3, column=4, padx=3, pady=10, sticky='w', columnspan=2)
        #Specific Oil Type Dropdown
        self.dropdownLabel = tk.Label(self, text='Classification:',font=SMALL_FONT, fg='white', bg='orchid4')
        self.dropdownLabel.grid(row=4, column=0, columnspan=3, sticky='w', padx=5, pady=3)
        self.oiltype = StringVar(self)
        self.oiltype.set("Unspecified or Unknown") #default value
        self.dropdown = OptionMenu(self, self.oiltype,"Unspecified or Unknown", "Other (see comments)", "Crude", "Bunker-C", "Marine Fuel", "Lubricant", "Gasoline", "Diesel", "Bunker-A", "Bunker-B", "Marine Gas", "Marine Diesel")
        self.dropdown.grid(row=5, column=0, columnspan=4, sticky='w', padx=5, pady=3)
        #Comments Box
        self.commentsLabel=tk.Label(self, text='Comments (Optional):',font=SMALL_FONT, fg='white', bg='orchid4')
        self.commentsLabel.grid(row=6, column=0, columnspan=5, sticky='w', padx=5, pady=3)
        self.comments=StringVar()
        self.commentsEntry=tk.Entry(self, textvariable=self.comments)
        self.commentsEntry.grid(row=7, column=0, columnspan=5, sticky='w', padx=5, pady=3)
        #SARA Content Entry Dropdowns
        self.saturateValue1 = IntVar()
        self.saturateValue2 = IntVar()
        self.aromaticValue1 = IntVar()
        self.aromaticValue2 = IntVar()
        self.resinValue1 = IntVar()
        self.resinValue2 = IntVar()
        self.asphalteneValue1 = IntVar()
        self.asphalteneValue2 = IntVar()
        self.saturateValue1.set("0") #default value
        self.saturateValue2.set("0") #default value
        self.aromaticValue1.set("0") #default value
        self.aromaticValue2.set("0") #default value
        self.resinValue1.set("0") #default value
        self.resinValue2.set("0") #default value
        self.asphalteneValue1.set("0") #default value
        self.asphalteneValue2.set("0") #default value
        self.saturateLabel = tk.Label(self, text='Saturate Content:',font=SMALL_FONT, fg='white', bg='orchid4')
        self.saturateLabel.grid(row=4, column=4, padx=10, sticky='w')
        self.saturateDropdown1 = OptionMenu(self, self.saturateValue1, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.saturateDropdown1.grid(row=4, column=5, sticky='w', pady=3)
        self.saturateDropdown2 = OptionMenu(self, self.saturateValue2, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.saturateDropdown2.grid(row=4, column=6, sticky='w', pady=3)
        self.saturatePercentSign=tk.Label(self, text='%    ', fg='white', bg='orchid4')
        self.saturatePercentSign.grid(row=4, column=7, padx=3, sticky='w')
        self.aromaticLabel = tk.Label(self, text='Aromatic Content:',font=SMALL_FONT, fg='white', bg='orchid4')
        self.aromaticLabel.grid(row=5, column=4, padx=10, sticky='w')
        self.aromaticDropdown1 = OptionMenu(self, self.aromaticValue1, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.aromaticDropdown1.grid(row=5, column=5, sticky='w', pady=3)
        self.aromaticDropdown2 = OptionMenu(self, self.aromaticValue2, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.aromaticDropdown2.grid(row=5, column=6, sticky='w', pady=3)
        self.aromaticPercentSign=tk.Label(self, text='%    ', fg='white', bg='orchid4')
        self.aromaticPercentSign.grid(row=5, column=7, padx=3, sticky='w')
        self.resinLabel = tk.Label(self, text='Resin Content:',font=SMALL_FONT, fg='white', bg='orchid4')
        self.resinLabel.grid(row=6, column=4, padx=10, sticky='w')
        self.resinDropdown1 = OptionMenu(self, self.resinValue1, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.resinDropdown1.grid(row=6, column=5, pady=3)
        self.resinDropdown2 = OptionMenu(self, self.resinValue2, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.resinDropdown2.grid(row=6, column=6, pady=3)
        self.resinPercentSign=tk.Label(self, text='%    ', fg='white', bg='orchid4')
        self.resinPercentSign.grid(row=6, column=7, padx=3, sticky='w')
        self.asphalteneLabel = tk.Label(self, text='Asphaltene Content:',font=SMALL_FONT, fg='white', bg='orchid4')
        self.asphalteneLabel.grid(row=7, column=4, padx=10, sticky='w')
        self.asphalteneDropdown1 = OptionMenu(self, self.asphalteneValue1, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.asphalteneDropdown1.grid(row=7, column=5, pady=3)
        self.asphalteneDropdown2 = OptionMenu(self, self.asphalteneValue2, "0", "1", "2", "3", "4", "5", "6", "7", "8","9")
        self.asphalteneDropdown2.grid(row=7, column=6, pady=3)
        self.asphaltenePercentSign=tk.Label(self, text='%    ', fg='white', bg='orchid4')
        self.asphaltenePercentSign.grid(row=7, column=7, padx=3, sticky='w')
        #self.startButton.config(state=DISABLED) #prevent user from running another sample while this one is running
        self.sampleSubmit = tk.Button(self, text='Submit and Start', font=LARGE_BOLD_FONT)
        self.sampleSubmit['command'] = lambda: self.appendInitialData()
        self.sampleSubmit.grid(row=14, column=0, padx=5, pady=10)
        
    def appendInitialData(self):
        #Get the current value of the ID counter
        IDfile = open('/home/pi/Desktop/Data/IDcounter.txt', 'r')
        self.ID=0
        for number in IDfile:
            newNumber=number.split(',')
            self.ID=newNumber[0]
        IDfile.close()
        #Write the current ID to the list to be appended to CSV file
        self.data_to_append[0]=self.ID
        #Write the user-defined manufacturer name to the list to be appended to CSV file
        self.data_to_append[1]=self.manufacturer.get()
        #Write the user-defined sample name to the list to be appended to CSV file
        self.data_to_append[2]=self.name.get()
        #Write the user-defined replicate number to the list to be appended to CSV file
        self.data_to_append[3]=self.replicate.get()
        #Convert user-defined specific classification to broad class,
        #and write to list to be appended to CSV file
        if self.oiltype.get()=="Crude":
            self.data_to_append[4]=0
        elif self.oiltype.get()=="Bunker-C" or self.oiltype.get()=="Marine Fuel":
            self.data_to_append[4]=1
        elif self.oiltype.get()=="Lubricant":
            self.data_to_append[4]=2
        elif self.oiltype.get()=="Gasoline" or self.oiltype.get()=="Diesel" or self.oiltype.get()=="Bunker-A" or self.oiltype.get()=="Bunker-B" or self.oiltype.get()=="Marine Gas" or self.oiltype.get()=="Marine Diesel":
            self.data_to_append[4]=3
#         else: #If user selected "Other" or "Unspecified or Unknown"
#             self.data_to_append[4]=4
        #Write specific classification, SARA content,
        #and left and right bounds to list to be appended to CSV file
        self.data_to_append[5]=self.oiltype.get()
        saturate = int(self.saturateValue1.get())*10+int(self.saturateValue2.get())
        aromatic = int(self.aromaticValue1.get())*10+int(self.aromaticValue2.get())
        resin = int(self.resinValue1.get())*10+int(self.resinValue2.get())
        asphaltene = int(self.asphalteneValue1.get())*10+int(self.asphalteneValue2.get())
        self.data_to_append[6]=[saturate, aromatic, resin, asphaltene]
        self.data_to_append[7]=self.rescaled_x1
        self.data_to_append[8]=self.rescaled_x2
        self.data_to_append[15]=self.comments.get()
        
        self.SubmitAndStart()
        
    def SubmitAndStart(self):
        #Remove entry items from GUI while sample is running
        self.nameLabel.destroy()
        self.nameEntry.destroy()
        self.manufacturerLabel.destroy()
        self.manufacturerDropdown.destroy()
        self.replicateLabel.destroy()
        self.replicateDropdown.destroy()
        self.dropdownLabel.destroy()
        self.dropdown.destroy()
        self.commentsLabel.destroy()
        self.commentsEntry.destroy()
        self.saturateLabel.destroy()
        self.saturateDropdown1.destroy()
        self.saturateDropdown2.destroy()
        self.aromaticLabel.destroy()
        self.aromaticDropdown1.destroy()
        self.aromaticDropdown2.destroy()
        self.resinLabel.destroy()
        self.resinDropdown1.destroy()
        self.resinDropdown2.destroy()
        self.asphalteneLabel.destroy()
        self.asphalteneDropdown1.destroy()
        self.asphalteneDropdown2.destroy()
        self.saturatePercentSign.destroy()
        self.aromaticPercentSign.destroy()
        self.resinPercentSign.destroy()
        self.asphaltenePercentSign.destroy()
        
        self.sampleSubmit.destroy()
        file1 = '/home/pi/Desktop/Data/image_data/%s_365.jpg' %self.ID #365nm image taken
        file2 = '/home/pi/Desktop/Data/image_data/%s_375.jpg' %self.ID #375nm image taken
        file3 = '/home/pi/Desktop/Data/image_data/%s_385.jpg' %self.ID #385nm image taken
#         intensity1 = '/home/pi/Desktop/Data/intensity_data/%s_365_intensities.txt' %name
#         intensity2 = '/home/pi/Desktop/Data/intensity_data/%s_375_intensities.txt' %name
#         intensity3 = '/home/pi/Desktop/Data/intensity_data/%s_385_intensities.txt' %name
#         cropintensity1 = '/home/pi/Desktop/Data/cropped_intensity_data/%s_365_cropped_intensities.txt' %name
#         cropintensity2 = '/home/pi/Desktop/Data/cropped_intensity_data/%s_375_cropped_intensities.txt' %name
#         cropintensity3 = '/home/pi/Desktop/Data/cropped_intensity_data/%s_385_cropped_intensities.txt' %name

        #GPIO setup for board and UV LEDs
        GPIO.setmode(GPIO.BCM) # Broadcom pin-numbering scheme
        GPIO.setup(pin365, GPIO.OUT) # LED pins set as outputs
        GPIO.setup(pin375, GPIO.OUT)
        GPIO.setup(pin385, GPIO.OUT)
        GPIO.output(pin365, GPIO.LOW)
        GPIO.output(pin375, GPIO.LOW)
        GPIO.output(pin385, GPIO.LOW)
        #take images with each of the LEDs on in succession
        GPIO.output(pin365, GPIO.HIGH)
        camera.capture(file1)
        GPIO.output(pin365, GPIO.LOW)

        GPIO.output(pin375, GPIO.HIGH)
        camera.capture(file2)
        GPIO.output(pin375, GPIO.LOW)

        GPIO.output(pin385, GPIO.HIGH)
        camera.capture(file3)
        GPIO.output(pin385, GPIO.LOW)
        try:
            #analyze images
            filez = [file1, file2, file3]
            lst = list(filez)
            i=1
            for image in lst:
                #filename = intensity1 if i==1 else intensity2 if i==2 else intensity3
                #savename = cropintensity1 if i == 1 else cropintensity2 if i == 2 else cropintensity3
                
                #OPEN IMAGE
                original = Image.open(image)
                #Rotate image for finding peaks
                original = original.rotate(180)
                
                #CROP ORIGINAL IMAGE TO BOUNDARIES SPECIFIED BY CALIBRATION
                cropped=original.crop((x1, 0, x2, 1944))
                cropped.save(file1 if i==1 else file2 if i==2 else file3)

                #CONVERT CROPPED IMAGE TO GRAYSCALE NUMPY ARRAY
                grey=cropped.convert('L')
                #grey.save("greyimage.jpg") #may not need this
                grey_array=np.array(grey)

                #Find average intensity for each row in image
                average_intensities = np.zeros(1944) #will there be a problem for zeros?
                cols, rows = grey.size
                for row in range(0, rows):
                    average_intensity = np.mean(grey_array[row])
                    average_intensities[row] = average_intensity        

                print('Analyzing %s...' %image)
                #print(average_intensities)
                #np.savetxt(filename, average_intensities)
                
                f = open('/home/pi/Desktop/Data/calibration_data/desiredpixels.txt', 'r+')
                data = np.loadtxt(f, dtype = int)

                desiredintensities = [0]
                desiredintensities[0] = average_intensities[data[0]]

                s = 1
                while s < len(data):
                    desiredint = average_intensities[data[s]]
                    desiredintensities.append(desiredint)
                    s = s+1
                #np.savetxt(savename, desiredintensities)
                #Add data to list (to be appended to CSV file for storage)
                #Cut off first 20 wavelengths (350-369) and last 50 wavelengths (701-750)
                if i==1:
                    self.data_to_append[12] = desiredintensities[20:351]
                elif i==2:
                    self.data_to_append[13] = desiredintensities[20:351]
                else:
                    self.data_to_append[14] = desiredintensities[20:351]
                i+=1
            #Take average over entire spectrum (all 3 LEDs) and send error to screen if the average intensity is too low
            all_data = np.concatenate((self.data_to_append[12], self.data_to_append[13], self.data_to_append[14]), axis=0)
            overall_average = np.average(all_data)
            if overall_average < 3:
                results_string = str('Intensity is too low! Please check that oil sample was added or increase concentration.')
            else:

                #*****Dimensionality Reduction and Machine Learning********************************************
                
                # Read in the training data for type classification:
                os.chdir('/home/pi/Desktop/Data/')
                train_data = pd.read_csv('OTA2_types_doublesliced.csv') 
                y = train_data.loc[:,'best1'].values
                x = train_data.drop(['ID', 'Name', 'OldAlgorithmClassification', 'best1'],axis=1)

                #Read in the training data for SARA content classification (4 class dataset)
                train_data_SARA = pd.read_csv('OTA2_SARA_doublesliced.csv')
                y_sat = train_data_SARA.loc[:,'levelSaturate']
                y_aro = train_data_SARA.loc[:,'levelAromatic']
                y_res = train_data_SARA.loc[:,'levelResin']
                y_asp = train_data_SARA.loc[:,'levelAsphaltene']
                x_SARA = train_data_SARA.drop(['ID', 'Name', 'OldAlgorithmClassification', '%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin', 'levelResin', '%Asphaltene', 'levelAsphaltene'],axis=1)

                #Define the test data from sample run:
                testx = self.data_to_append[12] + self.data_to_append[13] + self.data_to_append[14]
                testxarray = np.asarray(testx)
                testxarray = testxarray.reshape(1, -1)

                #PCA for type classification
                self.PCA = PCA(n_components=10, svd_solver='randomized', whiten=True)
                X = self.PCA.fit_transform(x)
                X_test = self.PCA.transform(testxarray)

                #PCA for SARA content classification
                self.PCA_SARA = PCA(n_components=10, svd_solver='randomized', whiten=True)
                X_SARA = self.PCA_SARA.fit_transform(x_SARA)
                X_test_SARA = self.PCA_SARA.transform(testxarray)
                
                #SVM for type classification
                #Create classifier
                clf = svm.SVC(kernel='rbf', gamma=0.01, C=1)
                #Generate predicted class (y=label 1,2,or 3; X=PCs) *****THIS IS THE CLASSIFIER*****
                y_pred = clf.fit(X, y).predict(X_test)
                y_distance = clf.decision_function(X_test)
                #Create element to add to csv later
                self.data_to_append[11]=y_pred[0]
                        
                #SVM for SARA content classification
                #Create classifiers
                clf_sat = svm.SVC(kernel='rbf', gamma=1, C=10)
                clf_aro = svm.SVC(kernel='rbf', gamma=1, C=10)
                clf_res = svm.SVC(kernel='rbf', gamma=1, C=10)
                clf_asp = svm.SVC(kernel='rbf', gamma=0.1, C=100)
                #Generate predicted classes
                y_sat_pred = clf_sat.fit(X_SARA, y_sat).predict(X_test_SARA)
                y_aro_pred = clf_aro.fit(X_SARA, y_aro).predict(X_test_SARA)
                y_res_pred = clf_res.fit(X_SARA, y_res).predict(X_test_SARA)
                y_asp_pred = clf_asp.fit(X_SARA, y_asp).predict(X_test_SARA)
                #Create elements to add to csv later
                self.data_to_append[16] = y_sat_pred[0]
                self.data_to_append[17] = y_aro_pred[0]
                self.data_to_append[18] = y_res_pred[0]
                self.data_to_append[19] = y_asp_pred[0]

                #Perform secondary classification if the predicted type is light fuel oil
                if y_pred[0]==1:
                    #Read in training data
                    train_data_LF = pd.read_csv('OTA2_lightfuels_doublesliced.csv')
                    y_LF = train_data_LF.loc[:,'best1'].values
                    x_LF = train_data_LF.drop(['ID', 'Name', 'OldAlgorithmClassification', 'best1'],axis=1)
                    
                    #PCA for light fuels
                    self.PCA_LF = PCA(n_components=10, svd_solver='randomized', whiten=True)
                    X_LF = self.PCA_LF.fit_transform(x_LF)
                    X_test_LF = self.PCA_LF.transform(testxarray)
                    
                    #SVM for light fuels
                    #Create classifier
                    clf_LF = svm.SVC(kernel='rbf', gamma=0.01, C=10)
                    #Generate predicted class (y=label 1 or 4; X=PCs) *****THIS IS THE CLASSIFIER*****
                    y_pred_LF = clf_LF.fit(X_LF, y_LF).predict(X_test_LF)
                    y_distance_LF = clf_LF.decision_function(X_test_LF)
                    #Update element to add to csv later 
                    self.data_to_append[11]=y_pred_LF[0]
                else: #set y_pred_LF to 0 for warnings later
                    y_pred_LF = [0]  
                
                #Generate predicted types to display to user
                #if y_pred[0]==0: #Classified as Crude Oil
                    #results_string = str('Predicted Oil Type: Crude Oil')
                if y_pred[0]==3: #Classified as Heavy Fuel Oil
                    results_string = str('Predicted Oil Type: Heavy Fuel Oil')
                elif y_pred[0]==2: #Classified as Lubricant Oil
                    results_string = str('Predicted Oil Type: Lubricant Oil')
                elif y_pred[0]==1: #Classified as Light Fuel Oil
                    if y_pred_LF[0]==1: #Classified as MGO-type oil
                        results_string = str('Predicted Oil Type: MGO')
                    elif y_pred_LF[0]==4: #Classified as Bunker A-type oil
                        results_string = str('Predicted Oil Type: Bunker A')
                    else: #If there's a typo in the training set
                        results_string = str('Error: Typos in training data (LF) and class is not 1 or 4!')
                else:
                    #If there's a typo in the training set
                    results_string = str('Error: Typos in training data (types) and class is not 0, 1, 2,or 3!')

                #Warnings of closeness to decision boundaries for primary classification
                if y_pred[0] == 1: #light fuel
                    if y_distance[0,1] > 1.1:
                        results_string = results_string + str('\n\nWarning: sample could be lubricant oil!')
                    if y_distance[0,2] > 1.1:
                        results_string = results_string + str('\n\nWarning: sample could be heavy fuel oil!')
                if y_pred[0] == 2: #lubricant
                    if y_distance[0,0] > 1.1:
                        results_string = results_string + str('\n\nWarning: sample could be light fuel oil (MGO or Bunker A)!')
                    if y_distance[0,2] > 1.1:
                        results_string = results_string + str('\n\nWarning: sample could be heavy fuel oil!')
                if y_pred[0] == 3: #heavy fuel
                    if y_distance[0,0] > 1.1:
                        results_string = results_string + str('\n\nWarning: sample could be light fuel oil (MGO or Bunker A)!')
                    if y_distance[0,1] > 1.1:
                        results_string = results_string + str('\n\nWarning: sample could be lubricant oil!')

                #Warnings of closeness to decision boundaries for secondary classification
                if y_pred_LF[0] == 1: #MGO
                    if np.abs(y_distance_LF) < 0.4:
                        results_string = results_string + str('\n\nWarning: sample could be Bunker A!')
                if y_pred_LF[0] ==4: #BA
                    if np.abs(y_distance_LF) < 0.4:
                        results_string = results_string + str('\n\nWarning: sample could be MGO!')  

                #Generate predicted saturate contents to display to user
                if y_sat_pred[0]==0: #Classified as Very Low Saturate
                    results_string = results_string + str('\n\nPredicted Saturate Level: Very Low (<34.9%)')
                elif y_sat_pred[0]==1: #Classified as Low Saturate
                    results_string = results_string + str('\n\nPredicted Saturate Level: Low (34.9-62.4%)')
                elif y_sat_pred[0]==2: #Classified as Medium Saturate
                    results_string = results_string + str('\n\nPredicted Saturate Level: Medium (62.4-71.2%)')
                elif y_sat_pred[0]==3: #Classified as High Saturate
                    results_string = results_string + str('\n\nPredicted Saturate Level: High (>71.2%)')
                else:
                    #If there's a typo in the training set
                    results_string = results_string + str('Error: Typos in training data (saturate) and class is not 0, 1, 2,or 3!')

                #Generate predicted aromatic contents to display to user
                if y_aro_pred[0]==0: #Classified as Very Low Aromatic
                    results_string = results_string + str('\n\nPredicted Aromatic Level: Very Low (<16.3%)')
                elif y_aro_pred[0]==1: #Classified as Low Aromatic
                    results_string = results_string + str('\n\nPredicted Aromatic Level: Low (16.3-25.7%)')
                elif y_aro_pred[0]==2: #Classified as Medium Aromatic
                    results_string = results_string + str('\n\nPredicted Aromatic Level: Medium (25.7-37%)')
                elif y_aro_pred[0]==3: #Classified as High Aromatic
                    results_string = results_string + str('\n\nPredicted Aromatic Level: High (>37%)')
                else:
                    #If there's a typo in the training set 
                    results_string = results_string + str('Error: Typos in training data (aromatic) and class is not 0, 1, 2,or 3!')

                #Generate predicted resin contents to display to user
                if y_res_pred[0]==0: #Classified as Very Low Resin
                    results_string = results_string + str('\n\nPredicted Resin Level: Very Low (<6.2%)')
                elif y_res_pred[0]==1: #Classified as Low Resin
                    results_string = results_string + str('\n\nPredicted Resin Level: Low (6.2-9.9%)')
                elif y_res_pred[0]==2: #Classified as Medium Resin
                    results_string = results_string + str('\n\nPredicted Resin Level: Medium (9.9-14%)')
                elif y_res_pred[0]==3: #Classified as High Resin
                    results_string = results_string + str('\n\nPredicted Resin Level: High (>14%)')
                else:
                    #If there's a typo in the training set 
                    results_string = results_string + str('Error: Typos in training data (resin) and class is not 0, 1, 2,or 3!')

                #Generate predicted asphaltene contents to display to user
                if y_asp_pred[0]==0: #Classified as Very Low Asphaltene
                    results_string = results_string + str('\n\nPredicted Asphaltene Level: Very Low (0%)')
                elif y_asp_pred[0]==1: #Classified as Low Asphaltene
                    results_string = results_string + str('\n\nPredicted Asphaltene Level: Low (0-0.8%)')
                elif y_asp_pred[0]==2: #Classified as Medium Asphaltene
                    results_string = results_string + str('\n\nPredicted Asphaltene Level: Medium (0.8-9.8%)')
                elif y_asp_pred[0]==3: #Classified as High Asphaltene
                    results_string = results_string + str('\n\nPredicted Asphaltene Level: High (>9.8%)')
                else:
                    #If there's a typo in the training set 
                    results_string = results_string + str('Error: Typos in training data (asphaltene) and class is not 0, 1, 2,or 3!')

            #Display results
            self.finished = tk.Message(self, text=results_string, font=SMALL_FONT)
            self.finished.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

            self.saveButton=tk.Button(self, text='Save', font=LARGE_FONT)
            self.saveButton['command'] = lambda: self.saveDataAndRestart()
            self.saveButton.grid(row=3, column=0, padx=5, pady=2)
            self.deleteButton=tk.Button(self, text='Delete', font=LARGE_FONT)
            self.deleteButton['command'] = lambda: self.deleteDataAndRestart()
            self.deleteButton.grid(row=3, column=1, pady=2)

            #SEND RESULTS TO PHONE OVER BLUETOOTH
            #Attempt to connect to phone
            try:
                phone = serial.Serial('/dev/rfcomm0', baudrate=9600)
                #Encode results string and send data to phone
                phone.write(results_string.encode('utf-8'))
                print('Data sent to phone')
            #Print warning in the event that the OTA cannot connect to phone
            except serial.SerialException:
                print('Could not connect to Phone')
                print('Be sure to allow the Pi to accept a bluetooth communication in a different terminal window,')
                print('then connect to the Pi on the smartphone app before running this program')

        except FileNotFoundError:
            #will occur if folder names are changed, or if an image cannot be generated by camera
            #in testing mode, this occurs if name entered is not an existing image in Data folder
            self.dataError = tk.Message(self,
                                        text= 'Error - please ensure all relevant files are located in the Data folder on this device. If that is not the issue, there may be a problem with the camera.',
                                        font=LARGE_FONT)
            self.dataError.grid(row=2, column=0, pady=2)
            self.okButton4=tk.Button(self, text='OK', font=LARGE_FONT)
            self.okButton4['command']=lambda:self.clearDataErrorAlert()
            self.okButton4.grid(row=3, column=0, pady=2)
        
    def saveDataAndRestart(self):
        #Delete items from screen
        self.finished.destroy()
        self.saveButton.destroy()
        self.deleteButton.destroy()
        #Open data file and append the initial (user-entered) data
        csvfile= open("/home/pi/Desktop/Data/oildata.csv", "a+")
        #Write collected data to csv file
        wr=csv.writer(csvfile)
        wr.writerow(self.data_to_append)
        #Reopen the ID counter and increment it
        IDfile = open('/home/pi/Desktop/Data/IDcounter.txt', 'w')
        IDfile.write(str(int(self.ID)+1))
        IDfile.close()
        #Allow user to start a new sample
        self.startAnalyzingSamplesButton.grid(row=1, column=0)
        
    def deleteDataAndRestart(self):
        deleteQuestion = messagebox.askyesno('Delete', 'Are you sure you want to delete the data?')
        if deleteQuestion==True:
            #Delete items from screen
            self.finished.destroy()
            self.saveButton.destroy()
            self.deleteButton.destroy()
            #Delete saved images
            os.remove('/home/pi/Desktop/Data/image_data/%s_365.jpg' %self.ID) #365nm image taken
            os.remove('/home/pi/Desktop/Data/image_data/%s_375.jpg' %self.ID) #375nm image taken
            os.remove('/home/pi/Desktop/Data/image_data/%s_385.jpg' %self.ID) #385nm image taken
            #Allow user to start a new sample
            self.startAnalyzingSamplesButton.grid(row=1, column=0, padx=5)

    def clearDataErrorAlert(self):
        self.dataError.destroy()
        self.okButton4.destroy()
        self.startButton.config(state=NORMAL)
    
if __name__ == "__main__":
    app = CrudeOilGUI()
    #app.attributes('-fullscreen', True)
    def on_closing(): #Will only enter this function if fullscreen attribute is commented out and user closes Tk window
        app.destroy()
        GPIO.cleanup()
    def quit(event):
        app.destroy()
        GPIO.cleanup()
    app.protocol("WM_DELETE_WINDOW", on_closing) #Only accessible if fullscreen attribute is commented out
    app.bind('<Control-c>', quit)
    app.mainloop()
    
    
