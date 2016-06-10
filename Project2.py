# Computer Vision and Scene Analysis Project 2
# Name          : RUPANTA RWITEEJ DUTTA
# University ID : N15786532
# Creation Date : Tue Dec 22 2:42:51 2015 Eastern Daylight Time

from PIL import Image
import os
import glob
import math
import numpy
import warnings

warnings.filterwarnings("ignore")                                                                           #Ignore Console Warnings(if any)

print ("\nInterpreting Project2.py...")
print ("\n===========================================================================")
print ("                          HOG Feature Descriptor")
print ("===========================================================================")

# Delete all Previously Created Output Files
print ("Removing Previously Generated Output Files...")
filePath = []
filePath.append("ComputedResults/HOGDescriptors/TrainingSet/PositiveSamples/*")
filePath.append("ComputedResults/HOGDescriptors/TrainingSet/NegativeSamples/*")
filePath.append("ComputedResults/HOGDescriptors/TestingSet/PositiveSamples/*")
filePath.append("ComputedResults/HOGDescriptors/TestingSet/NegativeSamples/*")
filePath.append("ComputedResults/MeanDescriptors/PositiveSamples/*")
filePath.append("ComputedResults/MeanDescriptors/NegativeSamples/*")
filePath.append("ComputedResults/EucleidianDistances/PositiveSamples/*")
filePath.append("ComputedResults/EucleidianDistances/NegativeSamples/*")
filePath.append("ComputedResults/wValue/*")
filePath.append("ComputedResults/*.txt")
for i in filePath:
    files = glob.glob(i)
    for f in files:
        os.remove(f)

# Create List of Positive and Negative Training & Test Samples
print ("Taking Input Image Files...")
trainingPositiveList = glob.glob("TrainingSet/PositiveSamples/*.bmp")                                       #List to store Positive Training Samples
trainingNegativeList = glob.glob("TrainingSet/NegativeSamples/*.bmp")                                       #List to store Negative Training Samples

testList1 = glob.glob("TestingSet/PositiveSamples/*.bmp")                                                   #List to store Test Samples I
testList2 = glob.glob("TestingSet/NegativeSamples/*.bmp")                                                   #List to store Test Samples II

# Declare Arrays to store Gradient Orientation Histograms
positiveSamplesHOG = numpy.zeros(shape=(10, 3780, 1))                                                       #Array to store final HOG Descriptor for Positive Training Samples
positiveMeanDescriptor = numpy.zeros(shape=(1, 3780, 1))                                                    #Array to store final Mean Descriptor for Positive Training Samples
positiveSamplesDistance = numpy.zeros(shape=(10, 3780, 1))                                                  #Array to store Distance between Positive Mean HOG Descriptor and each Positive Training Sample

negativeSamplesHOG = numpy.zeros(shape=(10, 3780, 1))                                                       #Array to store final HOG Descriptor for Negative Training Samples
negativeMeanDescriptor = numpy.zeros(shape=(1, 3780, 1))                                                    #Array to store final Mean Descriptor for Negative Training Samples
negativeSamplesDistance = numpy.zeros(shape=(10, 3780, 1))                                                  #Array to store Distance between Negative Mean HOG Descriptor and each Negative Training Sample

testSamples1HOG = numpy.zeros(shape=(10, 3780, 1))                                                          #Array to store final HOG Descriptor for Test Samples I
testSamples2HOG = numpy.zeros(shape=(10, 3780, 1))                                                          #Array to store final HOG Descriptor for Test Samples II

wValue = numpy.zeros(shape=(1, 3781, 1))                                                                    #Array to store Initial W-Value
wFinalValue = numpy.zeros(shape=(1, 3781, 1))                                                               #Array to store Final W-Value
alpha = 0.5                                                                                                 #Alpha is set to a Fixed Value of 0.5

# Function to Convert Input Image to Array
def convertToArray(img):
    return numpy.array(img.getdata(),numpy.uint8).reshape(img.size[1], img.size[0], 3)

# Read Images into Arrays and Convert into GrayScale
print("Converting Images to Gray Scale...")
trainingPositiveSamples = numpy.zeros(shape=(10, 160, 96))                                                  #Array to store Positive Training Samples
trainingNegativeSamples = numpy.zeros(shape=(10, 160, 96))                                                  #Array to store Negative Training Samples

testSamples1 = numpy.zeros(shape=(5, 160, 96))                                                              #Array to store Test Samples I
testSamples2 = numpy.zeros(shape=(5, 160, 96))                                                              #Array to store Test Samples II

for index in range(0, 10):
    current = Image.open(trainingPositiveList[index])
    imgArray = convertToArray(current)
    for i in range(0, 160):
        for j in range(0, 96):
            trainingPositiveSamples[index][i][j] = (imgArray[i][j][0] * 0.21) + (imgArray[i][j][1] * 0.72) + (imgArray[i][j][2] * 0.07)
            trainingPositiveSamples[index][i][j] = int(trainingPositiveSamples[index][i][j])

for index in range(0, 10):
    current = Image.open(trainingNegativeList[index])
    imgArray = convertToArray(current)
    for i in range(0, 160):
        for j in range(0, 96):
            trainingNegativeSamples[index][i][j] = (imgArray[i][j][0] * 0.21) + (imgArray[i][j][1] * 0.72) + (imgArray[i][j][2] * 0.07)
            trainingNegativeSamples[index][i][j] = int(trainingNegativeSamples[index][i][j])

for index in range(0, 5):
    current = Image.open(testList1[index])
    imgArray = convertToArray(current)
    for i in range(0, 160):
        for j in range(0, 96):
            testSamples1[index][i][j] = (imgArray[i][j][0] * 0.21) + (imgArray[i][j][1] * 0.72) + (imgArray[i][j][2] * 0.07)
            testSamples1[index][i][j] = int(testSamples1[index][i][j])


for index in range(0, 5):
    current = Image.open(testList2[index])
    imgArray = convertToArray(current)
    for i in range(0, 160):
        for j in range(0, 96):
            testSamples2[index][i][j] = (imgArray[i][j][0] * 0.21) + (imgArray[i][j][1] * 0.72) + (imgArray[i][j][2] * 0.07)
            testSamples2[index][i][j] = int(testSamples2[index][i][j])

# Calculate Horizontal & Vertical Gradients
print("Computing Vertical and Horizontal Gradients...")
trainingPositiveHorizontalGradient = numpy.zeros(shape=(10, 160, 96))
trainingPositiveVerticalGradient = numpy.zeros(shape=(10, 160, 96))

trainingNegativeHorizontalGradient = numpy.zeros(shape=(10, 160, 96))
trainingNegativeVerticalGradient = numpy.zeros(shape=(10, 160, 96))

testSamplesHorizontalGradient1 = numpy.zeros(shape=(5, 160, 96))
testSamplesVerticalGradient1 = numpy.zeros(shape=(5, 160, 96))

testSamplesHorizontalGradient2 = numpy.zeros(shape=(5, 160, 96))
testSamplesVerticalGradient2 = numpy.zeros(shape=(5, 160, 96))

for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 94):
            trainingPositiveHorizontalGradient[index][i][j] = (trainingPositiveSamples[index][i][j-1] * -1) + (trainingPositiveSamples[index][i][j] * 0) + (trainingPositiveSamples[index][i][j+1] * 1)
            trainingNegativeHorizontalGradient[index][i][j] = (trainingNegativeSamples[index][i][j-1] * -1) + (trainingNegativeSamples[index][i][j] * 0) + (trainingNegativeSamples[index][i][j+1] * 1)

for index in range(0, 10):
    for i in range(0, 158):
        for j in range(0, 96):
            trainingPositiveVerticalGradient[index][i][j] = (trainingPositiveSamples[index][i-1][j] * -1) + (trainingPositiveSamples[index][i][j] * 0) + (trainingPositiveSamples[index][i+1][j] * 1)
            trainingNegativeVerticalGradient[index][i][j] = (trainingNegativeSamples[index][i-1][j] * -1) + (trainingNegativeSamples[index][i][j] * 0) + (trainingNegativeSamples[index][i+1][j] * 1)

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 94):
            testSamplesHorizontalGradient1[index][i][j] = (testSamples1[index][i][j-1] * -1) + (testSamples1[index][i][j] * 0) + (testSamples1[index][i][j+1] * 1)
            testSamplesHorizontalGradient2[index][i][j] = (testSamples2[index][i][j-1] * -1) + (testSamples2[index][i][j] * 0) + (testSamples2[index][i][j+1] * 1)

for index in range(0, 5):
    for i in range(0, 158):
        for j in range(0, 96):
            testSamplesVerticalGradient1[index][i][j] = (testSamples1[index][i-1][j] * -1) + (testSamples1[index][i][j] * 0) + (testSamples1[index][i+1][j] * 1)
            testSamplesVerticalGradient2[index][i][j] = (testSamples2[index][i-1][j] * -1) + (testSamples2[index][i][j] * 0) + (testSamples2[index][i+1][j] * 1)

# Calculate Edge Magnitude and Gradient Angles
print("Computing Edge Magnitudes & Gradient Angles...")
trainingPositiveGradientMagnitude = numpy.zeros(shape=(10, 160, 96))
trainingPositiveGradientAngle = numpy.zeros(shape=(10, 160, 96))

trainingNegativeGradientMagnitude = numpy.zeros(shape=(10, 160, 96))
trainingNegativeGradientAngle = numpy.zeros(shape=(10, 160, 96))

testSamplesGradientMagnitude1 = numpy.zeros(shape=(5, 160, 96))
testSamplesGradientAngle1 = numpy.zeros(shape=(5, 160, 96))

testSamplesGradientMagnitude2 = numpy.zeros(shape=(5, 160, 96))
testSamplesGradientAngle2 = numpy.zeros(shape=(5, 160, 96))

for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 96):
            if trainingPositiveHorizontalGradient[index][i][j] == 0 and trainingPositiveVerticalGradient[index][i][j] == 0:
                trainingPositiveGradientMagnitude[index][i][j] = 0
                trainingPositiveGradientAngle[index][i][j] = None
            elif trainingPositiveHorizontalGradient[index][i][j] == 0 and trainingPositiveVerticalGradient[index][i][j] < 0:
                trainingPositiveGradientMagnitude[index][i][j] = int((((trainingPositiveHorizontalGradient[index][i][j]**2 + trainingPositiveVerticalGradient[index][i][j]**2)/2)**0.5))
                trainingPositiveGradientAngle[index][i][j] = -90
            elif trainingPositiveHorizontalGradient[index][i][j] == 0 and trainingPositiveVerticalGradient[index][i][j] > 0:
                trainingPositiveGradientMagnitude[index][i][j] = int((((trainingPositiveHorizontalGradient[index][i][j]**2 + trainingPositiveVerticalGradient[index][i][j]**2)/2)**0.5))
                trainingPositiveGradientAngle[index][i][j] = 90
            else:
                trainingPositiveGradientMagnitude[index][i][j] = int((((trainingPositiveHorizontalGradient[index][i][j]**2 + trainingPositiveVerticalGradient[index][i][j]**2)/2)**0.5))
                trainingPositiveGradientAngle[index][i][j] = float(math.degrees(math.atan(trainingPositiveVerticalGradient[index][i][j]/trainingPositiveHorizontalGradient[index][i][j])))

for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 96):
            if trainingNegativeHorizontalGradient[index][i][j] == 0 and trainingNegativeVerticalGradient[index][i][j] == 0:
                trainingNegativeGradientMagnitude[index][i][j] = 0
                trainingNegativeGradientAngle[index][i][j] = None
            elif trainingNegativeHorizontalGradient[index][i][j] == 0 and trainingNegativeVerticalGradient[index][i][j] < 0:
                trainingNegativeGradientMagnitude[index][i][j] = int((((trainingNegativeHorizontalGradient[index][i][j]**2 + trainingNegativeVerticalGradient[index][i][j]**2)/2)**0.5))
                trainingNegativeGradientAngle[index][i][j] = -90
            elif trainingNegativeHorizontalGradient[index][i][j] == 0 and trainingNegativeVerticalGradient[index][i][j] > 0:
                trainingNegativeGradientMagnitude[index][i][j] = int((((trainingNegativeHorizontalGradient[index][i][j]**2 + trainingNegativeVerticalGradient[index][i][j]**2)/2)**0.5))
                trainingNegativeGradientAngle[index][i][j] = 90
            else:
                trainingNegativeGradientMagnitude[index][i][j] = int((((trainingNegativeHorizontalGradient[index][i][j]**2 + trainingNegativeVerticalGradient[index][i][j]**2)/2)**0.5))
                trainingNegativeGradientAngle[index][i][j] = float(math.degrees(math.atan(trainingNegativeVerticalGradient[index][i][j]/trainingNegativeHorizontalGradient[index][i][j])))

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 96):
            if testSamplesHorizontalGradient1[index][i][j] == 0 and testSamplesVerticalGradient1[index][i][j] == 0:
                testSamplesGradientMagnitude1[index][i][j] = 0
                testSamplesGradientAngle1[index][i][j] = None
            elif testSamplesHorizontalGradient1[index][i][j] == 0 and testSamplesVerticalGradient1[index][i][j] < 0:
                testSamplesGradientMagnitude1[index][i][j] = int((((testSamplesHorizontalGradient1[index][i][j]**2 + testSamplesVerticalGradient1[index][i][j]**2)/2)**0.5))
                testSamplesGradientAngle1[index][i][j] = -90
            elif testSamplesHorizontalGradient1[index][i][j] == 0 and testSamplesVerticalGradient1[index][i][j] > 0:
                testSamplesGradientMagnitude1[index][i][j] = int((((testSamplesHorizontalGradient1[index][i][j]**2 + testSamplesVerticalGradient1[index][i][j]**2)/2)**0.5))
                testSamplesGradientAngle1[index][i][j] = 90
            else:
                testSamplesGradientMagnitude1[index][i][j] = int((((testSamplesHorizontalGradient1[index][i][j]**2 + testSamplesVerticalGradient1[index][i][j]**2)/2)**0.5))
                testSamplesGradientAngle1[index][i][j] = float(math.degrees(math.atan(testSamplesVerticalGradient1[index][i][j]/testSamplesHorizontalGradient1[index][i][j])))

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 96):
            if testSamplesHorizontalGradient2[index][i][j] == 0 and testSamplesVerticalGradient2[index][i][j] == 0:
                testSamplesGradientMagnitude2[index][i][j] = 0
                testSamplesGradientAngle2[index][i][j] = None
            elif testSamplesHorizontalGradient2[index][i][j] == 0 and testSamplesVerticalGradient2[index][i][j] < 0:
                testSamplesGradientMagnitude2[index][i][j] = int((((testSamplesHorizontalGradient2[index][i][j]**2 + testSamplesVerticalGradient2[index][i][j]**2)/2)**0.5))
                testSamplesGradientAngle2[index][i][j] = -90
            elif testSamplesHorizontalGradient2[index][i][j] == 0 and testSamplesVerticalGradient2[index][i][j] > 0:
                testSamplesGradientMagnitude2[index][i][j] = int((((testSamplesHorizontalGradient2[index][i][j]**2 + testSamplesVerticalGradient2[index][i][j]**2)/2)**0.5))
                testSamplesGradientAngle2[index][i][j] = 90
            else:
                testSamplesGradientMagnitude2[index][i][j] = int((((testSamplesHorizontalGradient2[index][i][j]**2 + testSamplesVerticalGradient2[index][i][j]**2)/2)**0.5))
                testSamplesGradientAngle2[index][i][j] = float(math.degrees(math.atan(testSamplesVerticalGradient2[index][i][j]/testSamplesHorizontalGradient2[index][i][j])))

# Calculate Negative of Gradient Angles
for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 96):
            trainingPositiveGradientAngle[index][i][j] = trainingPositiveGradientAngle[index][i][j] * -1
            if trainingPositiveGradientAngle[index][i][j] < 0:
                trainingPositiveGradientAngle[index][i][j] = trainingPositiveGradientAngle[index][i][j] + 360
            trainingNegativeGradientAngle[index][i][j] = trainingNegativeGradientAngle[index][i][j] * -1
            if trainingNegativeGradientAngle[index][i][j] < 0:
                trainingNegativeGradientAngle[index][i][j] = trainingNegativeGradientAngle[index][i][j] + 360

for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 96):
            if trainingPositiveGradientAngle[index][i][j] >= 180:
                trainingPositiveGradientAngle[index][i][j] = trainingPositiveGradientAngle[index][i][j] - 180
            if trainingNegativeGradientAngle[index][i][j] >= 180:
                trainingNegativeGradientAngle[index][i][j] = trainingNegativeGradientAngle[index][i][j] - 180

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 96):
            testSamplesGradientAngle1[index][i][j] = testSamplesGradientAngle1[index][i][j] * -1
            if testSamplesGradientAngle1[index][i][j] < 0:
                testSamplesGradientAngle1[index][i][j] = testSamplesGradientAngle1[index][i][j] + 360

            testSamplesGradientAngle2[index][i][j] = testSamplesGradientAngle2[index][i][j] * -1
            if testSamplesGradientAngle2[index][i][j] < 0:
                testSamplesGradientAngle2[index][i][j] = testSamplesGradientAngle2[index][i][j] + 360

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 96):
            if testSamplesGradientAngle1[index][i][j] >= 180:
                testSamplesGradientAngle1[index][i][j] = testSamplesGradientAngle1[index][i][j] - 180

            if testSamplesGradientAngle2[index][i][j] >= 180:
                testSamplesGradientAngle2[index][i][j] = testSamplesGradientAngle2[index][i][j] - 180

# Set Window Size of 128x64 Pixels
print("Setting Window Size to 128x64 Pixels...")
for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 96):
            if (i >= 0 and i < 16) or (i > 143 and i <= 160) or (j >= 0 and j < 16) or (j > 79 and j <= 96):
                trainingPositiveGradientMagnitude[index][i][j] = None
                trainingPositiveGradientAngle[index][i][j] = None
                trainingNegativeGradientMagnitude[index][i][j] = None
                trainingNegativeGradientAngle[index][i][j] = None

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 96):
            if (i >= 0 and i < 16) or (i > 143 and i <= 160) or (j >= 0 and j < 16) or (j > 79 and j <= 96):
                testSamplesGradientMagnitude1[index][i][j] = None
                testSamplesGradientAngle1[index][i][j] = None
                testSamplesGradientMagnitude2[index][i][j] = None
                testSamplesGradientAngle2[index][i][j] = None

# Quantize Gradient Angles into 9 Bins
print("Quantizing Gradient Angles into 9 Bins(Unsigned)...")
trainingPositiveQuantizedAngle = numpy.zeros(shape=(10, 160, 96))
trainingNegativeQuantizedAngle = numpy.zeros(shape=(10, 160, 96))

testQuantizedAngle1 = numpy.zeros(shape=(5, 160, 96))
testQuantizedAngle2 = numpy.zeros(shape=(5, 160, 96))

for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 96):
            if (trainingPositiveGradientAngle[index][i][j] >= 0 and trainingPositiveGradientAngle[index][i][j] < 20) or (trainingPositiveGradientAngle[index][i][j] >= 180 and trainingPositiveGradientAngle[index][i][j] < 200):
                trainingPositiveQuantizedAngle[index][i][j] = 10
            elif (trainingPositiveGradientAngle[index][i][j] >= 20 and trainingPositiveGradientAngle[index][i][j] < 40) or (trainingPositiveGradientAngle[index][i][j] >= 200 and trainingPositiveGradientAngle[index][i][j] < 220):
                trainingPositiveQuantizedAngle[index][i][j] = 30
            elif (trainingPositiveGradientAngle[index][i][j] >= 40 and trainingPositiveGradientAngle[index][i][j] < 60) or (trainingPositiveGradientAngle[index][i][j] >= 220 and trainingPositiveGradientAngle[index][i][j] < 240):
                trainingPositiveQuantizedAngle[index][i][j] = 50
            elif (trainingPositiveGradientAngle[index][i][j] >= 60 and trainingPositiveGradientAngle[index][i][j] < 80) or (trainingPositiveGradientAngle[index][i][j] >= 240 and trainingPositiveGradientAngle[index][i][j] < 260):
                trainingPositiveQuantizedAngle[index][i][j] = 70
            elif (trainingPositiveGradientAngle[index][i][j] >= 80 and trainingPositiveGradientAngle[index][i][j] < 100) or (trainingPositiveGradientAngle[index][i][j] >= 260 and trainingPositiveGradientAngle[index][i][j] < 280):
                trainingPositiveQuantizedAngle[index][i][j] = 90
            elif (trainingPositiveGradientAngle[index][i][j] >= 100 and trainingPositiveGradientAngle[index][i][j] < 120) or (trainingPositiveGradientAngle[index][i][j] >= 280 and trainingPositiveGradientAngle[index][i][j] < 300):
                trainingPositiveQuantizedAngle[index][i][j] = 110
            elif (trainingPositiveGradientAngle[index][i][j] >= 120 and trainingPositiveGradientAngle[index][i][j] < 140) or (trainingPositiveGradientAngle[index][i][j] >= 300 and trainingPositiveGradientAngle[index][i][j] < 320):
                trainingPositiveQuantizedAngle[index][i][j] = 130
            elif (trainingPositiveGradientAngle[index][i][j] >= 140 and trainingPositiveGradientAngle[index][i][j] < 160) or (trainingPositiveGradientAngle[index][i][j] >= 320 and trainingPositiveGradientAngle[index][i][j] < 340):
                trainingPositiveQuantizedAngle[index][i][j] = 150
            elif (trainingPositiveGradientAngle[index][i][j] >= 160 and trainingPositiveGradientAngle[index][i][j] < 180) or (trainingPositiveGradientAngle[index][i][j] >= 340 and trainingPositiveGradientAngle[index][i][j] < 360):
                trainingPositiveQuantizedAngle[index][i][j] = 170
            else:
                trainingPositiveQuantizedAngle[index][i][j] = None

for index in range(0, 10):
    for i in range(0, 160):
        for j in range(0, 96):
            if (trainingNegativeGradientAngle[index][i][j] >= 0 and trainingNegativeGradientAngle[index][i][j] < 20) or (trainingNegativeGradientAngle[index][i][j] >= 180 and trainingNegativeGradientAngle[index][i][j] < 200):
                trainingNegativeQuantizedAngle[index][i][j] = 10
            elif (trainingNegativeGradientAngle[index][i][j] >= 20 and trainingNegativeGradientAngle[index][i][j] < 40) or (trainingNegativeGradientAngle[index][i][j] >= 200 and trainingNegativeGradientAngle[index][i][j] < 220):
                trainingNegativeQuantizedAngle[index][i][j] = 30
            elif (trainingNegativeGradientAngle[index][i][j] >= 40 and trainingNegativeGradientAngle[index][i][j] < 60) or (trainingNegativeGradientAngle[index][i][j] >= 220 and trainingNegativeGradientAngle[index][i][j] < 240):
                trainingNegativeQuantizedAngle[index][i][j] = 50
            elif (trainingNegativeGradientAngle[index][i][j] >= 60 and trainingNegativeGradientAngle[index][i][j] < 80) or (trainingNegativeGradientAngle[index][i][j] >= 240 and trainingNegativeGradientAngle[index][i][j] < 260):
                trainingNegativeQuantizedAngle[index][i][j] = 70
            elif (trainingNegativeGradientAngle[index][i][j] >= 80 and trainingNegativeGradientAngle[index][i][j] < 100) or (trainingNegativeGradientAngle[index][i][j] >= 260 and trainingNegativeGradientAngle[index][i][j] < 280):
                trainingNegativeQuantizedAngle[index][i][j] = 90
            elif (trainingNegativeGradientAngle[index][i][j] >= 100 and trainingNegativeGradientAngle[index][i][j] < 120) or (trainingNegativeGradientAngle[index][i][j] >= 280 and trainingNegativeGradientAngle[index][i][j] < 300):
                trainingNegativeQuantizedAngle[index][i][j] = 110
            elif (trainingNegativeGradientAngle[index][i][j] >= 120 and trainingNegativeGradientAngle[index][i][j] < 140) or (trainingNegativeGradientAngle[index][i][j] >= 300 and trainingNegativeGradientAngle[index][i][j] < 320):
                trainingNegativeQuantizedAngle[index][i][j] = 130
            elif (trainingNegativeGradientAngle[index][i][j] >= 140 and trainingNegativeGradientAngle[index][i][j] < 160) or (trainingNegativeGradientAngle[index][i][j] >= 320 and trainingNegativeGradientAngle[index][i][j] < 340):
                trainingNegativeQuantizedAngle[index][i][j] = 150
            elif (trainingNegativeGradientAngle[index][i][j] >= 160 and trainingNegativeGradientAngle[index][i][j] < 180) or (trainingNegativeGradientAngle[index][i][j] >= 340 and trainingNegativeGradientAngle[index][i][j] < 360):
                trainingNegativeQuantizedAngle[index][i][j] = 170
            else:
                trainingNegativeQuantizedAngle[index][i][j] = None

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 96):
            if (testSamplesGradientAngle1[index][i][j] >= 0 and testSamplesGradientAngle1[index][i][j] < 20) or (testSamplesGradientAngle1[index][i][j] >= 180 and testSamplesGradientAngle1[index][i][j] < 200):
                testQuantizedAngle1[index][i][j] = 10
            elif (testSamplesGradientAngle1[index][i][j] >= 20 and testSamplesGradientAngle1[index][i][j] < 40) or (testSamplesGradientAngle1[index][i][j] >= 200 and testSamplesGradientAngle1[index][i][j] < 220):
                testQuantizedAngle1[index][i][j] = 30
            elif (testSamplesGradientAngle1[index][i][j] >= 40 and testSamplesGradientAngle1[index][i][j] < 60) or (testSamplesGradientAngle1[index][i][j] >= 220 and testSamplesGradientAngle1[index][i][j] < 240):
                testQuantizedAngle1[index][i][j] = 50
            elif (testSamplesGradientAngle1[index][i][j] >= 60 and testSamplesGradientAngle1[index][i][j] < 80) or (testSamplesGradientAngle1[index][i][j] >= 240 and testSamplesGradientAngle1[index][i][j] < 260):
                testQuantizedAngle1[index][i][j] = 70
            elif (testSamplesGradientAngle1[index][i][j] >= 80 and testSamplesGradientAngle1[index][i][j] < 100) or (testSamplesGradientAngle1[index][i][j] >= 260 and testSamplesGradientAngle1[index][i][j] < 280):
                testQuantizedAngle1[index][i][j] = 90
            elif (testSamplesGradientAngle1[index][i][j] >= 100 and testSamplesGradientAngle1[index][i][j] < 120) or (testSamplesGradientAngle1[index][i][j] >= 280 and testSamplesGradientAngle1[index][i][j] < 300):
                testQuantizedAngle1[index][i][j] = 110
            elif (testSamplesGradientAngle1[index][i][j] >= 120 and testSamplesGradientAngle1[index][i][j] < 140) or (testSamplesGradientAngle1[index][i][j] >= 300 and testSamplesGradientAngle1[index][i][j] < 320):
                testQuantizedAngle1[index][i][j] = 130
            elif (testSamplesGradientAngle1[index][i][j] >= 140 and testSamplesGradientAngle1[index][i][j] < 160) or (testSamplesGradientAngle1[index][i][j] >= 320 and testSamplesGradientAngle1[index][i][j] < 340):
                testQuantizedAngle1[index][i][j] = 150
            elif (testSamplesGradientAngle1[index][i][j] >= 160 and testSamplesGradientAngle1[index][i][j] < 180) or (testSamplesGradientAngle1[index][i][j] >= 340 and testSamplesGradientAngle1[index][i][j] < 360):
                testQuantizedAngle1[index][i][j] = 170
            else:
                testQuantizedAngle1[index][i][j] = None

for index in range(0, 5):
    for i in range(0, 160):
        for j in range(0, 96):
            if (testSamplesGradientAngle2[index][i][j] >= 0 and testSamplesGradientAngle2[index][i][j] < 20) or (testSamplesGradientAngle2[index][i][j] >= 180 and testSamplesGradientAngle2[index][i][j] < 200):
                testQuantizedAngle2[index][i][j] = 10
            elif (testSamplesGradientAngle2[index][i][j] >= 20 and testSamplesGradientAngle2[index][i][j] < 40) or (testSamplesGradientAngle2[index][i][j] >= 200 and testSamplesGradientAngle2[index][i][j] < 220):
                testQuantizedAngle2[index][i][j] = 30
            elif (testSamplesGradientAngle2[index][i][j] >= 40 and testSamplesGradientAngle2[index][i][j] < 60) or (testSamplesGradientAngle2[index][i][j] >= 220 and testSamplesGradientAngle2[index][i][j] < 240):
                testQuantizedAngle2[index][i][j] = 50
            elif (testSamplesGradientAngle2[index][i][j] >= 60 and testSamplesGradientAngle2[index][i][j] < 80) or (testSamplesGradientAngle2[index][i][j] >= 240 and testSamplesGradientAngle2[index][i][j] < 260):
                testQuantizedAngle2[index][i][j] = 70
            elif (testSamplesGradientAngle2[index][i][j] >= 80 and testSamplesGradientAngle2[index][i][j] < 100) or (testSamplesGradientAngle2[index][i][j] >= 260 and testSamplesGradientAngle2[index][i][j] < 280):
                testQuantizedAngle2[index][i][j] = 90
            elif (testSamplesGradientAngle2[index][i][j] >= 100 and testSamplesGradientAngle2[index][i][j] < 120) or (testSamplesGradientAngle2[index][i][j] >= 280 and testSamplesGradientAngle2[index][i][j] < 300):
                testQuantizedAngle2[index][i][j] = 110
            elif (testSamplesGradientAngle2[index][i][j] >= 120 and testSamplesGradientAngle2[index][i][j] < 140) or (testSamplesGradientAngle2[index][i][j] >= 300 and testSamplesGradientAngle2[index][i][j] < 320):
                testQuantizedAngle2[index][i][j] = 130
            elif (testSamplesGradientAngle2[index][i][j] >= 140 and testSamplesGradientAngle2[index][i][j] < 160) or (testSamplesGradientAngle2[index][i][j] >= 320 and testSamplesGradientAngle2[index][i][j] < 340):
                testQuantizedAngle2[index][i][j] = 150
            elif (testSamplesGradientAngle2[index][i][j] >= 160 and testSamplesGradientAngle2[index][i][j] < 180) or (testSamplesGradientAngle2[index][i][j] >= 340 and testSamplesGradientAngle2[index][i][j] < 360):
                testQuantizedAngle2[index][i][j] = 170
            else:
                testQuantizedAngle2[index][i][j] = None

# Calculate Normalized Gradient Orientation Histograms for Positive Training Images
print("Computing Gradient Orientation Histogram for Positive Training Images...")
for index in range(0, 10):
    count = 0
    for i in xrange(16, 129, 8):                                                                            #For Each Block
        for j in xrange(16, 65, 8):
            h1 = 0.0
            h2 = 0.0
            h3 = 0.0
            h4 = 0.0
            h5 = 0.0
            h6 = 0.0
            h7 = 0.0
            h8 = 0.0
            h9 = 0.0
            h12 = 0.0
            h22 = 0.0
            h32 = 0.0
            h42 = 0.0
            h52 = 0.0
            h62 = 0.0
            h72 = 0.0
            h82 = 0.0
            h92 = 0.0
            h13 = 0.0
            h23 = 0.0
            h33 = 0.0
            h43 = 0.0
            h53 = 0.0
            h63 = 0.0
            h73 = 0.0
            h83 = 0.0
            h93 = 0.0
            h14 = 0.0
            h24 = 0.0
            h34 = 0.0
            h44 = 0.0
            h54 = 0.0
            h64 = 0.0
            h74 = 0.0
            h84 = 0.0
            h94 = 0.0
            h = 0.0
            for k in range(i, i+8):                                                                         #For Cell 1
                for l in range(j, j+8):
                    if (trainingPositiveQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h1 = h1 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h2 = h2 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h9 = h9 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h2 = h2 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h3 = h3 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h1 = h1 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h3 = h3 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h4 = h4 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h2 = h2 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h4 = h4 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h5 = h5 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h3 = h3 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h5 = h5 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h6 = h6 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h4 = h4 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h6 = h6 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h7 = h7 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h5 = h5 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h7 = h7 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h8 = h8 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h6 = h6 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h8 = h8 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h9 = h9 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h7 = h7 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h9 = h9 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h1 = h1 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h8 = h8 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                                
            for k in range(i, i+8):                                                                         #For Cell 2
                for l in range(j+8, j+16):
                    if (trainingPositiveQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h12 = h12 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h22 = h22 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h92 = h92 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h22 = h22 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h32 = h32 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h12 = h12 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h32 = h32 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h42 = h42 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h22 = h22 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h42 = h42 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h52 = h52 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h32 = h32 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h52 = h52 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h62 = h62 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h42 = h42 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h62 = h62 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h72 = h72 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h52 = h52 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h72 = h72 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h82 = h82 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h62 = h62 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h82 = h82 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h92 = h92 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h72 = h72 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h92 = h92 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h12 = h12 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h82 = h82 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                
            for k in range(i+8, i+16):                                                                      #For Cell 3
                for l in range(j, j+8):
                    if (trainingPositiveQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h13 = h13 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h23 = h23 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h93 = h93 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h23 = h23 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h33 = h33 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h13 = h13 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h33 = h33 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h43 = h43 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h23 = h23 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h43 = h43 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h53 = h53 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h33 = h33 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h53 = h53 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h63 = h63 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h43 = h43 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h63 = h63 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h73 = h73 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h53 = h53 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h73 = h73 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h83 = h83 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h63 = h63 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h83 = h83 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h93 = h93 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h73 = h73 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h93 = h93 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h13 = h13 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h83 = h83 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])

            for k in range(i+8, i+16):                                                                      #For Cell 4
                for l in range(j+8, j+16):
                    if (trainingPositiveQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h14 = h14 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h24 = h24 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h94 = h94 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h24 = h24 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h34 = h34 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h14 = h14 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h34 = h34 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h44 = h44 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h24 = h24 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h4 = h4 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h54 = h54 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h34 = h34 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h54 = h54 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h64 = h64 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h44 = h44 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h64 = h64 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h74 = h74 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h54 = h54 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h74 = h74 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h84 = h84 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h64 = h64 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h84 = h84 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h94 = h94 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h74 = h74 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                    if (trainingPositiveQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingPositiveGradientAngle[index][k][l]-trainingPositiveQuantizedAngle[index][k][l])/20)
                        h94 = h94 + (proportion * trainingPositiveGradientMagnitude[index][k][l])
                        if (trainingPositiveGradientAngle[index][k][l] > trainingPositiveQuantizedAngle[index][k][l]):
                            h14 = h14 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
                        else:
                            h84 = h84 + ((1 - proportion) * trainingPositiveGradientMagnitude[index][k][l])
            h = ((h1*h1)+(h2*h2)+(h3*h3)+(h4*h4)+(h5*h5)+(h6*h6)+(h7*h7)+(h8*h8)+(h9*h9)+(h12*h12)+(h22*h22)+(h32*h32)+(h42*h42)+(h52*h52)+(h62*h62)+(h72*h72)+(h82*h82)+(h92*h92)+(h13*h13)+(h23*h23)+(h33*h33)+(h43*h43)+(h53*h53)+(h63*h63)+(h73*h73)+(h83*h83)+(h93*h93)+(h14*h14)+(h24*h24)+(h34*h34)+(h44*h44)+(h54*h54)+(h64*h64)+(h74*h74)+(h84*h84)+(h94*h94))**0.5
            if h != 0.0:
                h1 = '{0:.2f}'.format(h1 / h)
                h2 = '{0:.2f}'.format(h2 / h)
                h3 = '{0:.2f}'.format(h3 / h)
                h4 = '{0:.2f}'.format(h4 / h)
                h5 = '{0:.2f}'.format(h5 / h)
                h6 = '{0:.2f}'.format(h6 / h)
                h7 = '{0:.2f}'.format(h7 / h)
                h8 = '{0:.2f}'.format(h8 / h)
                h9 = '{0:.2f}'.format(h9 / h)
                h12 = '{0:.2f}'.format(h12 / h)
                h22 = '{0:.2f}'.format(h22 / h)
                h32 = '{0:.2f}'.format(h32 / h)
                h42 = '{0:.2f}'.format(h42 / h)
                h52 = '{0:.2f}'.format(h52 / h)
                h62 = '{0:.2f}'.format(h62 / h)
                h72 = '{0:.2f}'.format(h72 / h)
                h82 = '{0:.2f}'.format(h82 / h)
                h92 = '{0:.2f}'.format(h92 / h)
                h13 = '{0:.2f}'.format(h13 / h)
                h23 = '{0:.2f}'.format(h23 / h)
                h33 = '{0:.2f}'.format(h33 / h)
                h43 = '{0:.2f}'.format(h43 / h)
                h53 = '{0:.2f}'.format(h53 / h)
                h63 = '{0:.2f}'.format(h63 / h)
                h73 = '{0:.2f}'.format(h73 / h)
                h83 = '{0:.2f}'.format(h83 / h)
                h93 = '{0:.2f}'.format(h93 / h)
                h14 = '{0:.2f}'.format(h14 / h)
                h24 = '{0:.2f}'.format(h24 / h)
                h34 = '{0:.2f}'.format(h34 / h)
                h44 = '{0:.2f}'.format(h44 / h)
                h54 = '{0:.2f}'.format(h54 / h)
                h64 = '{0:.2f}'.format(h64 / h)
                h74 = '{0:.2f}'.format(h74 / h)
                h84 = '{0:.2f}'.format(h84 / h)
                h94 = '{0:.2f}'.format(h94 / h)
            else:
                h1 = '{0:.2f}'.format(h1)
                h2 = '{0:.2f}'.format(h2)
                h3 = '{0:.2f}'.format(h3)
                h4 = '{0:.2f}'.format(h4)
                h5 = '{0:.2f}'.format(h5)
                h6 = '{0:.2f}'.format(h6)
                h7 = '{0:.2f}'.format(h7)
                h8 = '{0:.2f}'.format(h8)
                h9 = '{0:.2f}'.format(h9)
                h12 = '{0:.2f}'.format(h12)
                h22 = '{0:.2f}'.format(h22)
                h32 = '{0:.2f}'.format(h32)
                h42 = '{0:.2f}'.format(h42)
                h52 = '{0:.2f}'.format(h52)
                h62 = '{0:.2f}'.format(h62)
                h72 = '{0:.2f}'.format(h72)
                h82 = '{0:.2f}'.format(h82)
                h92 = '{0:.2f}'.format(h92)
                h13 = '{0:.2f}'.format(h13)
                h23 = '{0:.2f}'.format(h23)
                h33 = '{0:.2f}'.format(h33)
                h43 = '{0:.2f}'.format(h43)
                h53 = '{0:.2f}'.format(h53)
                h63 = '{0:.2f}'.format(h63)
                h73 = '{0:.2f}'.format(h73)
                h83 = '{0:.2f}'.format(h83)
                h93 = '{0:.2f}'.format(h93)
                h14 = '{0:.2f}'.format(h14)
                h24 = '{0:.2f}'.format(h24)
                h34 = '{0:.2f}'.format(h34)
                h44 = '{0:.2f}'.format(h44)
                h54 = '{0:.2f}'.format(h54)
                h64 = '{0:.2f}'.format(h64)
                h74 = '{0:.2f}'.format(h74)
                h84 = '{0:.2f}'.format(h84)
                h94 = '{0:.2f}'.format(h94)
            positiveSamplesHOG[index][count][0] = h1
            count = count + 1
            positiveSamplesHOG[index][count][0] = h2
            count = count + 1
            positiveSamplesHOG[index][count][0] = h3
            count = count + 1
            positiveSamplesHOG[index][count][0] = h4
            count = count + 1
            positiveSamplesHOG[index][count][0] = h5
            count = count + 1
            positiveSamplesHOG[index][count][0] = h6
            count = count + 1
            positiveSamplesHOG[index][count][0] = h7
            count = count + 1
            positiveSamplesHOG[index][count][0] = h8
            count = count + 1
            positiveSamplesHOG[index][count][0] = h9
            count = count + 1
            positiveSamplesHOG[index][count][0] = h12
            count = count + 1
            positiveSamplesHOG[index][count][0] = h22
            count = count + 1
            positiveSamplesHOG[index][count][0] = h32
            count = count + 1
            positiveSamplesHOG[index][count][0] = h42
            count = count + 1
            positiveSamplesHOG[index][count][0] = h52
            count = count + 1
            positiveSamplesHOG[index][count][0] = h62
            count = count + 1
            positiveSamplesHOG[index][count][0] = h72
            count = count + 1
            positiveSamplesHOG[index][count][0] = h82
            count = count + 1
            positiveSamplesHOG[index][count][0] = h92
            count = count + 1
            positiveSamplesHOG[index][count][0] = h13
            count = count + 1
            positiveSamplesHOG[index][count][0] = h23
            count = count + 1
            positiveSamplesHOG[index][count][0] = h33
            count = count + 1
            positiveSamplesHOG[index][count][0] = h43
            count = count + 1
            positiveSamplesHOG[index][count][0] = h53
            count = count + 1
            positiveSamplesHOG[index][count][0] = h63
            count = count + 1
            positiveSamplesHOG[index][count][0] = h73
            count = count + 1
            positiveSamplesHOG[index][count][0] = h83
            count = count + 1
            positiveSamplesHOG[index][count][0] = h93
            count = count + 1
            positiveSamplesHOG[index][count][0] = h14
            count = count + 1
            positiveSamplesHOG[index][count][0] = h24
            count = count + 1
            positiveSamplesHOG[index][count][0] = h34
            count = count + 1
            positiveSamplesHOG[index][count][0] = h44
            count = count + 1
            positiveSamplesHOG[index][count][0] = h54
            count = count + 1
            positiveSamplesHOG[index][count][0] = h64
            count = count + 1
            positiveSamplesHOG[index][count][0] = h74
            count = count + 1
            positiveSamplesHOG[index][count][0] = h84
            count = count + 1
            positiveSamplesHOG[index][count][0] = h94
            count = count + 1

            name = trainingPositiveList[index].replace(".bmp",".txt",1)
            fileName = "ComputedResults/HOGDescriptors/" + name
            with open(fileName, "a") as myfile:
                myfile.write("\n" +str(h1)+" "+str(h2)+" "+str(h3)+" "+str(h4)+" "+str(h5)+" "+str(h6)+" "+str(h7)+" "+str(h8)+" "+str(h9)+" "+str(h12)+" "+str(h22)+" "+str(h32)+" "+str(h42)+" "+str(h52)+" "+str(h62)+" "+str(h72)+" "+str(h82)+" "+str(h92)+" "+str(h13)+" "+str(h23)+" "+str(h33)+" "+str(h43)+" "+str(h53)+" "+str(h63)+" "+str(h73)+" "+str(h83)+" "+str(h93)+" "+str(h14)+" "+str(h24)+" "+str(h34)+" "+str(h44)+" "+str(h54)+" "+str(h64)+" "+str(h74)+" "+str(h84)+" "+str(h94))
            myfile.close()

# Calculate Normalized Gradient Orientation Histograms for Negative Training Images (Same as for Positive Training Images)
print("Computing Gradient Orientation Histogram for Negative Training Images...")
for index in range(0, 10):
    count = 0
    for i in xrange(16, 129, 8):
        for j in xrange(16, 65, 8):
            h1 = 0.0
            h2 = 0.0
            h3 = 0.0
            h4 = 0.0
            h5 = 0.0
            h6 = 0.0
            h7 = 0.0
            h8 = 0.0
            h9 = 0.0
            h12 = 0.0
            h22 = 0.0
            h32 = 0.0
            h42 = 0.0
            h52 = 0.0
            h62 = 0.0
            h72 = 0.0
            h82 = 0.0
            h92 = 0.0
            h13 = 0.0
            h23 = 0.0
            h33 = 0.0
            h43 = 0.0
            h53 = 0.0
            h63 = 0.0
            h73 = 0.0
            h83 = 0.0
            h93 = 0.0
            h14 = 0.0
            h24 = 0.0
            h34 = 0.0
            h44 = 0.0
            h54 = 0.0
            h64 = 0.0
            h74 = 0.0
            h84 = 0.0
            h94 = 0.0
            h = 0.0
            for k in range(i, i+8):
                for l in range(j, j+8):
                    if (trainingNegativeQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h1 = h1 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h2 = h2 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h9 = h9 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h2 = h2 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h3 = h3 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h1 = h1 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h3 = h3 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h4 = h4 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h2 = h2 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h4 = h4 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h5 = h5 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h3 = h3 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h5 = h5 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h6 = h6 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h4 = h4 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h6 = h6 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h7 = h7 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h5 = h5 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h7 = h7 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h8 = h8 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h6 = h6 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h8 = h8 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h9 = h9 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h7 = h7 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h9 = h9 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h1 = h1 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h8 = h8 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
        
            for k in range(i, i+8):
                for l in range(j+8, j+16):
                    if (trainingNegativeQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h12 = h12 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h22 = h22 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h92 = h92 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h22 = h22 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h32 = h32 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h12 = h12 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h32 = h32 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h42 = h42 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h22 = h22 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h42 = h42 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h52 = h52 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h32 = h32 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h52 = h52 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h62 = h62 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h42 = h42 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h62 = h62 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h72 = h72 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h52 = h52 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h72 = h72 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h82 = h82 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h62 = h62 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h82 = h82 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h92 = h92 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h72 = h72 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h92 = h92 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h12 = h12 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h82 = h82 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])

            for k in range(i+8, i+16):
                for l in range(j, j+8):
                    if (trainingNegativeQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h13 = h13 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h23 = h23 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h93 = h93 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h23 = h23 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h33 = h33 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h13 = h13 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h33 = h33 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h43 = h43 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h23 = h23 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h43 = h43 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h53 = h53 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h33 = h33 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h53 = h53 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h63 = h63 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h43 = h43 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h63 = h63 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h73 = h73 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h53 = h53 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h73 = h73 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h83 = h83 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h63 = h63 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h83 = h83 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h93 = h93 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h73 = h73 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h93 = h93 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h13 = h13 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h83 = h83 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])

            for k in range(i+8, i+16):
                for l in range(j+8, j+16):
                    if (trainingNegativeQuantizedAngle[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h14 = h14 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h24 = h24 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h94 = h94 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h24 = h24 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h34 = h34 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h14 = h14 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h34 = h34 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h44 = h44 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h24 = h24 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h4 = h4 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h54 = h54 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h34 = h34 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h54 = h54 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h64 = h64 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h44 = h44 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h64 = h64 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h74 = h74 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h54 = h54 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h74 = h74 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h84 = h84 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h64 = h64 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h84 = h84 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h94 = h94 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h74 = h74 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                    if (trainingNegativeQuantizedAngle[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(trainingNegativeGradientAngle[index][k][l]-trainingNegativeQuantizedAngle[index][k][l])/20)
                        h94 = h94 + (proportion * trainingNegativeGradientMagnitude[index][k][l])
                        if (trainingNegativeGradientAngle[index][k][l] > trainingNegativeQuantizedAngle[index][k][l]):
                            h14 = h14 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
                        else:
                            h84 = h84 + ((1 - proportion) * trainingNegativeGradientMagnitude[index][k][l])
            h = ((h1*h1)+(h2*h2)+(h3*h3)+(h4*h4)+(h5*h5)+(h6*h6)+(h7*h7)+(h8*h8)+(h9*h9)+(h12*h12)+(h22*h22)+(h32*h32)+(h42*h42)+(h52*h52)+(h62*h62)+(h72*h72)+(h82*h82)+(h92*h92)+(h13*h13)+(h23*h23)+(h33*h33)+(h43*h43)+(h53*h53)+(h63*h63)+(h73*h73)+(h83*h83)+(h93*h93)+(h14*h14)+(h24*h24)+(h34*h34)+(h44*h44)+(h54*h54)+(h64*h64)+(h74*h74)+(h84*h84)+(h94*h94))**0.5
            if h != 0.0:
                h1 = '{0:.2f}'.format(h1 / h)
                h2 = '{0:.2f}'.format(h2 / h)
                h3 = '{0:.2f}'.format(h3 / h)
                h4 = '{0:.2f}'.format(h4 / h)
                h5 = '{0:.2f}'.format(h5 / h)
                h6 = '{0:.2f}'.format(h6 / h)
                h7 = '{0:.2f}'.format(h7 / h)
                h8 = '{0:.2f}'.format(h8 / h)
                h9 = '{0:.2f}'.format(h9 / h)
                h12 = '{0:.2f}'.format(h12 / h)
                h22 = '{0:.2f}'.format(h22 / h)
                h32 = '{0:.2f}'.format(h32 / h)
                h42 = '{0:.2f}'.format(h42 / h)
                h52 = '{0:.2f}'.format(h52 / h)
                h62 = '{0:.2f}'.format(h62 / h)
                h72 = '{0:.2f}'.format(h72 / h)
                h82 = '{0:.2f}'.format(h82 / h)
                h92 = '{0:.2f}'.format(h92 / h)
                h13 = '{0:.2f}'.format(h13 / h)
                h23 = '{0:.2f}'.format(h23 / h)
                h33 = '{0:.2f}'.format(h33 / h)
                h43 = '{0:.2f}'.format(h43 / h)
                h53 = '{0:.2f}'.format(h53 / h)
                h63 = '{0:.2f}'.format(h63 / h)
                h73 = '{0:.2f}'.format(h73 / h)
                h83 = '{0:.2f}'.format(h83 / h)
                h93 = '{0:.2f}'.format(h93 / h)
                h14 = '{0:.2f}'.format(h14 / h)
                h24 = '{0:.2f}'.format(h24 / h)
                h34 = '{0:.2f}'.format(h34 / h)
                h44 = '{0:.2f}'.format(h44 / h)
                h54 = '{0:.2f}'.format(h54 / h)
                h64 = '{0:.2f}'.format(h64 / h)
                h74 = '{0:.2f}'.format(h74 / h)
                h84 = '{0:.2f}'.format(h84 / h)
                h94 = '{0:.2f}'.format(h94 / h)
            else:
                h1 = '{0:.2f}'.format(h1)
                h2 = '{0:.2f}'.format(h2)
                h3 = '{0:.2f}'.format(h3)
                h4 = '{0:.2f}'.format(h4)
                h5 = '{0:.2f}'.format(h5)
                h6 = '{0:.2f}'.format(h6)
                h7 = '{0:.2f}'.format(h7)
                h8 = '{0:.2f}'.format(h8)
                h9 = '{0:.2f}'.format(h9)
                h12 = '{0:.2f}'.format(h12)
                h22 = '{0:.2f}'.format(h22)
                h32 = '{0:.2f}'.format(h32)
                h42 = '{0:.2f}'.format(h42)
                h52 = '{0:.2f}'.format(h52)
                h62 = '{0:.2f}'.format(h62)
                h72 = '{0:.2f}'.format(h72)
                h82 = '{0:.2f}'.format(h82)
                h92 = '{0:.2f}'.format(h92)
                h13 = '{0:.2f}'.format(h13)
                h23 = '{0:.2f}'.format(h23)
                h33 = '{0:.2f}'.format(h33)
                h43 = '{0:.2f}'.format(h43)
                h53 = '{0:.2f}'.format(h53)
                h63 = '{0:.2f}'.format(h63)
                h73 = '{0:.2f}'.format(h73)
                h83 = '{0:.2f}'.format(h83)
                h93 = '{0:.2f}'.format(h93)
                h14 = '{0:.2f}'.format(h14)
                h24 = '{0:.2f}'.format(h24)
                h34 = '{0:.2f}'.format(h34)
                h44 = '{0:.2f}'.format(h44)
                h54 = '{0:.2f}'.format(h54)
                h64 = '{0:.2f}'.format(h64)
                h74 = '{0:.2f}'.format(h74)
                h84 = '{0:.2f}'.format(h84)
                h94 = '{0:.2f}'.format(h94)
            negativeSamplesHOG[index][count][0] = h1
            count = count + 1
            negativeSamplesHOG[index][count][0] = h2
            count = count + 1
            negativeSamplesHOG[index][count][0] = h3
            count = count + 1
            negativeSamplesHOG[index][count][0] = h4
            count = count + 1
            negativeSamplesHOG[index][count][0] = h5
            count = count + 1
            negativeSamplesHOG[index][count][0] = h6
            count = count + 1
            negativeSamplesHOG[index][count][0] = h7
            count = count + 1
            negativeSamplesHOG[index][count][0] = h8
            count = count + 1
            negativeSamplesHOG[index][count][0] = h9
            count = count + 1
            negativeSamplesHOG[index][count][0] = h12
            count = count + 1
            negativeSamplesHOG[index][count][0] = h22
            count = count + 1
            negativeSamplesHOG[index][count][0] = h32
            count = count + 1
            negativeSamplesHOG[index][count][0] = h42
            count = count + 1
            negativeSamplesHOG[index][count][0] = h52
            count = count + 1
            negativeSamplesHOG[index][count][0] = h62
            count = count + 1
            negativeSamplesHOG[index][count][0] = h72
            count = count + 1
            negativeSamplesHOG[index][count][0] = h82
            count = count + 1
            negativeSamplesHOG[index][count][0] = h92
            count = count + 1
            negativeSamplesHOG[index][count][0] = h13
            count = count + 1
            negativeSamplesHOG[index][count][0] = h23
            count = count + 1
            negativeSamplesHOG[index][count][0] = h33
            count = count + 1
            negativeSamplesHOG[index][count][0] = h43
            count = count + 1
            negativeSamplesHOG[index][count][0] = h53
            count = count + 1
            negativeSamplesHOG[index][count][0] = h63
            count = count + 1
            negativeSamplesHOG[index][count][0] = h73
            count = count + 1
            negativeSamplesHOG[index][count][0] = h83
            count = count + 1
            negativeSamplesHOG[index][count][0] = h93
            count = count + 1
            negativeSamplesHOG[index][count][0] = h14
            count = count + 1
            negativeSamplesHOG[index][count][0] = h24
            count = count + 1
            negativeSamplesHOG[index][count][0] = h34
            count = count + 1
            negativeSamplesHOG[index][count][0] = h44
            count = count + 1
            negativeSamplesHOG[index][count][0] = h54
            count = count + 1
            negativeSamplesHOG[index][count][0] = h64
            count = count + 1
            negativeSamplesHOG[index][count][0] = h74
            count = count + 1
            negativeSamplesHOG[index][count][0] = h84
            count = count + 1
            negativeSamplesHOG[index][count][0] = h94
            count = count + 1
            
            name = trainingNegativeList[index].replace(".bmp",".txt",1)
            fileName = "ComputedResults/HOGDescriptors/" + name
            with open(fileName, "a") as myfile:
                myfile.write("\n" +str(h1)+" "+str(h2)+" "+str(h3)+" "+str(h4)+" "+str(h5)+" "+str(h6)+" "+str(h7)+" "+str(h8)+" "+str(h9)+" "+str(h12)+" "+str(h22)+" "+str(h32)+" "+str(h42)+" "+str(h52)+" "+str(h62)+" "+str(h72)+" "+str(h82)+" "+str(h92)+" "+str(h13)+" "+str(h23)+" "+str(h33)+" "+str(h43)+" "+str(h53)+" "+str(h63)+" "+str(h73)+" "+str(h83)+" "+str(h93)+" "+str(h14)+" "+str(h24)+" "+str(h34)+" "+str(h44)+" "+str(h54)+" "+str(h64)+" "+str(h74)+" "+str(h84)+" "+str(h94))
            myfile.close()

# Calculate Normalized Gradient Orientation Histograms for Test Images Set 1  (Same as for Training Images)
print("Computing Gradient Orientation Histogram for Test Sample I Images...")
for index in range(0, 5):
    count = 0
    for i in xrange(16, 129, 8):
        for j in xrange(16, 65, 8):
            h1 = 0.0
            h2 = 0.0
            h3 = 0.0
            h4 = 0.0
            h5 = 0.0
            h6 = 0.0
            h7 = 0.0
            h8 = 0.0
            h9 = 0.0
            h12 = 0.0
            h22 = 0.0
            h32 = 0.0
            h42 = 0.0
            h52 = 0.0
            h62 = 0.0
            h72 = 0.0
            h82 = 0.0
            h92 = 0.0
            h13 = 0.0
            h23 = 0.0
            h33 = 0.0
            h43 = 0.0
            h53 = 0.0
            h63 = 0.0
            h73 = 0.0
            h83 = 0.0
            h93 = 0.0
            h14 = 0.0
            h24 = 0.0
            h34 = 0.0
            h44 = 0.0
            h54 = 0.0
            h64 = 0.0
            h74 = 0.0
            h84 = 0.0
            h94 = 0.0
            h = 0.0
            for k in range(i, i+8):
                for l in range(j, j+8):
                    if (testQuantizedAngle1[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h1 = h1 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h2 = h2 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h9 = h9 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h2 = h2 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h3 = h3 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h1 = h1 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h3 = h3 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h4 = h4 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h2 = h2 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h4 = h4 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h5 = h5 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h3 = h3 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h5 = h5 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h6 = h6 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h4 = h4 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h6 = h6 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h7 = h7 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h5 = h5 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h7 = h7 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h8 = h8 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h6 = h6 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h8 = h8 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h9 = h9 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h7 = h7 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h9 = h9 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h1 = h1 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h8 = h8 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
        
            for k in range(i, i+8):
                for l in range(j+8, j+16):
                    if (testQuantizedAngle1[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h12 = h12 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h22 = h22 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h92 = h92 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h22 = h22 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h32 = h32 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h12 = h12 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h32 = h32 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h42 = h42 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h22 = h22 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h42 = h42 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h52 = h52 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h32 = h32 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h52 = h52 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h62 = h62 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h42 = h42 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h62 = h62 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h72 = h72 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h52 = h52 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h72 = h72 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h82 = h82 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h62 = h62 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h82 = h82 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h92 = h92 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h72 = h72 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h92 = h92 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h12 = h12 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h82 = h82 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])

            for k in range(i+8, i+16):
                for l in range(j, j+8):
                    if (testQuantizedAngle1[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h13 = h13 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h23 = h23 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h93 = h93 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h23 = h23 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h33 = h33 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h13 = h13 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h33 = h33 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h43 = h43 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h23 = h23 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h43 = h43 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h53 = h53 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h33 = h33 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h53 = h53 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h63 = h63 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h43 = h43 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h63 = h63 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h73 = h73 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h53 = h53 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h73 = h73 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h83 = h83 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h63 = h63 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h83 = h83 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h93 = h93 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h73 = h73 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h93 = h93 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h13 = h13 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h83 = h83 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])

            for k in range(i+8, i+16):
                for l in range(j+8, j+16):
                    if (testQuantizedAngle1[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h14 = h14 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h24 = h24 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h94 = h94 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h24 = h24 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h34 = h34 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h14 = h14 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h34 = h34 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h44 = h44 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h24 = h24 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h4 = h4 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h54 = h54 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h34 = h34 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h54 = h54 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h64 = h64 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h44 = h44 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h64 = h64 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h74 = h74 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h54 = h54 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h74 = h74 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h84 = h84 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h64 = h64 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h84 = h84 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h94 = h94 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h74 = h74 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                    if (testQuantizedAngle1[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle1[index][k][l]-testQuantizedAngle1[index][k][l])/20)
                        h94 = h94 + (proportion * testSamplesGradientMagnitude1[index][k][l])
                        if (testSamplesGradientAngle1[index][k][l] > testQuantizedAngle1[index][k][l]):
                            h14 = h14 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
                        else:
                            h84 = h84 + ((1 - proportion) * testSamplesGradientMagnitude1[index][k][l])
            h = ((h1*h1)+(h2*h2)+(h3*h3)+(h4*h4)+(h5*h5)+(h6*h6)+(h7*h7)+(h8*h8)+(h9*h9)+(h12*h12)+(h22*h22)+(h32*h32)+(h42*h42)+(h52*h52)+(h62*h62)+(h72*h72)+(h82*h82)+(h92*h92)+(h13*h13)+(h23*h23)+(h33*h33)+(h43*h43)+(h53*h53)+(h63*h63)+(h73*h73)+(h83*h83)+(h93*h93)+(h14*h14)+(h24*h24)+(h34*h34)+(h44*h44)+(h54*h54)+(h64*h64)+(h74*h74)+(h84*h84)+(h94*h94))**0.5
            if h != 0.0:
                h1 = '{0:.2f}'.format(h1 / h)
                h2 = '{0:.2f}'.format(h2 / h)
                h3 = '{0:.2f}'.format(h3 / h)
                h4 = '{0:.2f}'.format(h4 / h)
                h5 = '{0:.2f}'.format(h5 / h)
                h6 = '{0:.2f}'.format(h6 / h)
                h7 = '{0:.2f}'.format(h7 / h)
                h8 = '{0:.2f}'.format(h8 / h)
                h9 = '{0:.2f}'.format(h9 / h)
                h12 = '{0:.2f}'.format(h12 / h)
                h22 = '{0:.2f}'.format(h22 / h)
                h32 = '{0:.2f}'.format(h32 / h)
                h42 = '{0:.2f}'.format(h42 / h)
                h52 = '{0:.2f}'.format(h52 / h)
                h62 = '{0:.2f}'.format(h62 / h)
                h72 = '{0:.2f}'.format(h72 / h)
                h82 = '{0:.2f}'.format(h82 / h)
                h92 = '{0:.2f}'.format(h92 / h)
                h13 = '{0:.2f}'.format(h13 / h)
                h23 = '{0:.2f}'.format(h23 / h)
                h33 = '{0:.2f}'.format(h33 / h)
                h43 = '{0:.2f}'.format(h43 / h)
                h53 = '{0:.2f}'.format(h53 / h)
                h63 = '{0:.2f}'.format(h63 / h)
                h73 = '{0:.2f}'.format(h73 / h)
                h83 = '{0:.2f}'.format(h83 / h)
                h93 = '{0:.2f}'.format(h93 / h)
                h14 = '{0:.2f}'.format(h14 / h)
                h24 = '{0:.2f}'.format(h24 / h)
                h34 = '{0:.2f}'.format(h34 / h)
                h44 = '{0:.2f}'.format(h44 / h)
                h54 = '{0:.2f}'.format(h54 / h)
                h64 = '{0:.2f}'.format(h64 / h)
                h74 = '{0:.2f}'.format(h74 / h)
                h84 = '{0:.2f}'.format(h84 / h)
                h94 = '{0:.2f}'.format(h94 / h)
            else:
                h1 = '{0:.2f}'.format(h1)
                h2 = '{0:.2f}'.format(h2)
                h3 = '{0:.2f}'.format(h3)
                h4 = '{0:.2f}'.format(h4)
                h5 = '{0:.2f}'.format(h5)
                h6 = '{0:.2f}'.format(h6)
                h7 = '{0:.2f}'.format(h7)
                h8 = '{0:.2f}'.format(h8)
                h9 = '{0:.2f}'.format(h9)
                h12 = '{0:.2f}'.format(h12)
                h22 = '{0:.2f}'.format(h22)
                h32 = '{0:.2f}'.format(h32)
                h42 = '{0:.2f}'.format(h42)
                h52 = '{0:.2f}'.format(h52)
                h62 = '{0:.2f}'.format(h62)
                h72 = '{0:.2f}'.format(h72)
                h82 = '{0:.2f}'.format(h82)
                h92 = '{0:.2f}'.format(h92)
                h13 = '{0:.2f}'.format(h13)
                h23 = '{0:.2f}'.format(h23)
                h33 = '{0:.2f}'.format(h33)
                h43 = '{0:.2f}'.format(h43)
                h53 = '{0:.2f}'.format(h53)
                h63 = '{0:.2f}'.format(h63)
                h73 = '{0:.2f}'.format(h73)
                h83 = '{0:.2f}'.format(h83)
                h93 = '{0:.2f}'.format(h93)
                h14 = '{0:.2f}'.format(h14)
                h24 = '{0:.2f}'.format(h24)
                h34 = '{0:.2f}'.format(h34)
                h44 = '{0:.2f}'.format(h44)
                h54 = '{0:.2f}'.format(h54)
                h64 = '{0:.2f}'.format(h64)
                h74 = '{0:.2f}'.format(h74)
                h84 = '{0:.2f}'.format(h84)
                h94 = '{0:.2f}'.format(h94)
            testSamples1HOG[index][count][0] = h1
            count = count + 1
            testSamples1HOG[index][count][0] = h2
            count = count + 1
            testSamples1HOG[index][count][0] = h3
            count = count + 1
            testSamples1HOG[index][count][0] = h4
            count = count + 1
            testSamples1HOG[index][count][0] = h5
            count = count + 1
            testSamples1HOG[index][count][0] = h6
            count = count + 1
            testSamples1HOG[index][count][0] = h7
            count = count + 1
            testSamples1HOG[index][count][0] = h8
            count = count + 1
            testSamples1HOG[index][count][0] = h9
            count = count + 1
            testSamples1HOG[index][count][0] = h12
            count = count + 1
            testSamples1HOG[index][count][0] = h22
            count = count + 1
            testSamples1HOG[index][count][0] = h32
            count = count + 1
            testSamples1HOG[index][count][0] = h42
            count = count + 1
            testSamples1HOG[index][count][0] = h52
            count = count + 1
            testSamples1HOG[index][count][0] = h62
            count = count + 1
            testSamples1HOG[index][count][0] = h72
            count = count + 1
            testSamples1HOG[index][count][0] = h82
            count = count + 1
            testSamples1HOG[index][count][0] = h92
            count = count + 1
            testSamples1HOG[index][count][0] = h13
            count = count + 1
            testSamples1HOG[index][count][0] = h23
            count = count + 1
            testSamples1HOG[index][count][0] = h33
            count = count + 1
            testSamples1HOG[index][count][0] = h43
            count = count + 1
            testSamples1HOG[index][count][0] = h53
            count = count + 1
            testSamples1HOG[index][count][0] = h63
            count = count + 1
            testSamples1HOG[index][count][0] = h73
            count = count + 1
            testSamples1HOG[index][count][0] = h83
            count = count + 1
            testSamples1HOG[index][count][0] = h93
            count = count + 1
            testSamples1HOG[index][count][0] = h14
            count = count + 1
            testSamples1HOG[index][count][0] = h24
            count = count + 1
            testSamples1HOG[index][count][0] = h34
            count = count + 1
            testSamples1HOG[index][count][0] = h44
            count = count + 1
            testSamples1HOG[index][count][0] = h54
            count = count + 1
            testSamples1HOG[index][count][0] = h64
            count = count + 1
            testSamples1HOG[index][count][0] = h74
            count = count + 1
            testSamples1HOG[index][count][0] = h84
            count = count + 1
            testSamples1HOG[index][count][0] = h94
            count = count + 1
            
            name = testList1[index].replace(".bmp",".txt",1)
            fileName = "ComputedResults/HOGDescriptors/" + name
            with open(fileName, "a") as myfile:
                myfile.write("\n" +str(h1)+" "+str(h2)+" "+str(h3)+" "+str(h4)+" "+str(h5)+" "+str(h6)+" "+str(h7)+" "+str(h8)+" "+str(h9)+" "+str(h12)+" "+str(h22)+" "+str(h32)+" "+str(h42)+" "+str(h52)+" "+str(h62)+" "+str(h72)+" "+str(h82)+" "+str(h92)+" "+str(h13)+" "+str(h23)+" "+str(h33)+" "+str(h43)+" "+str(h53)+" "+str(h63)+" "+str(h73)+" "+str(h83)+" "+str(h93)+" "+str(h14)+" "+str(h24)+" "+str(h34)+" "+str(h44)+" "+str(h54)+" "+str(h64)+" "+str(h74)+" "+str(h84)+" "+str(h94))
            myfile.close()

# Calculate Normalized Gradient Orientation Histograms for Test Images Set 2  (Same as for Training Images)
print("Computing Gradient Orientation Histogram for Test Sample II Images...")
for index in range(0, 5):
    count = 0
    for i in xrange(16, 129, 8):
        for j in xrange(16, 65, 8):
            h1 = 0.0
            h2 = 0.0
            h3 = 0.0
            h4 = 0.0
            h5 = 0.0
            h6 = 0.0
            h7 = 0.0
            h8 = 0.0
            h9 = 0.0
            h12 = 0.0
            h22 = 0.0
            h32 = 0.0
            h42 = 0.0
            h52 = 0.0
            h62 = 0.0
            h72 = 0.0
            h82 = 0.0
            h92 = 0.0
            h13 = 0.0
            h23 = 0.0
            h33 = 0.0
            h43 = 0.0
            h53 = 0.0
            h63 = 0.0
            h73 = 0.0
            h83 = 0.0
            h93 = 0.0
            h14 = 0.0
            h24 = 0.0
            h34 = 0.0
            h44 = 0.0
            h54 = 0.0
            h64 = 0.0
            h74 = 0.0
            h84 = 0.0
            h94 = 0.0
            h = 0.0
            for k in range(i, i+8):
                for l in range(j, j+8):
                    if (testQuantizedAngle2[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h1 = h1 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h2 = h2 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h9 = h9 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h2 = h2 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h3 = h3 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h1 = h1 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h3 = h3 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h4 = h4 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h2 = h2 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h4 = h4 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h5 = h5 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h3 = h3 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h5 = h5 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h6 = h6 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h4 = h4 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h6 = h6 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h7 = h7 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h5 = h5 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h7 = h7 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h8 = h8 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h6 = h6 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h8 = h8 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h9 = h9 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h7 = h7 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h9 = h9 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h1 = h1 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h8 = h8 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
        
            for k in range(i, i+8):
                for l in range(j+8, j+16):
                    if (testQuantizedAngle2[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h12 = h12 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h22 = h22 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h92 = h92 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h22 = h22 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h32 = h32 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h12 = h12 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h32 = h32 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h42 = h42 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h22 = h22 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h42 = h42 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h52 = h52 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h32 = h32 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h52 = h52 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h62 = h62 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h42 = h42 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h62 = h62 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h72 = h72 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h52 = h52 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h72 = h72 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h82 = h82 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h62 = h62 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h82 = h82 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h92 = h92 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h72 = h72 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h92 = h92 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h12 = h12 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h82 = h82 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])

            for k in range(i+8, i+16):
                for l in range(j, j+8):
                    if (testQuantizedAngle2[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h13 = h13 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h23 = h23 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h93 = h93 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h23 = h23 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h33 = h33 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h13 = h13 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h33 = h33 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h43 = h43 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h23 = h23 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h43 = h43 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h53 = h53 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h33 = h33 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h53 = h53 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h63 = h63 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h43 = h43 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h63 = h63 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h73 = h73 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h53 = h53 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h73 = h73 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h83 = h83 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h63 = h63 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h83 = h83 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h93 = h93 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h73 = h73 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h93 = h93 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h13 = h13 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h83 = h83 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])

            for k in range(i+8, i+16):
                for l in range(j+8, j+16):
                    if (testQuantizedAngle2[index][k][l] == 10):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h14 = h14 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h24 = h24 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h94 = h94 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 30):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h24 = h24 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h34 = h34 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h14 = h14 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 50):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h34 = h34 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h44 = h44 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h24 = h24 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 70):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h4 = h4 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h54 = h54 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h34 = h34 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 90):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h54 = h54 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h64 = h64 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h44 = h44 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 110):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h64 = h64 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h74 = h74 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h54 = h54 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 130):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h74 = h74 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h84 = h84 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h64 = h64 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 150):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h84 = h84 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h94 = h94 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h74 = h74 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                    if (testQuantizedAngle2[index][k][l] == 170):
                        proportion = 1.0 - (math.fabs(testSamplesGradientAngle2[index][k][l]-testQuantizedAngle2[index][k][l])/20)
                        h94 = h94 + (proportion * testSamplesGradientMagnitude2[index][k][l])
                        if (testSamplesGradientAngle2[index][k][l] > testQuantizedAngle2[index][k][l]):
                            h14 = h14 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
                        else:
                            h84 = h84 + ((1 - proportion) * testSamplesGradientMagnitude2[index][k][l])
            h = ((h1*h1)+(h2*h2)+(h3*h3)+(h4*h4)+(h5*h5)+(h6*h6)+(h7*h7)+(h8*h8)+(h9*h9)+(h12*h12)+(h22*h22)+(h32*h32)+(h42*h42)+(h52*h52)+(h62*h62)+(h72*h72)+(h82*h82)+(h92*h92)+(h13*h13)+(h23*h23)+(h33*h33)+(h43*h43)+(h53*h53)+(h63*h63)+(h73*h73)+(h83*h83)+(h93*h93)+(h14*h14)+(h24*h24)+(h34*h34)+(h44*h44)+(h54*h54)+(h64*h64)+(h74*h74)+(h84*h84)+(h94*h94))**0.5
            if h != 0.0:
                h1 = '{0:.2f}'.format(h1 / h)
                h2 = '{0:.2f}'.format(h2 / h)
                h3 = '{0:.2f}'.format(h3 / h)
                h4 = '{0:.2f}'.format(h4 / h)
                h5 = '{0:.2f}'.format(h5 / h)
                h6 = '{0:.2f}'.format(h6 / h)
                h7 = '{0:.2f}'.format(h7 / h)
                h8 = '{0:.2f}'.format(h8 / h)
                h9 = '{0:.2f}'.format(h9 / h)
                h12 = '{0:.2f}'.format(h12 / h)
                h22 = '{0:.2f}'.format(h22 / h)
                h32 = '{0:.2f}'.format(h32 / h)
                h42 = '{0:.2f}'.format(h42 / h)
                h52 = '{0:.2f}'.format(h52 / h)
                h62 = '{0:.2f}'.format(h62 / h)
                h72 = '{0:.2f}'.format(h72 / h)
                h82 = '{0:.2f}'.format(h82 / h)
                h92 = '{0:.2f}'.format(h92 / h)
                h13 = '{0:.2f}'.format(h13 / h)
                h23 = '{0:.2f}'.format(h23 / h)
                h33 = '{0:.2f}'.format(h33 / h)
                h43 = '{0:.2f}'.format(h43 / h)
                h53 = '{0:.2f}'.format(h53 / h)
                h63 = '{0:.2f}'.format(h63 / h)
                h73 = '{0:.2f}'.format(h73 / h)
                h83 = '{0:.2f}'.format(h83 / h)
                h93 = '{0:.2f}'.format(h93 / h)
                h14 = '{0:.2f}'.format(h14 / h)
                h24 = '{0:.2f}'.format(h24 / h)
                h34 = '{0:.2f}'.format(h34 / h)
                h44 = '{0:.2f}'.format(h44 / h)
                h54 = '{0:.2f}'.format(h54 / h)
                h64 = '{0:.2f}'.format(h64 / h)
                h74 = '{0:.2f}'.format(h74 / h)
                h84 = '{0:.2f}'.format(h84 / h)
                h94 = '{0:.2f}'.format(h94 / h)
            else:
                h1 = '{0:.2f}'.format(h1)
                h2 = '{0:.2f}'.format(h2)
                h3 = '{0:.2f}'.format(h3)
                h4 = '{0:.2f}'.format(h4)
                h5 = '{0:.2f}'.format(h5)
                h6 = '{0:.2f}'.format(h6)
                h7 = '{0:.2f}'.format(h7)
                h8 = '{0:.2f}'.format(h8)
                h9 = '{0:.2f}'.format(h9)
                h12 = '{0:.2f}'.format(h12)
                h22 = '{0:.2f}'.format(h22)
                h32 = '{0:.2f}'.format(h32)
                h42 = '{0:.2f}'.format(h42)
                h52 = '{0:.2f}'.format(h52)
                h62 = '{0:.2f}'.format(h62)
                h72 = '{0:.2f}'.format(h72)
                h82 = '{0:.2f}'.format(h82)
                h92 = '{0:.2f}'.format(h92)
                h13 = '{0:.2f}'.format(h13)
                h23 = '{0:.2f}'.format(h23)
                h33 = '{0:.2f}'.format(h33)
                h43 = '{0:.2f}'.format(h43)
                h53 = '{0:.2f}'.format(h53)
                h63 = '{0:.2f}'.format(h63)
                h73 = '{0:.2f}'.format(h73)
                h83 = '{0:.2f}'.format(h83)
                h93 = '{0:.2f}'.format(h93)
                h14 = '{0:.2f}'.format(h14)
                h24 = '{0:.2f}'.format(h24)
                h34 = '{0:.2f}'.format(h34)
                h44 = '{0:.2f}'.format(h44)
                h54 = '{0:.2f}'.format(h54)
                h64 = '{0:.2f}'.format(h64)
                h74 = '{0:.2f}'.format(h74)
                h84 = '{0:.2f}'.format(h84)
                h94 = '{0:.2f}'.format(h94)
            testSamples2HOG[index][count][0] = h1
            count = count + 1
            testSamples2HOG[index][count][0] = h2
            count = count + 1
            testSamples2HOG[index][count][0] = h3
            count = count + 1
            testSamples2HOG[index][count][0] = h4
            count = count + 1
            testSamples2HOG[index][count][0] = h5
            count = count + 1
            testSamples2HOG[index][count][0] = h6
            count = count + 1
            testSamples2HOG[index][count][0] = h7
            count = count + 1
            testSamples2HOG[index][count][0] = h8
            count = count + 1
            testSamples2HOG[index][count][0] = h9
            count = count + 1
            testSamples2HOG[index][count][0] = h12
            count = count + 1
            testSamples2HOG[index][count][0] = h22
            count = count + 1
            testSamples2HOG[index][count][0] = h32
            count = count + 1
            testSamples2HOG[index][count][0] = h42
            count = count + 1
            testSamples2HOG[index][count][0] = h52
            count = count + 1
            testSamples2HOG[index][count][0] = h62
            count = count + 1
            testSamples2HOG[index][count][0] = h72
            count = count + 1
            testSamples2HOG[index][count][0] = h82
            count = count + 1
            testSamples2HOG[index][count][0] = h92
            count = count + 1
            testSamples2HOG[index][count][0] = h13
            count = count + 1
            testSamples2HOG[index][count][0] = h23
            count = count + 1
            testSamples2HOG[index][count][0] = h33
            count = count + 1
            testSamples2HOG[index][count][0] = h43
            count = count + 1
            testSamples2HOG[index][count][0] = h53
            count = count + 1
            testSamples2HOG[index][count][0] = h63
            count = count + 1
            testSamples2HOG[index][count][0] = h73
            count = count + 1
            testSamples2HOG[index][count][0] = h83
            count = count + 1
            testSamples2HOG[index][count][0] = h93
            count = count + 1
            testSamples2HOG[index][count][0] = h14
            count = count + 1
            testSamples2HOG[index][count][0] = h24
            count = count + 1
            testSamples2HOG[index][count][0] = h34
            count = count + 1
            testSamples2HOG[index][count][0] = h44
            count = count + 1
            testSamples2HOG[index][count][0] = h54
            count = count + 1
            testSamples2HOG[index][count][0] = h64
            count = count + 1
            testSamples2HOG[index][count][0] = h74
            count = count + 1
            testSamples2HOG[index][count][0] = h84
            count = count + 1
            testSamples2HOG[index][count][0] = h94
            count = count + 1
            
            name = testList2[index].replace(".bmp",".txt",1)
            fileName = "ComputedResults/HOGDescriptors/" + name
            with open(fileName, "a") as myfile:
                myfile.write("\n" +str(h1)+" "+str(h2)+" "+str(h3)+" "+str(h4)+" "+str(h5)+" "+str(h6)+" "+str(h7)+" "+str(h8)+" "+str(h9)+" "+str(h12)+" "+str(h22)+" "+str(h32)+" "+str(h42)+" "+str(h52)+" "+str(h62)+" "+str(h72)+" "+str(h82)+" "+str(h92)+" "+str(h13)+" "+str(h23)+" "+str(h33)+" "+str(h43)+" "+str(h53)+" "+str(h63)+" "+str(h73)+" "+str(h83)+" "+str(h93)+" "+str(h14)+" "+str(h24)+" "+str(h34)+" "+str(h44)+" "+str(h54)+" "+str(h64)+" "+str(h74)+" "+str(h84)+" "+str(h94))
            myfile.close()

# Calculate Positve & Negative Mean Descriptors
print("Computing Mean Descriptor for Positive Training Images...")
for i in range (0, 3780):
    for j in range (0,10):
        positiveMeanDescriptor[0][i][0] = positiveMeanDescriptor[0][i][0] + positiveSamplesHOG[j][i][0]
for i in range (0, 3780):
    positiveMeanDescriptor[0][i][0] = '{0:.2f}'.format(positiveMeanDescriptor[0][i][0] / 10.0)
fileName = "ComputedResults/MeanDescriptors/PositiveSamples/positiveMeanDescriptor.txt"
with open(fileName, "w") as myfile:
    count = 0
    for i in range(0,3780):
        if count == 36:
            count = 0
            myfile.write("\n")
        myfile.write(str('{0:.2f}'.format(positiveMeanDescriptor[0][i][0])) + " ")
        count = count + 1
myfile.close()

print("Computing Mean Descriptor for Negative Training Images...")
for i in range (0, 3780):
    for j in range (0,10):
        negativeMeanDescriptor[0][i][0] = negativeMeanDescriptor[0][i][0] + negativeSamplesHOG[j][i][0]
for i in range (0, 3780):
    negativeMeanDescriptor[0][i][0] = '{0:.2f}'.format(negativeMeanDescriptor[0][i][0] / 10.0)
fileName = "ComputedResults/MeanDescriptors/NegativeSamples/negativeMeanDescriptor.txt"
with open(fileName, "w") as myfile:
    count = 0
    for i in range(0,3780):
        if count == 36:
            count = 0
            myfile.write("\n")
        myfile.write(str('{0:.2f}'.format(negativeMeanDescriptor[0][i][0])) + " ")
        count = count + 1
myfile.close()

# Calculate Eucleidian Distances for Positive & Negative Training Samples
print("\nComputing Eucleidiean Distance for Positive Training Images...")
for i in range(0, 10):
    for j in range(0, 3780):
        positiveSamplesDistance[i][j][0] = positiveMeanDescriptor[0][j][0] - positiveSamplesHOG[i][j][0]

for i in range(0,10):
    distance = 0.0
    for j in range(0, 3780):
        distance = distance + (positiveSamplesDistance[i][j][0] * positiveSamplesDistance[i][j][0])
    distance = (distance)**0.5
    print(trainingPositiveList[i] + "      :   " + str('{0:.2f}'.format(distance)))
    fileName = "ComputedResults/EucleidianDistances/PositiveSamples/positiveEucleidianDistances.txt"
    with open(fileName, "a") as myfile:
        myfile.write(trainingPositiveList[i] + "      :   " + str('{0:.2f}'.format(distance)) + "\n")
    myfile.close()

print("\nComputing Eucleidiean Distance for Negative Training Images...")
for i in range(0, 10):
    for j in range(0, 3780):
        negativeSamplesDistance[i][j][0] = negativeMeanDescriptor[0][j][0] - negativeSamplesHOG[i][j][0]

for i in range(0,10):
    distance = 0.0
    for j in range(0, 3780):
        distance = distance + (negativeSamplesDistance[i][j][0] * negativeSamplesDistance[i][j][0])
    distance = (distance)**0.5
    print(trainingNegativeList[i] + "      :   " + str('{0:.2f}'.format(distance)))
    fileName = "ComputedResults/EucleidianDistances/NegativeSamples/negativeEucleidianDistances.txt"
    with open(fileName, "a") as myfile:
        myfile.write(trainingNegativeList[i] + "      :   " + str('{0:.2f}'.format(distance)) + "\n")
    myfile.close()

# Setting W-Value for training the 2-Class Classifier
print ("\nTraining the 2-Class Classifier with alpha = 0.5 ...")
for i in range(0, 3780):
    wValue[0][i][0] = '{0:.2f}'.format((positiveMeanDescriptor[0][i][0] + negativeMeanDescriptor[0][i][0])/2.0)
wValue[0][3780][0] = '{0:.2f}'.format(0.20)
fileName = "ComputedResults/wValue/wValue.txt"
with open(fileName, "w") as myfile:
    count = 0
    for i in range(0,3781):
        if count == 36:
            count = 0
            myfile.write("\n")
        myfile.write(str('{0:.2f}'.format(wValue[0][i][0])) + " ")
        count = count + 1
myfile.close()

# Training the 2-Class Classifier
check = 0
c = 0
while check == 0:
    check = 1
    c = c + 1
    for i in range(0, 10):
        value1 = 0.0
        value2 = 0.0
        for j in range(0, 3780):
            value1 = value1 + (positiveSamplesHOG[i][j][0] * wValue[0][j][0])
        value1 = value1 + (1.0 * wValue[0][3780][0])
        if value1 <= 0:
            check = 0
            for j in range(0, 3780):
                wValue[0][j][0] = wValue[0][j][0] + (alpha * positiveSamplesHOG[i][j][0])
            wValue[0][3780][0] = wValue[0][3780][0] + (1.0 * alpha)

        for j in range(0, 3780):
            value2 = value2 + (negativeSamplesHOG[i][j][0] * wValue[0][j][0])
        value2 = value2 + (1.0 * wValue[0][3780][0])
        if value2 >= 0:
            check = 0
            for j in range(0, 3780):
                wValue[0][j][0] = wValue[0][j][0] - (alpha * negativeSamplesHOG[i][j][0])
            wValue[0][3780][0] = wValue[0][3780][0] - (1.0 * alpha)

for i in range (0, 3781):
    wFinalValue[0][i][0] = wValue[0][i][0]
fileName = "ComputedResults/wValue/wFinalValue.txt"
with open(fileName, "w") as myfile:
    count = 0
    for i in range(0,3781):
        if count == 36:
            count = 0
            myfile.write("\n")
        myfile.write(str('{0:.2f}'.format(wFinalValue[0][i][0])) + " ")
        count = count + 1
myfile.close()
print ("Number of Iterations while Training: " + str(c))

# Classify Images as Human or Non-Human
print ("\nClassifying Test Images as Human or Non-Human...")
fileName = "ComputedResults/ClassificationResult.txt"
for i in range(0, 5):
    value1 = 0.0
    for j in range(0,3780):
        value1 = value1 + (testSamples1HOG[i][j][0] * wFinalValue[0][j][0])
    value1 = value1 + (1.0 * wFinalValue[0][3780][0])
    if value1 > 0:
        print (testList1[i] + "    : " + "Human Detected")
        with open(fileName, "a") as myfile:
            myfile.write("\n" + testList1[i] + "    : " + "Human Detected")
    else:
        print (testList1[i] + "    : " + "No Human Detected")
        with open(fileName, "a") as myfile:
            myfile.write("\n" + testList1[i] + "    : " + "No Human Detected")
with open(fileName, "a") as myfile:
    myfile.write("\n")

for i in range(0, 5):
    value1 = 0.0
    for j in range(0,3780):
        value1 = value1 + (testSamples2HOG[i][j][0] * wFinalValue[0][j][0])
    value1 = value1 + (1.0 * wFinalValue[0][3780][0])
    if value1 > 0:
        print (testList2[i] + "    : " + "Human Detected")
        with open(fileName, "a") as myfile:
            myfile.write("\n" + testList2[i] + "    : " + "Human Detected")
    else:
        print (testList2[i] + "    : " + "No Human Detected")
        with open(fileName, "a") as myfile:
            myfile.write("\n" + testList2[i] + "    : " + "No Human Detected")
myfile.close()
print ("===========================================================================")
print ("\nProject2.py Interpretd Successfully. Check 'ComputedResults' Directory for all Generated Results\n")
