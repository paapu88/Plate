"""
From an image get a list of possible licence plate strings.
The first one has the highest probability.

Usage:
    python3 image2characters.py "plate.jpg"

    or from other python modules:

    from image2characters import image2Characters
    app = image2Characters(npImage=myNParr)
    app.getChars()

"""
import sys

from Plate.rekkariDetectionSave import DetectPlate
from Plate.filterImage import FilterImage
from Plate.filterCharacterRegions import FilterCharacterRegions
from Plate.initialCharacterRegions import InitialCharacterRegions
from Plate.video2images import VideoIO
# from myTesseract import MyTesseract
from Plate.myClassifier import Classifier
from Plate.detect_oneImage import Detect
from Plate.mydetect import MyDetect
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class image2Characters():
    """ 
    from an input file or yuv numpy array get an array of strings 
    representing characters in (a) number plate(s) 
    """
    def __init__(self, npImage=None):
        self.img = npImage  # image as numpy array

    def setNumpyImage(self, image, imageType=None):
        """
        set image from numpy array
        """
        self.img = image
        if imageType is not None:
            if "RGB24FrameView" in str(imageType):
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
            else:
                print ("mok WARNING: color conversion not implemented: ", imageType)


    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debuggin image can be read from file also"""
        self.img = cv2.imread(imageFileName)
        try:
            self.img = cv2.cvtColor(self.img, colorConversion)
        except:
            self.img = cv2.imread(imageFileName, 0)

    def getChars(self):
        """
        From Image to list of strings, representing characters of (a) number plate(s)
        """
        from Plate import __path__ as module_path
        

        allChars = []
        app1 = DetectPlate(trainedHaarFileName=module_path[0]+'/rekkari.xml',
                           npImage=self.img)

        plates = app1.getNpPlates()
        #print("mok shape ",self.img.shape, len(plates))

        #app1.showPlates()
        #app1.writePlates(name='plateOnly-'+sys.argv[1])
        #print(file+' number of plates found '+ str(len(plates)))
        for plate in plates:
            myChars = []
            myProb = []
            # from a plate image to list of six-rectangles
            #app2 = FilterImage(npImage=plate)
            #plate = app2.filterOtsu()
            app3 = FilterCharacterRegions(npImage=plate)
            platesWithCharacterRegions = app3.imageToPlatesWithCharacterRegions()
            app5 = Classifier(npImage=plate)
            #app3.showImage()
            app5.defineSixPlateCharactersbyLogReg(platesWithCharacterRegions)
            plate_chars, plate_probability = app5.getFinalStrings()
            myChars = myChars + plate_chars
            #app3.showAllRectangles()

            if plate_probability is None:
                plate_probability = 0.0
            myProb = myProb + plate_probability

            # sort so that most probable comes first, return None if we fail
            try:
                myProb, myChars = zip(*sorted(zip(myProb, myChars)))
                allChars.append(myChars[::-1])
                #allChars = allChars + myChars[::-1]
            except:
                pass

        if len(plates) == 0:
            # no plate found
            print("no plate found")
            return None

        if len(allChars) == 0:
            # if there are no likely plates
            print ("possible plate found, but no characters assigned")
            return None
        else:
            #print("PLATE(S): ", allChars)
            #app1.showPlates()
            return allChars

    def double_plate(self, whole_img, rect):
        """ returns the plate double sized plate image
        resizing is NOT by scaling but by taking more background
        """
        height_whole, width_whole = whole_img.shape[:2]
        [x,y,w,h] = rect        
        dw = int(round(1*w))
        dh = int(round(1*h))
        #dw=0
        #dh=0
        # dh = h
        try:
            return whole_img[y-dh:y+h+dh, x-dw: x+w+dw]
        except:
            return None


    def getCharsByNeuralNetworkRaw(self):
        """
        get the caharcters of a plate by neural network,
        the image is first carefully filtered, so it only contains the actual plate
        """
        from Plate import __path__ as module_path
        
        app1 = DetectPlate(trainedHaarFileName=module_path[0]+'/rekkari.xml',
                           npImage=self.img)
        detect_exe = module_path[0]+'/detect.py'
        app1.image2Plates() # get rectangles
        print("PLATES", app1.plates)
        for plate in app1.plates:
            double_plate= self.double_plate(self.img, plate)
            #plt.imshow(double_plate)
            #plt.show()            
            cv2.imwrite('double.png', double_plate)
            command = "python3 "+detect_exe+" double.png weights.npz"
            os.system(command)

    def getCharsByNeuralNetwork(self):
        """
        get the charcters of a plate by neural network,
        """
        from Plate import __path__ as module_path
        allChars = []
        
        app1 = DetectPlate(trainedHaarFileName=module_path[0] + '/rekkari.xml',
                           npImage=self.img)
        app1.image2Plates()  # get rectangles
        #print("PLATES", app1.plates)
        app2 = MyDetect()
        for plate in app1.plates:
            double_plate = self.double_plate(self.img, plate)
            #plt.imshow(double_plate)
            #plt.show()
            app2.setNumpyImage(double_plate)
            #print("NN plate result:", app2.get_result())
            allChars.append( app2.get_result())
        if not allChars:
            # if empty list (no plates found) , return none
            return None
        else:
            return allChars 

# cd ~/Python/Plate/Test
# python3 image2characters.py /home/mka/Videos/10330101/22550005.MOV
if __name__ == '__main__':
    import sys, glob
    
    resultNoNN = []  # classical machine learning result
    resultNN = []    # deep neural network result
    app = image2Characters()

    """
    files=glob.glob(sys.argv[1])
    # print(files)
    if len(files)==0:
        raise FileNotFoundError('no files with search term: '+sys.argv[1])
    
    for file in files:
        
        video2images = VideoIO(videoFileName=file,
                           stride=24,
                           colorChange=cv2.COLOR_RGB2GRAY)
        video2images.readVideoFrames(videoFileName=file)
        for image in video2images.getFrames():
            app.setNumpyImage(image=image)
            resultNoNN = resultNoNN + app.getChars()
            resultNN = resultNN + app.getCharsByNeuralNetwork()
        """

    app.setImageFromFile(sys.argv[1])
    resultNoNN = app.getChars()
    print("Classical ML result: ", resultNoNN)
    resultNN = app.getCharsByNeuralNetwork()
    print("Deep neural network result: ", resultNN)
