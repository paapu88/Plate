"""
Returns rectangles containing possible licence plates from a given image
for parameters see
http://docs.opencv.org/3.0-beta/modules/objdetect/doc/cascade_classification.html

python3 rekkariDetectionSave.py 0-751.jpg
(there must be trained classifiier file, assumption is 'rekkari.xml' )
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
import sys
import os


class DetectPlate():
    """
    Returns rectangles containing possible licence plates from a given image
    for parameters see
    http://docs.opencv.org/3.0-beta/modules/objdetect/doc/cascade_classification.html

    python3 rekkariDetectionSave.py 0-751.jpg
    (there must be trained classifiier file, assumption is 'rekkari.xml' )
"""
    def __init__(self, trainedHaarFileName='rekkari.xml',
                npImage=None, imageFileName=None,
                detectFactor=5, scaleFactor=1.03, minSize=(5,18),
                colorConversion=cv2.COLOR_BGR2GRAY,
                imageXfactor=None, imageYfactor=None):

        if not os.path.isfile(trainedHaarFileName):
            raise FileNotFoundError('you need cascade training file for cv2.CascadeClassifier')
        self.trainedHaarFileName = trainedHaarFileName
        self.cascade = cv2.CascadeClassifier(trainedHaarFileName)
        self.imageFileName = imageFileName
        self.detectFactor = detectFactor
        self.scaleFactor = scaleFactor
        self.minSize = minSize
        self.colorConversion = colorConversion
        self.img = npImage  # image as numpy array
        self.gray = None
        self.plates = None
        self.rotation_angles = None # for each plate, an optimal rotation angle of the whole image
        self.rotation_centers = None # for each plate, center of rotation
        self.npPlates = []  #images of plate(s) as array of numpy arrays

    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ read image from file """
        self.img = cv2.imread(imageFileName)
        self.img = cv2.cvtColor(self.img, colorConversion)
        
    def image2Plates(self):
        """ from image, produce rectangle(s) that contain possible plate, output as list of rectanges"""
        if self.img is None:
            if not os.path.isfile(self.imageFileName):
                raise FileNotFoundError('NO imagefile with name: ' + self.imageFileName)
            self.img = cv2.imread(self.imageFileName)
        if len(self.img.shape)> 2:
            self.gray = cv2.cvtColor(self.img.copy(), self.colorConversion)
        else:
            self.gray = self.img.copy()
        rectangles = self.cascade.detectMultiScale(self.gray, self.scaleFactor, self.detectFactor, minSize=self.minSize)
        plates = []
        for [x,y,w,h] in rectangles:
            #print("xywhh", x,y,w,h)
            plates.append([x,y,w,h])
        self.plates = plates

    def image2PlateNumpyArrays(self, rotate=True):
        """ if rotate is true, try to make plates horizontal
        from image, produce rectangle(s) that contain possible plate,
        output as list of numpy array(s) representing gray scale images(s) about possible plates"""
        if self.img is None:
            if not os.path.isfile(self.imageFileName):
                raise FileNotFoundError('NO imagefile with name: ' + self.imageFileName)
            self.img = cv2.imread(self.imageFileName)
        self.npPlates = []
        if len(self.img.shape)> 2:
            self.gray = cv2.cvtColor(self.img.copy(), self.colorConversion)
        else:
            self.gray = self.img.copy()
        self.plates = self.cascade.detectMultiScale(self.gray, self.scaleFactor, self.detectFactor, minSize=self.minSize)
        for [x,y,w,h] in self.plates:
            if rotate:
                gray = self.rotatePlate(rectangle=[x,y,w,h])
            else:
                gray = self.gray
            roi_gray = self.gray[y:y+h, x:x+w]
            self.npPlates.append(roi_gray)

    def getNpPlates(self):
        """ give found plate(s) as array of numpy arrays"""
        self.image2PlateNumpyArrays()
        return self.npPlates

    def getGray(self):
        return self.gray.copy()

    def showPlates(self, rotate=True):
        """show plates """
        print("showing")
        clone = self.getGray()

        if self.plates is not None:
            for i, [x,y,w,h] in enumerate(self.plates):
                rows,cols = clone.shape
                cv2.rectangle(clone,(x,y),(x+w,y+h),(255,0,0),3)
        #plt.imshow(clone, cmap = 'gray', interpolation = 'bicubic')
        #if len(self.plates) > 0:
        plt.imshow(clone)
        plt.show()

    def rotatePlate(self, rectangle, minAng=-10, maxAng=10):
        """rotate single plate to make letters and digits horizontal"""
        from scipy import interpolate
        img = cv2.medianBlur(self.getGray(),1)
        thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

        angles_iter = np.linspace(minAng, maxAng, int(round(abs(maxAng)+abs(minAng)+1)))
        angles_many_iter = np.linspace(minAng, maxAng, 1+10*int(round(abs(maxAng)+abs(minAng))))
        clone = thresh
        rows,cols = clone.shape
        [x,y,w,h] = rectangle
        #print("xywh",x,y,w,h)
        #rotate image
        weights = []; angles=[]
        for angle in angles_iter:
            M = cv2.getRotationMatrix2D((x+0.5*w,y+0.5*h),angle,1)
            dst = cv2.warpAffine(clone,M,(cols, rows))
            hist=np.sum(dst[y:y+h,x:x+w],axis=1)[::-1]
            angles.append(angle)
            weights.append(np.var(hist))

        f = interpolate.interp1d(angles, weights)
        faas = f(angles_many_iter)
        angle=angles_many_iter[np.argmax(faas)]
        M = cv2.getRotationMatrix2D((x+0.5*w,y+0.5*h),angle,1)
        dst = cv2.warpAffine(clone,M,(cols,rows))
        return dst

    def getRotationAnglesCenters(self, minAng=-10, maxAng=10):
        """rotate image so that we get weighted maximum of sum over x-sum-values of the plate"""
        from filterImage import FilterImage
        from scipy import interpolate
        self.image2Plates()
        angles_iter = np.linspace(minAng, maxAng, int(round(abs(maxAng)+abs(minAng)+1)))
        angles_many_iter = np.linspace(minAng, maxAng, 1+10*int(round(abs(maxAng)+abs(minAng))))
        img = cv2.medianBlur(self.getGray(),1)
        thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        clone = thresh
        rows,cols = clone.shape
        self.rotation_angles = []
        self.rotation_centers = []
        for i, [x,y,w,h] in enumerate(self.plates):
            #print("xywh",x,y,w,h)
            weights = []; angles=[]
            #rotate image
            for angle in angles_iter:
                M = cv2.getRotationMatrix2D((x+0.5*w,y+0.5*h),angle,1)
                dst = cv2.warpAffine(clone,M,(cols, rows))
                hist=np.sum(dst[y:y+h,x:x+w],axis=1)[::-1]
                #print(angle, np.var(hist))
                angles.append(angle)
                weights.append(np.var(hist))

            f = interpolate.interp1d(angles, weights)
            faas = f(angles_many_iter)
            angle=angles_many_iter[np.argmax(faas)]
            self.rotation_angles.append(angle)
            self.rotation_centers.append((x+0.5*w,y+0.5*h))
            M = cv2.getRotationMatrix2D((x+0.5*w,y+0.5*h),angle,1)
            dst = cv2.warpAffine(clone,M,(cols,rows))
            cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 5)
            plt.imshow(dst[y:y+h,x:x+w], cmap = 'gray', interpolation = 'bicubic')
            plt.show()

    def get_rotation_angles(self):
        return self.rotation_angles

    def get_rotation_centers(self):
        return self.rotation_centers

    def writePlates(self, name=None):
        """debugging: write each plate to a separate file"""
        self.image2Plates()
        if self.plates is not None:
            print("writing to file", name, len(self.plates))
        else:
            print("NO PLATES FOUND")
            return
        for i, [x,y,w,h] in enumerate(self.plates):
            clone = self.getGray()
            rows,cols = clone.shape
            if self.rotation_angles is not None:
                M = cv2.getRotationMatrix2D(self.rotation_centers[i],self.rotation_angles[i],1)
                clone = cv2.warpAffine(clone,M,(cols,rows))
                print("xywhWrite",x,y,w,h,self.rotation_centers[i],self.rotation_angles[i])
            roi_gray = clone[y:y+h, x:x+w]
            if name is None:
                cv2.imwrite('orig'+str(i)+'-'+'plate'+'-'+sys.argv[1]+'.tif', roi_gray)
            else:
                cv2.imwrite('orig'+str(i)+'-'+name, roi_gray)
            # make height of figure bigger for neural network
            if h < 0.5*w:
                h_orig = h
                h = int(round(0.5 * w))
                change = h - h_orig
            y = int(round(y - change/2))
            ## check that we are image
            if (h+y)> clone.shape[0]:
                y = clone.shape[0] - h
            if y < 0:
                y = 0
            #print("x,y,w,h",x,y,w,h)
            roi_gray = clone[y:y+h, x:x+w]
            if name is None:
                cv2.imwrite(str(i)+'-'+'plate'+'-'+sys.argv[1]+'.tif', roi_gray)
            else:
                cv2.imwrite(str(i)+'-'+name, roi_gray)

if __name__ == '__main__':
    import sys, glob

    for imageFileName in glob.glob(sys.argv[1]):
        #app = DetectPlate(imageFileName=imageFileName,
        #                trainedHaarFileName='./rekkari.xml',
        #                detectFactor=1)
        app = DetectPlate(imageFileName=imageFileName,
                        trainedHaarFileName='/home/mka/PycharmProjects/Rekkari/Training_singlePlate_withChar/classifier/cascade.xml',
                         detectFactor=1)
        #app.getRotationAnglesCenters()

        #app.showPlates()
        app.writePlates(name='plateOnly-'+imageFileName)
