# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
import glob


class FabricSearchEngine:

    def __init__(self):
        self.database = []

        fileNames= glob.glob("Fabrics/*.jpg")
        for f in  fileNames:
            I=cv2.imread(f)
            IResized = cv2.resize(I, (100, 100)) #resize so all have the same size

            hsv = cv2.cvtColor(IResized,cv2.COLOR_BGR2HSV)
            #Convert to HSV
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            self.database.append({'image':I,"histogram":hist,"Rank":0}) #add the information as a dictionary

            #Show images with a little delay
            cv2.imshow("current Image",I)
            #cv2.waitKey(2)

        sorted(self.database, key=lambda x: x['Rank']) # sorte the entries
    def find_image_color(self,color):
        """ Seaches  the dataBase for images with a specific color"""
        ...

    def find_image_histogram(self,histogram):
        """ Seaches  the dataBase for images with a specific color histogram"""
        ...

if __name__=='__main__':
    searchEngine = FabricSearchEngine()



