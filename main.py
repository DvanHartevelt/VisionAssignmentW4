
import cv2
import numpy as np
from matplotlib import pylab as plt


def main():
    # Import the first picture
    imgShapes = cv2.imread("Resources/blocks.png", 0)

    #plt.hist(imgShapes.ravel(), 256, [0, 256])
    #plt.show()

    # 200 is a good threshold
    (thres, imgShapesBW) = cv2.threshold(imgShapes, 200, 255, cv2.THRESH_BINARY_INV)

    """
    #Row-by-row connected components methods
    """
    def rowByRow(img):
        height = img.shape[0]
        width = img.shape[1]

        labeledMatrix = np.zeros_like(img)
        shapeOrdinal = 0
        equivalencyList = [0]

        #first sweep
        for y in range(height):
            for x in range(height):
                if img[y, x] == 255: #maybe change a bit to include internal (empty) object?
                    #above value:
                    if y==0: above = 0
                    else: above = labeledMatrix[y-1, x]

                    #left value
                    if x ==0: left = 0
                    else: left = labeledMatrix[y, x - 1]

                    #determining label
                    if above != 0 and left != 0:
                        labeledMatrix[y, x] = min(above, left)
                        equivalencyList[max(above, left)] = min(above, left, equivalencyList[left], equivalencyList[above])
                    elif left != 0:
                        labeledMatrix[y, x] = left
                    elif above != 0:
                        labeledMatrix[y, x] = above
                    else: #above == 0 and left == 0:
                        shapeOrdinal += 1
                        labeledMatrix[y, x] = shapeOrdinal
                        equivalencyList.append(shapeOrdinal)

        # creating a messed up lookup table
        lookup = np.zeros(shapeOrdinal+1)
        reverseLookup = np.unique(np.array(equivalencyList))
        for i in range(len(reverseLookup)):
            lookup[reverseLookup[i]] = i

        # second sweep, changing all labels to be the appropriate
        for y in range(height):
            for x in range(width):
                labeledMatrix[y, x] = lookup[equivalencyList[labeledMatrix[y, x]]]

        return len(reverseLookup), labeledMatrix

    def colourIn(img):
        coloursRGB = np.array(
            [[255,   0,   0],  # blue
             [  0, 255,   0],  # green
             [  0,   0, 255],  # red
             [255, 255,  51],  # cyan
             [255,   0, 255],  # magenta
             [  0, 128, 255],  # orange
             [  0, 255, 255]]) # yellow

        imgNew = np.zeros([img.shape[0], img.shape[1], 3], np.uint8)

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img[y, x] > 0 and img[y, x] < len(coloursRGB) + 1:
                    imgNew[y, x] = coloursRGB[img[y, x] - 1]
                else:
                    imgNew[y, x] = [0, 0, 0]

        return imgNew

    nrofObjects, Row = rowByRow(imgShapesBW)
    RowColoured = colourIn(Row)


    cv2.imshow("original", imgShapes)
    cv2.imshow("Mat", imgShapesBW)
    cv2.imshow("Row by row", RowColoured)

    #cv2.waitKey(0)

    """
    # feature recognision
    """
    def pullSingleObject(img, value):
        imgNew = np.zeros_like(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img[y, x] == value: imgNew[y, x] = 255
                else: imgNew[y, x] = 0

        return imgNew

    def getArea(img):
        return img.sum() // 255

    def flood(img):
        imgFlood = img.copy()

        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

        cv2.floodFill(imgFlood, mask, (0, 0), 255)
        imgFloodInv = cv2.bitwise_not(imgFlood)

        imgOut = img | imgFloodInv

        return imgOut

    def getCircumference(img):
        flooded = flood(img)
        erode = cv2.morphologyEx(flooded, cv2.MORPH_ERODE, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
        circumMat = np.bitwise_xor(img, erode)

        return circumMat.sum() // 255

    def getCompactness(img):
        return 4*3.1415*getArea(img) / (getCircumference(img)**2)



    for i in range(1, nrofObjects):
        obj = pullSingleObject(Row, i)
        print(f"~~~~~~~~~~ Object nr {i} ~~~~~~~~~~")
        print(f"Area of object {i} is {getArea(obj)} pixels")
        print(f"Circumference of object {i} is {getCircumference(obj)} pixels.")
        print(f"Compactness of object {i} is {getCompactness(obj)} per pixel.")
        print(" ")
        cv2.imshow(f"Object {i}", obj)

    pass


if __name__ == "__main__":
    main()

