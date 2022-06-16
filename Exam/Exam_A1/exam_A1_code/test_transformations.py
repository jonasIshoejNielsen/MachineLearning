import cv2
from transformations import *

# Write tests here




def test():
    print("Default")
    print (transformation_3x3(0., (0.,0.), 1.0, (0., 0.0, 1.)) )
    print("Scale")
    print (transformation_3x3(0., (0.,0.), 3.0))
    print (transformation_3x3(0., (0.,0.), 0.0))
    print("Rotation")
    print (transformation_3x3(10.))
    print (transformation_3x3(20.))
    print (transformation_3x3(20., (0.,0.), (2.0, 1.)))
    print("Translation")
    print (transformation_3x3(0., (1.,2.)))
    print (transformation_3x3(0., (1.,2.), (3.0, 2.)))
    print (transformation_3x3(4., (1.,2.), 3.0, 4.))
    print("Skew")
    print (transformation_3x3(0., (0.,0.), (1.0,1.0), (1., 0.0, 0.)))
    print (transformation_3x3(0., (0.,0.), (1.0,1.0), (0., 2.0, 0.)))
    print (transformation_3x3(0., (0.,0.), (1.0,1.0), (0., 0.0, 3.)))
    print (transformation_3x3(0., (0.,0.), (1.0,1.0), (1., 2.0, 3.)))
    print("All")
    print (transformation_3x3(5., (6.,7.), (4.0,4.0), (1., 2.0, 3.)))
    print("inv")
    k=transformation_3x3(5., (6.,7.), (4.0, 4.0), (1., 2.0, 3.))
    print(k.inverseOfM())
    print(k.inverseOfM())
    print(k.inverseOfM())
    print("identity")
    print(identity_3x3())
    print(k.inverseOfM().combine(k))
    print ("point transform")
    point = np.array([[12],[55]])
    print("original point: " + str(point))
    print(transformation_3x3(0., (0.,0.), (3.0,3.0)).transformationPointApply_3x3(point))
    points = np.array([[12, 25, 33, 1],[55, 4, 6, 2]])
    print(transformation_3x3(0., (0.,0.), (3.0,3.0)).transformationMultiplePointsApply_3x3(points))
    pointsC = np.array([[12,55], [25,4],[33,6],[1,2]])
    print(transformation_3x3(0., (0.,0.), (3.0,3.0)).transformationMultipleColumnPointsApply_3x3(pointsC))
    
    print("combine")
    print(transformation_3x3(0., (1.,2.)).combine(transformation_3x3(0., (3.,3.))))
    print("combine multiple")
    print(combineMultiple([transformation_3x3(0., (1.,2.)), transformation_3x3(0., (3.,3.)),transformation_3x3(0., (3.,3.))]))
    print("pseudoinverse")
    print(transformation_3x3(0., (1.,2.)).pseudoinverse())
    print(transformation_3x3(0., (1.,2.)).pseudoinverse().pseudoinverse())
    
    print("apply to image tests")
    image = cv2.imread('inputs/dipsy.jpg')
    cv2.imshow("ImageTest1", transformation_3x3(math.pi/8).transformationImageApply_3x3(image))
    cv2.imshow("ImageTest2", transformation_3x3(math.pi/8, (100.,0.)).transformationImageApply_3x3(image))
    cv2.imshow("ImageTest3", transformation_arbitrary_3x3(math.pi/8, (100.,0.)).transformationImageApply_3x3(image))

    if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()



def image_test():
    
    t1 = transformation_3x3(0., (-300.,-250.)).combine(transformation_3x3(0., (0.,0.), (3.0,2.))).combine(transformation_3x3(math.pi/8))
    print( transformation_3x3(0., (-300.,-250.)))
    print( transformation_3x3(0., (0.,0.), (3.0,2.)))
    print( transformation_3x3(0., (-300.,-250.)).combine(transformation_3x3(0., (0.,0.), (3.0,2.))))
    print( transformation_3x3(0., (0.,0.), (3.0,2.)).combine(transformation_3x3(0., (-300.,-250.))))
    print()
    print( transformation_3x3(math.pi/8))
    print( t1)
    t0 = transformation_3x3(0., (-300.,-250.)).combine(transformation_3x3(0., (0.,0.), (3.0,2.)))
    print(t0.combine(transformation_3x3(math.pi/8)))
    
    t2 = transformation_3x3(math.pi/4).combine(transformation_3x3(0., (-150.,-200.), (1.5, -1.5))).combine(transformation_3x3(-math.pi/2))
    
    image = cv2.imread('inputs/dipsy.jpg')
    timg1, timg2 = t1.transformationImageApply_3x3(image), t2.transformationImageApply_3x3(image)
    
    
    np.save("outputs/image1.npy", timg1)
    np.save("outputs/image2.npy", timg2)
    
    #test by load
    load1 = np.load("outputs/image1.npy")
    load2 = np.load("outputs/image2.npy")
    cv2.imshow("Image", image)
    cv2.imshow("ImageTest1", timg1)
    cv2.imshow("ImageTest2", timg2)
    cv2.imshow("loadTest1", load1)
    cv2.imshow("loadTest2", load2)
    

    if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()



def point_test():
    
    t1 = transformation_3x3(0., (-300.,-250.)).combine(transformation_3x3(0., (0.,0.), (3.0,2.))).combine(transformation_3x3(math.pi/8))
    print(t1)
    print(t1.transformationPointApply_3x3(np.array([[12],[55]])))
    

#test()   
#image_test()
point_test()

