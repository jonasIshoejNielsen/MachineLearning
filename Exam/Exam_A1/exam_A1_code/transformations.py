import cv2
import math
import numpy as np


        
def to_homogeneous(points):
    """ Convert column vectors to row vectors"""
    if len(points.shape) == 1:
        points = points.reshape((*points.shape, 1))
    return np.vstack((points, np.ones((1, points.shape[1]))))

def to_euclidean(points, dimension=2):
    """
    function convert homogeneous vector 2 a point of dimension dimension
    """
    return points[:dimension] / points[dimension]

class Transformation:
    """Handles the transformation"""
    def __init__(self, M):
        self.M = M

    def __str__(self):
         return "{0}".format(self.M)
     
    def inverseOfM(self):
        """This function return the inverse of array m"""
        return Transformation(np.linalg.inv(self.M))
    
    def transpose(self):
        """ This function return the inverse of array m"""
        return Transformation(self.M.T)
    
    def pseudoinverse(self):
        """returns (self.T * self).inverse * self.T
        todo check if linalg.pinv is acceptable
        code that should work if not singular A.T*A
            return Transformation(np.linalg.inv(self.M.T.dot(self.M)).dot(self.M.T))
            # or 
            MT=self.transpose()
            return self.transpose().combine(self).inverseOfM().combine(self.transpose())
        """
        return Transformation(np.linalg.pinv(self.M))
        
    def combine (self, m):
        """ return self * m"""
        return Transformation(np.dot(self.M, m.M,))
    
    def transformationPointApply_3x3(self, point):
        """
        This function apply the 3x3 transformation array to a 2 dimensional euclidean point
        :param point: [[x]; [y]].
        output: [[x]; [y]].
        """
        return to_euclidean(np.dot(self.M, to_homogeneous(point)))
 
    def transformationMultiplePointsApply_3x3(self, points):
        """
        This function apply the 3x3 transformation array to a tuple of (x list ,y list)
        :param points: [[x1, x2,...,xn]; [y1, y2,...,yn]].
        output: [['x1, 'x2,..., 'xn]; ['y1, 'y2,..., 'yn]]
        """
        outputX = []
        outputY = []
        for i in range(len(points[0])):
            x = self.transformationPointApply_3x3(np.array([[points[0][i]], [points[1][i]]]))
            outputX.append(x[0][0])
            outputY.append(x[1][0])
        return [outputX, outputY]

    def transformationMultipleColumnPointsApply_3x3(self, points):      #todo test
        """
        This function apply the 3x3 transformation array to a tuple of (x list ,y list)
        :param points: [[x1, y1],[x2, y2], ..., [xn, yn]].
        output: [['x1, 'x2,..., 'xn]; ['y1, 'y2,..., 'yn]]
        """
        outputX = []
        outputY = []
        for p in points:
            x = self.transformationPointApply_3x3(np.array([p, p]))
            outputX.append(x[0][0])
            outputY.append(x[1][0])
        return [outputX, outputY]

    def transformationImageApply_3x3_My(self, image, pos, size):
        """
        This function apply the 3x3 transformation array to the image
        :param image: Input image.
        """
        rows, cols, _ = image.shape
        inverseEq = self.pseudoinverse()
        result = np.zeros(image.shape,  dtype=np.uint8)
        distValue = lambda cy, cx, prod: prod * image[cy][cx]
        print(pos[0], pos[0]+size[0])
        print(pos[1], pos[1]+size[1])
        for y in range(max(0, pos[1]),min(rows, pos[1]+size[1])):
            for x in range(max(0, pos[0]),min(cols, pos[0]+size[0])):
                cords = inverseEq.transformationPointApply_3x3(np.array([[x],[y]]))
                if cords[0]>=0 and cords[1]>=0 and math.ceil(cords[1]) < rows and math.ceil(cords[0])<cols:
                    dx, dy = cords[0] - math.floor(cords[0]), cords[1] - math.floor(cords[1])
                    result[y][x] = distValue(math.floor(cords[1]), math.floor(cords[0]), (1-dx)*(1-dy)) + distValue(math.floor(cords[1]), math.ceil(cords[0]), (dx)*(1-dy)) + distValue(math.ceil(cords[1]), math.floor(cords[0]), (1-dx)*(dy)) + distValue(math.ceil(cords[1]), math.ceil(cords[0]), dx*dy)
        return result
    
    def transformationImageApply_3x3(self, image):
        """
        This function apply the 3x3 transformation array to the image
        :param image: Input image.
        """
        rows, cols, _ = image.shape
        return cv2.warpPerspective(image, self.M, (cols, rows))


def identity_3x3():
    """ Return a 3x3 identity matrix"""
    return Transformation(np.array ([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

def combineMultiple (lst):
    """ take list [m1,m2,...,mn] return m1*m2*....*mn"""
    res = identity_3x3()
    for m in lst :
        res = res.combine(m)
    return res

def transformation_3x3(theta=0., t=(0., 0.), s=(1.,1), p=(0., 0., 1.)):  # todo scaling might be wrong
    """
    This function return a 3x3 homogeneous transformation array with parameters for rotation/theta, translation/t, scaling/s, and perspective warping/p.
    :param theta: Rotation angle.
    :param t: Translation (x, y).
    :param s: Scaling.
    :param p: Perspective parameters (a, b, c).      //makes it not affine
    order: rotation-->scaling-->translation, not sure about skew8)
    """
    M= np.array ([[s[0] * math.cos(theta), s[0]*(-math.sin(theta)), t[0]],
                [s[1] * math.sin(theta), s[1] * math.cos(theta), t[1]],
                [p[0], p[1], p[2]]])
    return Transformation(M)

def transformation_arbitrary_3x3(theta=0., t=(0., 0.)):
    """
    This function return a 3x3 homogeneous transformation array that rotate around a point
    :param theta: Rotation angle.
    :param t: Translation (x, y)
    """
    trans = transformation_3x3(0,t)
    tInv = trans.inverseOfM()
    return combineMultiple([tInv,transformation_3x3(theta), trans] )


"Affine"
def affineEquationForOnePoint(xi, yi):
    return np.array([[xi, yi, 1, 0, 0, 0], [0, 0, 0, xi, yi, 1]]) 

def stackIfAble(equations, eq):
    if(equations.shape[0] == 0):
        return eq
    else :
        return np.vstack((equations, eq))

def affineEquations(points_source, points_target):
    equations = np.array([])
    target    = np.array([])
    for i in range(len(points_source[0])):
        eq =  affineEquationForOnePoint(points_source[0][i], points_source[1][i])
        equations = stackIfAble(equations, eq)
        target = stackIfAble(target, np.array([[points_target[0][i]], [points_target[1][i]]]))
    return (Transformation(equations), Transformation(target))

def learn_affine(points_source, points_target): 
    """
    : param points_source: list of [[x1,x2,x3,...,xn],[y1,y2,y3,...,yn]]
    : param points_target: list of [[x1,x2,x3,...,xn],[y1,y2,y3,...,yn]]
    """
    equations, target = affineEquations(points_source, points_target)
    res = equations.pseudoinverse().combine(target)
    return Transformation(stackIfAble(res.M.reshape((2, 3)), np.array([0,0,1])))
    
def pointsToCorrecSyntax(points):
    """ from [[x1,y1] ,[x2,y2],[x3,y3],...,[xn, yn]] to [[x1,x2,x3,...,xn],[y1,y2,y3,...,yn]]"""
    
    resX,resY = [],[]
    for i in range(len(points)):
        resX.append(points[i][0])
        resY.append(points[i][1])
    return np.array([resX, resY])

