import transformations as trans
import numpy as np

# Write tests here
def test():
    input = [[2,3,4,5,6], [5,7,9,11,13]]
    output = [[2,3,4,5,6], [15,21,27,33,39]]
    res =trans.learn_affine(input, output)
    print(res)
    equal = True
    for i in range(len(input[0])):
        pointOut = np.array([[output[0][i]], [output[1][i]]])
        pointIn = np.array([[input[0][i]], [input[1][i]]])
        pointRes = res.transformationPointApply_3x3(pointIn)
        equal = equal & (0.001 > (pointRes[1] - pointOut[1])**2)
    print(equal)



def reportTest_2_1():
    eq = trans.transformation_3x3(32.0, (2., 3.), (3., 2.) )
    input = [[2, 5, 11], [5, 2,11]]
    output = eq.transformationMultiplePointsApply_3x3(input)
    
    resEq =trans.learn_affine(input, output)
    outputTest = resEq.transformationMultiplePointsApply_3x3(input)
    print("Test if perform same transformation:")
    equal = True
    for i in range(len(input)):
        equal = equal & (0.001 > (output[0][i] - outputTest[0][i])**2)& (0.001 > (output[1][i] - outputTest[1][i])**2)
    print(equal)
    
    
    print("Test if same transformation matrix:")
    equal = True
    for x in range(3):
        for y in range(3):
            equal = equal & (0.001 > (eq.M[x][y] - resEq.M[x][y])**2)
    print(equal)


#test()
reportTest_2_1()