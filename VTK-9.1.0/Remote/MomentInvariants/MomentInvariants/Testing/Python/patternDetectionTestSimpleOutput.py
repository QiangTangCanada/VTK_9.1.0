"""
    we cutout a random point from the given dataset,
    rotate it, scale it, multiply with a constant, add a constant
    and look for it.
    The test is considered successful if the original position is the global maximum of similarity.
    Inaccuracies come from the resolution of the dataset and the pattern as well as the coarse resolutions.
    I chose the numbers to perfectly match and the rotation angle to be a multiple of 90 degree. So that the sampling error is suppressed.
    """
from __future__ import absolute_import, division, print_function

import vtk
try:
    import numpy as np
except ImportError:
    print("Numpy (http://numpy.scipy.org) not found.")
    print("This test requires numpy!")
    from vtk.test import Testing
    Testing.skip()
import sys
import os, shutil
import math
from patternDetectionHelper import *
import random

from vtk.util.misc import vtkGetDataRoot
VTK_DATA_ROOT = vtkGetDataRoot()

numberOfIntegrationSteps = 0
order = 2
angleResolution = 8
eps = 1e-2
numberOfMoments = 5 #21 11 6 5
nameOfPointData = "values"

def doTest(filePath):
    if not os.path.isfile(filePath):
      print("File: " + filePath + " does not exist.")
      sys.exit(1)
    else:

        (dataset, bounds, dimension) = readDataset(filePath)
        npBounds = np.array(bounds)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(dataset)
        writer.SetFileName('/Users/bujack/Documents/moments/output/data.vti')
        writer.SetDataModeToAscii()
        writer.Write()

        if (dataset.GetDimensions()[0]-1) % (numberOfMoments-1) != 0:
          print("Warning! The numberOfMoments will not produce a true coarse version of the input dataset.")

        # produce coarse grid where the moments will be computed
        if dimension == 3:
            momentDataCoarse = createCoarseDataset(bounds, numberOfMoments, numberOfMoments, numberOfMoments)
        else:
            momentDataCoarse = createCoarseDataset(bounds, numberOfMoments, numberOfMoments, 0)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(momentDataCoarse)
        writer.SetFileName('/Users/bujack/Documents/moments/output/coarse.vti')
        writer.SetDataModeToAscii()
        writer.Write()

        # produce pattern
        if dimension == 3:
          radiusPattern = np.amin(npBounds[1::2]-npBounds[0::2]) / 10
          offset = max(radiusPattern, momentDataCoarse.GetSpacing()[0], momentDataCoarse.GetSpacing()[1], momentDataCoarse.GetSpacing()[2])
          patternPosition = [random.uniform(bounds[2*i]+offset, bounds[2*i+1]-offset) for i in range(3)]
        else:
          radiusPattern = np.amin((npBounds[1::2]-npBounds[0::2])[:-1]) / 10
          offset = max(radiusPattern, momentDataCoarse.GetSpacing()[0], momentDataCoarse.GetSpacing()[1])
          patternPosition = [random.uniform(bounds[2*i]+offset, bounds[2*i+1]-offset) for i in range(3)]
          patternPosition[2] = bounds[5]

        patternPosition = momentDataCoarse.GetPoint(momentDataCoarse.FindPoint(patternPosition))

#        patternPosition= (0.5, 0.75, 0.0)
        pattern = cutoutPattern(dataset, dimension, patternPosition, radiusPattern)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(pattern)
        writer.SetFileName('/Users/bujack/Documents/moments/output/pattern.vti')
        writer.SetDataModeToAscii()
        writer.Write()

        # rotate, multiply constant, change size, add constant
        rotationAngle = 2 * math.pi * 0.25 * random.randint(0,3)
        factorOuter = random.uniform(0.5,2)
        factorInner = random.uniform(0.5,2)
        summand = [random.uniform(-1,1) for i in range(9)]
#        rotationAngle= 1.57079632679
#        factorInner= 1.19586318888
#        factorOuter= 1.07949514065
#        summand= [0.3702342064716502, 0.3168554304529523, 0, 0.29536964718187986, -0.3021482994854201, 0, 0, 0, 0]
#
#        rotationAngle = 0
#        factorInner= 1
#        factorOuter= 1
#        summand = [0 for i in range(9)]
        if dimension == 2:
          summand[2] = summand[5] = summand[6] = summand[7] = summand[8] = 0

        pattern = scaleDataset(shiftDataset(rotateDatasetExact(cutoutPattern(dataset, dimension, patternPosition, radiusPattern), rotationAngle, nameOfPointData),summand, nameOfPointData),factorOuter, nameOfPointData)
        pattern.SetSpacing(pattern.GetSpacing()[0]*factorInner, pattern.GetSpacing()[1]*factorInner, pattern.GetSpacing()[2]*factorInner)
        print("patternPosition=", patternPosition, " rotationAngle=", rotationAngle, " factorInner=", factorInner, " factorOuter=", factorOuter, " summand=", summand)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(pattern)
        writer.SetFileName('/Users/bujack/Documents/moments/output/pattern2.vti')
        writer.SetDataModeToAscii()
        writer.Write()

        # compute moments
        momentsAlgo = vtk.vtkComputeMoments()
        momentsAlgo.SetFieldData(dataset)
        momentsAlgo.SetGridData(momentDataCoarse)
        momentsAlgo.SetNameOfPointData(nameOfPointData)
        momentsAlgo.SetNumberOfIntegrationSteps(numberOfIntegrationSteps)
        momentsAlgo.SetOrder(order)
        momentsAlgo.SetUseFFT(0)
        momentsAlgo.SetRadiiArray([radiusPattern,0,0,0,0,0,0,0,0,0])
        momentsAlgo.Update()
#        print(momentsAlgo)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(momentsAlgo.GetOutput())
        writer.SetFileName('/Users/bujack/Documents/moments/output/moments.vti')
        writer.SetDataModeToAscii()
        writer.Write()

        # pattern detetction
        invariantsAlgo = vtk.vtkMomentInvariants()
        invariantsAlgo.SetMomentData(momentsAlgo.GetOutput())
        invariantsAlgo.SetPatternData(pattern)
        invariantsAlgo.SetNameOfPointData(nameOfPointData)
        invariantsAlgo.SetIsScaling(1)
        invariantsAlgo.SetIsTranslation(1)
        invariantsAlgo.SetIsRotation(1)
        invariantsAlgo.SetIsReflection(1)
        invariantsAlgo.SetNumberOfIntegrationSteps(numberOfIntegrationSteps)
        invariantsAlgo.SetAngleResolution(angleResolution)
        invariantsAlgo.SetEps(eps)
        invariantsAlgo.Update()
#        print(invariantsAlgo)
        print("maxSimilarity =", invariantsAlgo.GetOutput().GetPointData().GetArray(0).GetRange()[1])

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(invariantsAlgo.GetOutput())
        writer.SetFileName('/Users/bujack/Documents/moments/output/invariants.vti')
        writer.SetDataModeToAscii()
        writer.Write()

        # detetction of the local maxima and provide visualization with solid balls and hollow spheres
        ballsAlgo = vtk.vtkSimilarityBalls()
        ballsAlgo.SetSimilarityData(invariantsAlgo.GetOutput())
        ballsAlgo.SetGridData(dataset)
        #    ballsAlgo.SetKindOfMaxima(2)
        ballsAlgo.Update()

        print("maxSimilarity =", ballsAlgo.GetOutput(0).GetPointData().GetArray("localMaxValue").GetRange()[1])
#        print(ballsAlgo.GetOutput().FindPoint(patternPosition))
        if ballsAlgo.GetOutput(0).GetPointData().GetArray("localMaxValue").GetTuple1( ballsAlgo.GetOutput().FindPoint(patternPosition)) == ballsAlgo.GetOutput(0).GetPointData().GetArray("localMaxValue").GetRange()[1]:
            print("pattern detected")
            return
        else:
            sys.exit(1)

files = ["2DScalar.vti", "2DMatrix.vti", "2DVector.vti", "3DScalar.vti"]
files = ["2DScalar.vti"]
for f in files:
    filePath = VTK_DATA_ROOT + "/Data/" + f
    doTest(filePath)


sys.exit(0)
