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
import sys
try:
    import numpy as np
except ImportError:
    print("Numpy (http://numpy.scipy.org) not found.")
    print("This test requires numpy!")
    from vtk.test import Testing
    Testing.skip()

from patternDetectionHelper import *
import os, shutil
import math
import random

from vtk.util.misc import vtkGetDataRoot
VTK_DATA_ROOT = vtkGetDataRoot()

UseOriginalResolution = 1
order = 2
angleResolution = 8
eps = 1e-2
numberOfMoments = 5 #21 11 6 5
nameOfPointData = "values"

def doTest(filePath, use_fft, invariantMethod):
    if not os.path.isfile(filePath):
      print("File: " + filePath + " does not exist.")
      sys.exit(1)
    else:
        (dataset, bounds, dimension) = readDataset(filePath)
        npBounds = np.array(bounds)

        if (dataset.GetDimensions()[0]-1) % (numberOfMoments-1) != 0:
          print("Warning! The numberOfMoments will not produce a true coarse version of the input dataset.")

        # produce coarse grid where the moments will be computed
        if dimension == 3:
            momentDataCoarse = createCoarseDataset(bounds, numberOfMoments, numberOfMoments, numberOfMoments)
        else:
            momentDataCoarse = createCoarseDataset(bounds, numberOfMoments, numberOfMoments, 0)

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
        
        # rotate, multiply constant, change size, add constant
        rotationAngle = 2 * math.pi * 0.25 * random.randint(0,3)
        factorOuter = random.uniform(0.5,2)
        factorInner = random.uniform(0.5,2)
        summand = [random.uniform(-1,1) for i in range(9)]
    #    rotationAngle = 0
    #    factorInner= 1
    #    factorOuter= 1
    #    summand = [0 for i in range(9)]
        if dimension == 2:
          summand[2] = summand[5] = summand[6] = summand[7] = summand[8] = 0

        pattern = scaleDataset(shiftDataset(rotateDatasetExact(cutoutPattern(dataset, dimension, patternPosition, radiusPattern), rotationAngle, nameOfPointData),summand, nameOfPointData),factorOuter, nameOfPointData)
        pattern.SetSpacing(pattern.GetSpacing()[0]*factorInner, pattern.GetSpacing()[1]*factorInner, pattern.GetSpacing()[2]*factorInner)
        print("patternPosition=", patternPosition, " rotationAngle=", rotationAngle, " factorInner=", factorInner, " factorOuter=", factorOuter, " summand=", summand)

        # compute moments
        momentsAlgo = vtk.vtkComputeMoments()
        momentsAlgo.SetFieldData(dataset)
        momentsAlgo.SetGridData(momentDataCoarse)
        momentsAlgo.SetNameOfPointData(nameOfPointData)
        momentsAlgo.SetUseFFT(1 if use_fft else 0)
        momentsAlgo.SetUseOriginalResolution(UseOriginalResolution)
        momentsAlgo.SetOrder(order)
        momentsAlgo.SetRadiiArray([radiusPattern,0,0,0,0,0,0,0,0,0])
        momentsAlgo.Update()

        # writer = vtk.vtkXMLImageDataWriter()
        # writer.SetDataModeToAscii()

        # writer.SetInputData(dataset)
        # writer.SetFileName('./dataset.vti')
        # writer.Write()

        # writer.SetInputData(momentsAlgo.GetOutput())
        # writer.SetFileName('./serialMoments.vti')
        # writer.Write()

        # pattern detetction
        invariantsAlgo = vtk.vtkMomentInvariants()
        invariantsAlgo.SetInvariantMethod(invariantMethod)
        invariantsAlgo.SetMomentData(momentsAlgo.GetOutput())
        invariantsAlgo.SetPatternData(pattern)
        invariantsAlgo.SetNameOfPointData(nameOfPointData)
        invariantsAlgo.SetIsScaling(1)
        invariantsAlgo.SetIsTranslation(1)
        invariantsAlgo.SetIsRotation(1)
        invariantsAlgo.SetIsReflection(1)
        invariantsAlgo.SetUseOriginalResolution(UseOriginalResolution)
        invariantsAlgo.SetAngleResolution(angleResolution)
        invariantsAlgo.SetEps(eps)
        invariantsAlgo.Update()

        # writer.SetInputData(invariantsAlgo.GetOutput())
        # writer.SetFileName('./invariants.vti')
        # writer.Write()

    #    print invariantsAlgo

        # detetction of the local maxima and provide visualization with solid balls and hollow spheres
        ballsAlgo = vtk.vtkSimilarityBalls()
        ballsAlgo.SetSimilarityData(invariantsAlgo.GetOutput())
        ballsAlgo.SetGridData(dataset)
        #    ballsAlgo.SetKindOfMaxima(2)
        ballsAlgo.Update()

        # writer.SetInputData(ballsAlgo.GetOutput())
        # writer.SetFileName('./balls.vti')
        # writer.Write()

        localMaxValueArray = ballsAlgo.GetOutput(0).GetPointData().GetArray("localMaxValue")
        print("maxSimilarity =", localMaxValueArray.GetRange()[1])
    #    print ballsAlgo.GetOutput().FindPoint(patternPosition)
        if localMaxValueArray.GetTuple1( ballsAlgo.GetOutput().FindPoint(patternPosition)) == localMaxValueArray.GetRange()[1] and localMaxValueArray.GetRange()[1] > 1e3:
            return
        else:
            sys.exit(-1)

def compute(files):
    for f in files:
        for use_fft in [True, False]:
            for invariantMethod in ["Normalization", "LangbeinInvariants", "Generator"]:
                doTest(f, use_fft, invariantMethod)
    sys.exit(0)

if __name__ == '__main__':
    files = sys.argv[1:]
    exit(compute(files))
