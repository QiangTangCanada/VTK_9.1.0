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
    import numpy
except ImportError:
    print("Numpy (http://numpy.scipy.org) not found.")
    print("This test requires numpy!")
    from vtk.test import Testing
    Testing.skip()

from parallelPatternDetectionHelper import *

# xrange doesn't exist in Python3
if sys.version_info >= (3,0):
  xrange = range

UseOriginalResolution = 1
numberOfIntegrationSteps = (1 - UseOriginalResolution) * 5
order = 2
nameOfPointData = "values"
radius = 0.1

def compute(whole, split, useFFT):
  # prepare
  controller = vtk.vtkMultiProcessController.GetGlobalController()
  assert (controller != None)
  rank = controller.GetLocalProcessId()
  nprocs = controller.GetNumberOfProcesses()
  del controller

  # print('rank={0}'.format(rank))
  # print('nprocs={0}'.format(nprocs))

  # whole
  (wholeDataset, bounds, dimension) = readDataset(whole)
  if pow(2,dimension) != nprocs:
    print('the number of procs must be 2^dimension')
    sys.exit(-1)

  # coarse
  coarseData = vtk.vtkImageData()
  coarseData.CopyStructure(wholeDataset)
  coarseData.SetSpacing(0.1,0.1,0.1)
  if dimension == 2:
    coarseData.SetDimensions(11,11,1)
  else:
    coarseData.SetDimensions(11,11,11)
  # print data
  # print coarseData

  # compute moments
  momentsFilter = vtk.vtkComputeMoments()
  momentsFilter.SetFieldData(wholeDataset)
  momentsFilter.SetGridData(wholeDataset)
  momentsFilter.SetNameOfPointData(nameOfPointData)
  momentsFilter.SetUseFFT(1 if useFFT else 0)
  momentsFilter.SetUseOriginalResolution(UseOriginalResolution)
  momentsFilter.SetNumberOfIntegrationSteps(numberOfIntegrationSteps)
  momentsFilter.SetOrder(order)
  momentsFilter.SetRadiiArray([radius,0,0,0,0,0,0,0,0,0])
  momentsFilter.Update()
  # if rank ==0: print momentsFilter

  # this is to make sure each proc only loads their piece
  ids_to_read = [rank]
  reqs = vtk.vtkInformation()
  reqs.Set(vtk.vtkCompositeDataPipeline.UPDATE_COMPOSITE_INDICES(), ids_to_read, 1)

  reader = vtk.vtkXMLMultiBlockDataReader()
  reader.SetFileName(split)
  reader.Update(reqs)
  # print('reader={0}'.format(reader.GetOutput()))

  multiblock = vtk.vtkMultiBlockDataSet.SafeDownCast(reader.GetOutput())
  # print('multiblock={0}'.format(multiblock))

  multipiece = vtk.vtkMultiPieceDataSet.SafeDownCast(multiblock.GetBlock(0))
  # print('multipiece={0}'.format(multipiece))

  data = vtk.vtkImageData.SafeDownCast(multipiece.GetPiece(rank))
  # print('data={0}'.format(data))

  # coarse
  #coarseData = vtk.vtkImageData()
  #coarseData.CopyStructure(data)
  #coarseData.SetSpacing(0.1,0.1,0.1)
  # incorrect setdimensions for parallel dataset
  #if dimension == 2:
  #  coarseData.SetDimensions(6,6,1)
  #else:
  #  coarseData.SetDimensions(6,6,6)
  # print data
  # print coarseData

  # parallel
  pMomentsFilter = vtk.vtkPComputeMoments()
  pMomentsFilter.SetFieldData(data)
  pMomentsFilter.SetGridData(data)
  pMomentsFilter.SetNameOfPointData(nameOfPointData)
  pMomentsFilter.SetUseFFT(1 if useFFT else 0)
  pMomentsFilter.SetUseOriginalResolution(UseOriginalResolution)
  pMomentsFilter.SetNumberOfIntegrationSteps(numberOfIntegrationSteps)
  pMomentsFilter.SetRadiiArray([ radius, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
  pMomentsFilter.Update(reqs)
  # if rank ==0: print pMomentsFilter

  # check for difference except for the global boundary (the probe and parallel probe filters behave differently there. So, there is a difference if numberOfIntegrationSteps > 0 )
  diff = 0
  for i in xrange(pMomentsFilter.GetOutput().GetPointData().GetNumberOfArrays()):
    diffArray = vtk.vtkDoubleArray()
    diffArray.SetName( pMomentsFilter.GetOutput().GetPointData().GetArrayName(i) )
    diffArray.SetNumberOfComponents( 1 )
    diffArray.SetNumberOfTuples( pMomentsFilter.GetOutput().GetNumberOfPoints() )
    diffArray.Fill( 0.0 )

    momentsArray = momentsFilter.GetOutput().GetPointData().GetArray(i)
    pMomentsArray = pMomentsFilter.GetOutput().GetPointData().GetArray(i)

    if momentsArray.GetName() != pMomentsArray.GetName():
      print("names don't match")

    for j in xrange(pMomentsFilter.GetOutput().GetNumberOfPoints()):
      diffArray.SetTuple1(j, abs(momentsArray.GetTuple1(momentsFilter.GetOutput().FindPoint(pMomentsFilter.GetOutput().GetPoint(j))) - pMomentsArray.GetTuple1(j)))
      if (dimension == 2 and pMomentsFilter.GetOutput().GetPoint(j)[0] > 0 and \
          pMomentsFilter.GetOutput().GetPoint(j)[0] < 1 and \
          pMomentsFilter.GetOutput().GetPoint(j)[1] > 0 and \
          pMomentsFilter.GetOutput().GetPoint(j)[1] < 1) or \
         (dimension == 3 and pMomentsFilter.GetOutput().GetPoint(j)[0] > 0 and \
          pMomentsFilter.GetOutput().GetPoint(j)[0] < 1 and \
          pMomentsFilter.GetOutput().GetPoint(j)[1] > 0 and \
          pMomentsFilter.GetOutput().GetPoint(j)[1] < 1 and \
          pMomentsFilter.GetOutput().GetPoint(j)[2] > 0 and \
          pMomentsFilter.GetOutput().GetPoint(j)[2] < 1):
        # if diffArray.GetTuple1(j) > 1e-5:
        #   print rank, i, j, pMomentsFilter.GetOutput().GetPoint(j), diffArray.GetTuple(j)
        diff = max(diff, diffArray.GetTuple1(j))

  # writer = vtk.vtkXMLPImageDataWriter()
  # npieces = nprocs
  # writer.SetNumberOfPieces(npieces)
  # pperrank = npieces // nprocs
  # start = pperrank * rank
  # end = start + pperrank - 1
  # writer.SetStartPiece(start)
  # writer.SetEndPiece(end)
  # writer.SetUseSubdirectory(True)
  # writer.SetDataModeToAscii()

  # momentsDiff = vtk.vtkImageData()
  # momentsDiff.GetPointData().AddArray(diffArray)
  # writer.SetInputData(momentsDiff)
  # writer.SetFileName('./momentsDiff.pvti')
  # writer.Write()

  # writer.SetInputData(momentsFilter.GetOutput())
  # writer.SetFileName('/projects/users/ktsai/alpine_moment_invariants/vtk-dev/build/serialMoments.pvti')
  # writer.Write()

  # writer.SetInputData(pMomentsFilter.GetOutput())
  # writer.SetFileName('/projects/users/ktsai/alpine_moment_invariants/vtk-dev/build/parallelMoments.pvti')
  # writer.Write()

  if diff > 1e-5:
    print("test failed, maxdiff =", diff)
    return -1
  else:
    print("test successful ", diff)
    return 0

  # if rank == 0:
  #   print('all done!')
  # return 0

if __name__ == '__main__':
  if len(sys.argv) <= 2:
    print('usage: <file whole> <file split>')
    exit(0)
  else:
    use_fft = False
    if len(sys.argv) > 3:
      if sys.argv[3] == "fft" or sys.argv[3] == "FFT":
        use_fft = True
    exit(compute(sys.argv[1], sys.argv[2], use_fft))
