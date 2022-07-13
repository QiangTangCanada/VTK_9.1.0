import vtk
import sys

if len(sys.argv) < 5:
  print("Invalid number of arguments")
  print("Usage: $ " + sys.argv[0] + " dataset_file field_name order radius1 ... radius10")
  exit(1)

if len(sys.argv) > 14:
  print("Only upto 10 radii are supported")
  exit(1)

datasetName = sys.argv[1]
fieldName = sys.argv[2]
order = int(sys.argv[3])
radii = [float(x) for x in sys.argv[4:]]
radii.extend([0] * (10 - len(radii)))

reader = vtk.vtkDataSetReader()
reader.SetFileName(datasetName)
reader.Update()

inputData = reader.GetOutput()

momentsFilter = vtk.vtkComputeMoments()
momentsFilter.SetFieldConnection(reader.GetOutputPort())
momentsFilter.SetGridConnection(reader.GetOutputPort())
momentsFilter.SetNameOfPointData(fieldName)
momentsFilter.SetUseFFT(0)
momentsFilter.SetUseOriginalResolution(1)
momentsFilter.SetOrder(order)
momentsFilter.SetRadiiArray(radii)
momentsFilter.Update()

output = momentsFilter.GetOutput()
expected = output.NewInstance()
expected.DeepCopy(output)

momentsFilter.SetUseGPU(1)
momentsFilter.Update()
output = momentsFilter.GetOutput()

if output.GetPointData().GetNumberOfArrays() != expected.GetPointData().GetNumberOfArrays():
  print("The number of arrays dont't match")
  exit(1)

pd = output.GetPointData()
arrays = [pd.GetArray(i) for i in xrange(0, pd.GetNumberOfArrays())]
print("Number of Arrays: " + str(len(arrays)))

success = True
for array in arrays:
  name = array.GetName()
  print("Name: " + name)
  expArray = expected.GetPointData().GetArray(array.GetName())
  print(expArray.GetName())

  radius = float(name[len("radius") : name.find("index")])
  dims = inputData.GetDimensions()
  spacing = inputData.GetSpacing()
  dr = [ int(radius / (x - 1e-10)) for x in spacing ]

  mismatch = 0
  e = 1e-3
  for y in xrange(dr[1], dims[1] - dr[1]):
    for x in xrange(dr[0], dims[0] - dr[0]):
      index = y * dims[0] + x
      if abs(array.GetValue(index) - expArray.GetValue(index)) > e:
        mismatch += 1

  if mismatch > 0:
    success = False
    print("Failed. " + str(mismatch) + " points don't match")


if success:
  print("Test completed successfully")
