get_filename_component(DATA_ROOT "DATA{${DATA_PATH}/2DScalar.vti}" DIRECTORY)

set(test_files
  "2DScalar.vti"
  "2DMatrix.vti"
  "2DVector.vti"
  "3DScalar.vti"
  )

set(test_files_full "")
foreach(file ${test_files})
  list(APPEND test_files_full "DATA{${DATA_PATH}/${file}}")
endforeach()

vtk_add_test_python(
  NO_VALID NO_DATA NO_OUTPUT
  patternDetectionTestSimple,patternDetectionTestSimple.py
  "${test_files_full}")

vtk_add_test_python(
  NO_VALID NO_DATA NO_OUTPUT
  serialCheckComputeShort2D,TestConvFFTComputeShort.py
  "DATA{${DATA_PATH}/2DScalar.vti}"
  "DATA{${DATA_PATH}/2DScalar.vtm}")

vtk_add_test_python(
  NO_VALID NO_DATA NO_OUTPUT
  serialCheckComputeShort3D,TestConvFFTComputeShort.py
  "DATA{${DATA_PATH}/3DScalar.vti}"
  "DATA{${DATA_PATH}/3DScalar.vtm}")
