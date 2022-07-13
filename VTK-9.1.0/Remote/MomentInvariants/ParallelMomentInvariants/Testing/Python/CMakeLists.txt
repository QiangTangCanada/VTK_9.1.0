set(ParallelMomentInvariants_NUMPROCS 4)

vtk_add_test_python_mpi(
  NO_VALID NO_DATA NO_OUTPUT
  parallelMomentsComputeShort-p4,TestParallelComputeShort.py
  "DATA{${DATA_PATH}/2DScalar.vti}"
  "DATA{${DATA_PATH}/2DScalar.vtm}")

vtk_add_test_python_mpi(
  NO_VALID NO_DATA NO_OUTPUT
  parallelMomentsFFTComputeShort-p4,TestParallelComputeShort.py
  "DATA{${DATA_PATH}/2DScalar.vti}"
  "DATA{${DATA_PATH}/2DScalar.vtm}"
  "fft")

vtk_add_test_python_mpi(
  NO_VALID NO_DATA NO_OUTPUT
  parallelCheckComputeShort-p4,TestConvFFTComputeShort.py
  "DATA{${DATA_PATH}/2DScalar.vti}"
  "DATA{${DATA_PATH}/2DScalar.vtm}"
  "parallel")

set(ParallelMomentInvariants_NUMPROCS 8)

vtk_add_test_python_mpi(
  NO_VALID NO_DATA NO_OUTPUT
  parallelMomentsComputeShort-p8,TestParallelComputeShort.py
  "DATA{${DATA_PATH}/3DScalar.vti}"
  "DATA{${DATA_PATH}/3DScalar.vtm}")

vtk_add_test_python_mpi(
  NO_VALID NO_DATA NO_OUTPUT
  parallelMomentsFFTComputeShort-p8,TestParallelComputeShort.py
  "DATA{${DATA_PATH}/3DScalar.vti}"
  "DATA{${DATA_PATH}/3DScalar.vtm}"
  "fft")

vtk_add_test_python_mpi(
  NO_VALID NO_DATA NO_OUTPUT
  parallelCheckComputeShort-p8,TestConvFFTComputeShort.py
  "DATA{${DATA_PATH}/3DScalar.vti}"
  "DATA{${DATA_PATH}/3DScalar.vtm}"
  "parallel")
