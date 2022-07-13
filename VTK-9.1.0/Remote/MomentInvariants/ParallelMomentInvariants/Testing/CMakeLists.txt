set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Testing/Data/Input")
vtk_module_test_data("DATA{${DATA_PATH}/,RECURSE:,REGEX:.*}")

# Disable CUDA for parallel tests since they are failing
if (VTK_WRAP_PYTHON AND NOT VTK_USE_CUDA)
  add_subdirectory(Python)
endif()
