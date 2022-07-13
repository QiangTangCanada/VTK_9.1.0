set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Testing/Data/Input")
vtk_module_test_data("DATA{${DATA_PATH}/,RECURSE:,REGEX:.*}")

if (VTK_WRAP_PYTHON)
  add_subdirectory(Python)
endif()
