message(STATUS "Powercrust: Building as a Remote VTK module")
 
set(Module_SRCS
  vtkPowerCrustSurfaceReconstruction.cxx
  )

if (VTK_VERSION VERSION_LESS "8.90.0")
  vtk_module_library(Powercrust ${Module_SRCS})
else()
  vtk_module_add_module(VTK::Powercrust
    SOURCES ${Module_SRCS}
    )
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options (Powercrust PUBLIC -Wno-unused-parameter  -Wno-maybe-uninitialized -Wno-unused-variable -Wno-deprecated -Wno-unused-but-set-variable)
endif()
  