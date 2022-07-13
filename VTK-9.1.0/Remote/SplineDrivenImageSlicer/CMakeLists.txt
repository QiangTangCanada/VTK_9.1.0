message(STATUS "SplineDrivenImageSlicer: Building as a Remote VTK module")

set(Module_SRCS
  vtkFrenetSerretFrame.cxx
  vtkSplineDrivenImageSlicer.cxx  
)

if (VTK_VERSION VERSION_LESS "8.90.0")
  vtk_module_library(SplineDrivenImageSlicer ${Module_SRCS})
else()
  vtk_module_add_module(VTK::SplineDrivenImageSlicer
    SOURCES ${Module_SRCS}
    )
endif()
