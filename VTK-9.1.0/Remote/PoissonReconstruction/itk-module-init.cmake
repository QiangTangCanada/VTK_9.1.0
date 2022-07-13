#
# Find the packages required by this module
#

# Look for VTK
message(STATUS "PoissonReconstruction: Finding VTK with required modules...")        
set(VTK_USE_FILE_INCLUDED 0)
  find_package(VTK COMPONENTS
    vtkCommonCore
    vtkCommonExecutionModel
    vtkIOXML
    REQUIRED
  )
include(${VTK_USE_FILE})
message(STATUS "PoissonReconstruction: VTK Found.")        
