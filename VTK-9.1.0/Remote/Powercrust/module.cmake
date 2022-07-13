set(DOCUMENTATION "This module contains a class to reconstruct surfaces from unorganized point data.
The original algorithm is the Power Crust, for full details and for the original code, see: http://www.cs.utexas.edu/users/amenta/powercrust/welcome.html)")

vtk_module( Powercrust
  DESCRIPTION
    "${DOCUMENTATION}"
  DEPENDS
    vtkCommonCore
    vtkCommonExecutionModel
    vtkFiltersPoints
  KIT
    vtkRemote
)
