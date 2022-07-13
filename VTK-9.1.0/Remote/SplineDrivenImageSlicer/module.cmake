set(DOCUMENTATION "A Spline Driven Reslice Algorithm as described in the VTK Journal Article at http://www.vtkjournal.org/browse/publication/838 .")

vtk_module( SplineDrivenImageSlicer
  DESCRIPTION
    "${DOCUMENTATION}"
  DEPENDS
    vtkCommonCore
    vtkFiltersCore
    vtkFiltersSources
    vtkImagingCore
  TEST_DEPENDS
    vtkTestingCore
    vtkImagingSources
  KIT
    vtkRemote
)
