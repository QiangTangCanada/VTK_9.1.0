set(DOCUMENTATION "This module contains a class to perform Poisson Surface reconstruction.
(This code was, with permission, adapted directly from the original implementation by Kazhdan, Bolitho, and Hugues.)")

vtk_module(PoissonReconstruction
  DESCRIPTION
    "${DOCUMENTATION}"
  DEPENDS
    vtkCommonCore
    vtkCommonExecutionModel
    vtkIOXML
  KIT
    vtkRemote
)
