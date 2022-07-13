set(MI_SOURCES
  vtkPMomentsHelper.cxx
  vtkPComputeMoments.cxx
  vtkPSimilarityBalls.cxx
)

set(MI_HEADERS
  vtkPMomentsHelper.h
  vtkPComputeMoments.h
  vtkPSimilarityBalls.h
)

vtk_module_add_module(VTK::ParallelMomentInvariants
  SOURCES ${MI_SOURCES}
  HEADERS ${MI_HEADERS})

if(VTK_USE_CUDA)
  set_source_files_properties(vtkPComputeMoments.cxx PROPERTIES LANGUAGE "CUDA")
  # warning #381-D: extra ";" ignored
  set(momentinvariants_disable_warnings
    "-Xcudafe --diag_suppress=381")
  set_source_files_properties(vtkPComputeMoments.cxx PROPERTIES
    COMPILE_FLAGS ${momentinvariants_disable_warnings})
  vtk_module_set_property(VTK::ParallelMomentInvariants
      PROPERTY  CUDA_RESOLVE_DEVICE_SYMBOLS
      VALUE     OFF)
endif()
