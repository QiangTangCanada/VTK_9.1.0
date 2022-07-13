set(MI_SOURCES
  vtkComputeMoments.cxx
  vtkmComputeMoments.cxx
  vtkMomentInvariants.cxx
  vtkMomentsHelper.cxx
  vtkReconstructFromMoments.cxx
  vtkSimilarityBalls.cxx
  vtkMomentsTensorSimple.cxx
  vtkMomentsIndex.cxx
  vtkMomentsFactor.cxx
  vtkMomentsProduct.cxx
  vtkMomentsPolynomial.cxx
)

set(MI_HEADERS
  vtkComputeMoments.h
  vtkMomentInvariants.h
  vtkMomentsHelper.h
  vtkReconstructFromMoments.h
  vtkSimilarityBalls.h
  vtkMomentsGeneratorHelper.h
  vtkMomentInvariantData.h
  vtkMomentsTensorSimple.h
  vtkMomentsIndex.h
  vtkMomentsFactor.h
  vtkMomentsProduct.h
  vtkMomentsPolynomial.h
)

if (TARGET vtkm)
  set_source_files_properties(
      vtkmComputeMoments.cxx
      PROPERTIES COMPILE_DEFINITIONS "VTKM_AVAILABLE=1")

  if (TARGET vtkm::cuda)
      set_source_files_properties(
        vtkmComputeMoments.cxx
        PROPERTIES LANGUAGE "CUDA")
  endif ()
endif()		   

vtk_module_add_module(VTK::MomentInvariants
  SOURCES ${MI_SOURCES}
  HEADERS ${MI_HEADERS})
