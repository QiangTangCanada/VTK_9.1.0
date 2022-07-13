project(DFFTLIB)

set(BUILD_SHARED_LIBS OFF)

set(DFFT_SRCS
  bare_fft_interface.c
  bare_fft.c
  dfft_common.c
  dfft_host.c)

set(DFFT_HDRS
  bare_fft_interface.h
  bare_fft.h
  dfft_common.h
  dfft_host.h
  dfft_lib_config.h
  dfft_local_fft_config.h)

# optional cuda + mpi support, when MPI lib supports it.
# include(CUDA_MPI.cmake)

if(VTK_USE_CUDA)
    # CUFFT is default for CUDA
    set(CUDA_SOURCES
      cufft_single_interface.c
      dfft_cuda.c
      dfft_cuda.cu)

    set_source_files_properties(${DFFT_SRCS} PROPERTIES LANGUAGE "CUDA")
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE "CUDA")

    set(CUDA_HEADERS
      cufft_single_interface.h
      dfft_cuda.h
      dfft_cuda.cuh)
endif()

vtk_module_add_module(VTK::dfft
  SOURCES ${DFFT_SRCS} ${CUDA_SOURCES}
  HEADERS ${DFFT_HDRS} ${CUDA_HEADERS})

vtk_module_compile_features(VTK::dfft
  PRIVATE c_std_99)

option(VTK_DFFT_ENABLE_HOST "Enable CPU FFT" ON)
if (VTK_DFFT_ENABLE_HOST)
  vtk_module_definitions(VTK::dfft PUBLIC "ENABLE_HOST")
endif()

if(VTK_USE_CUDA)
  vtk_module_definitions(VTK::dfft PUBLIC "ENABLE_CUDA")
  vtk_module_link(VTK::dfft PRIVATE cufft)
endif()

set(local_fft_lib_options "BARE" "MKL" "ACML")
set(VTK_DFFT_LOCAL_FFT_LIB "BARE" CACHE STRING "Local FFT library to use")
set_property(CACHE VTK_DFFT_LOCAL_FFT_LIB PROPERTY STRINGS ${local_fft_lib_options})
vtk_module_definitions(VTK::dfft PUBLIC "LOCAL_FFT_LIB=LOCAL_LIB_${VTK_DFFT_LOCAL_FFT_LIB}")
