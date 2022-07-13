/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkPComputeMoments.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*=========================================================================

Copyright (c) 2017, Los Alamos National Security, LLC

All rights reserved.

Copyright 2017. Los Alamos National Security, LLC.
This software was produced under U.S. Government contract DE-AC52-06NA25396
for Los Alamos National Laboratory (LANL), which is operated by
Los Alamos National Security, LLC for the U.S. Department of Energy.
The U.S. Government has rights to use, reproduce, and distribute this software.
NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY,
EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
If software is modified to produce derivative works, such modified software
should be clearly marked, so as not to confuse it with the version available
from LANL.

Additionally, redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following conditions
are met:
-   Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
-   Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
-   Neither the name of Los Alamos National Security, LLC, Los Alamos National
    Laboratory, LANL, the U.S. Government, nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/
#include "vtkPComputeMoments.h"

#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiProcessController.h"
#include "vtkPointData.h"
#include "vtkPResampleWithDataSet.h"
#include "vtkProbeFilter.h"
#include "vtkStreamingDemandDrivenPipeline.h"

// #include "vtkMPIController.h"
// #include "vtkXMLPImageDataWriter.h"

#include "vtkMPICommunicator.h"
#include "vtkImageTranslateExtent.h"
#include "vtkTrivialProducer.h"

#include "vtkPMomentsHelper.h"
#include "vtkMomentsHelper.h"
#include "vtkMomentsTensor.h"

#include <vector>

#include "dfft_host.h"

#ifdef __CUDACC__
#include "dfft_cuda.h"
#endif

namespace dfft {

#ifdef __CUDACC__
using complex_t = cuda_cpx_t;
#else
using complex_t = cpx_t;
#endif

inline complex_t *allocate_cpx(std::size_t count) {
  complex_t *ptr = nullptr;
  const std::size_t byte_size = sizeof(complex_t) * count;
#ifdef __CUDACC__
  cudaMallocManaged((void **)&ptr, byte_size);
#else
  ptr = static_cast<complex_t *>(malloc(byte_size));
#endif
  return ptr;
}

inline void free_cpx(complex_t* ptr)
{
#ifdef __CUDACC__
  cudaFree(ptr);
#else
  free(ptr);
#endif
}

template <typename... T> inline void create_plan(T &&... args) {
#ifdef __CUDACC__
  dfft_cuda_create_plan(std::forward<T>(args)...);
#else
  dfft_create_plan(std::forward<T>(args)...);
#endif
}

template <typename... T> inline void destroy_plan(T &&... args) {
#ifdef __CUDACC__
  dfft_cuda_destroy_plan(std::forward<T>(args)...);
#else
  dfft_destroy_plan(std::forward<T>(args)...);
#endif
}

inline void execute(complex_t *id_in, complex_t *d_out, int dir, dfft_plan p) {
#ifdef __CUDACC__
  dfft_cuda_execute(id_in, d_out, dir, &p);
#else
  dfft_execute(id_in, d_out, dir, p);
#endif
}

} // namespace dfft

#define RE(X) X.x
#define IM(X) X.y

/**
 * standard vtk new operator
 */
vtkStandardNewMacro(vtkPComputeMoments);
vtkSetObjectImplementationMacro(vtkPComputeMoments, Controller, vtkMultiProcessController);

/**
 * constructor setting defaults
 */
vtkPComputeMoments::vtkPComputeMoments()
{
  this->Controller = NULL;
  this->SetController(vtkMultiProcessController::GetGlobalController());
  // If we are top-level and need to create controller ourselves...
  // this->Controller = vtkMPIController()::New();
  // this->Controller->Initialize();
  // vtkMultiProcessController::SetGlobalController(this->Controller);

  // this->MPI_Controller = NULL;
  // this->MPI_Controller = vtkMPIController::SafeDownCast(this->Controller);

  this->MPI_Communicator = NULL;
  vtkMPICommunicator *comm = vtkMPICommunicator::SafeDownCast(this->Controller->GetCommunicator());
  this->MPI_Communicator = comm->GetMPIComm()->GetHandle();

  this->FieldGlobalExtent = std::vector<int>(6, 0);
}

/**
 * destructor
 */
vtkPComputeMoments::~vtkPComputeMoments()
{
  this->SetController(nullptr);
}

/**
 * Helper method for checking the validity of given number of ranks.
 */
void vtkPComputeMoments::CheckRanksValidity()
{
  /* Getting info from controller */
  int numProcs = this->Controller->GetNumberOfProcesses();
  while(numProcs % 2 == 0)
  {
    numProcs /= 2;
  }
  if(numProcs != 1)
  {
    vtkErrorMacro("The number of ranks must be a power of 2.");
  }
}

/**
 * Helper method for checking the validity of given grid.
 * @param field: function of which the moments are computed
 * @param grid: the uniform grid on which the moments are computed
 */
void vtkPComputeMoments::CheckGridValidity(vtkImageData* field, vtkImageData* grid, vtkInformation* gridInfo)
{
  if (!gridInfo)
  {
    return;
  }
  double gridSpacing[3];
  grid->GetSpacing(gridSpacing);
  for (int d = 0; d < 3; ++d)
  {
    if (gridSpacing[d] <= 0)
    {
      vtkErrorMacro("The grid must have positive spacing in all 3 components even if it is 2D.");
    }
  }

  int numProcs = this->Controller->GetNumberOfProcesses();
  double allFieldBounds[6 * numProcs + 6];
  double allGridBounds[6 * numProcs + 6];
  vtkPMomentsHelper::GetAllBounds(allFieldBounds, field, this->Controller);
  vtkPMomentsHelper::GetAllBounds(allGridBounds, grid, this->Controller);

  double gridGlobalBounds[6];
  for (int i = 0; i < 6; i++)
  {
    this->FieldGlobalBounds[i] = allFieldBounds[6 * numProcs + i];
    gridGlobalBounds[i] = allGridBounds[6 * numProcs + i];
  }

  for (int d = 0; d < this->Dimension; ++d)
  {
    if (this->FieldGlobalBounds[2 * d] > gridGlobalBounds[2 * d] + 1e-10 ||
      this->FieldGlobalBounds[2 * d + 1] < gridGlobalBounds[2 * d + 1] - 1e-10)
    {
      std::cout << "Global field bounds: " << FieldGlobalBounds[2*d] << ", " << FieldGlobalBounds[2*d+1] << std::endl;
      std::cout << "Global grid bounds: " << gridGlobalBounds[2*d] << ", " << gridGlobalBounds[2*d+1] << std::endl;
      vtkErrorMacro("The grid must be inside the field bounds.");
    }
  }

  if (this->CoarseningFactor > 1)
  {
    vtkErrorMacro("The CoarseningFactor will be ignored and should be 1 if an outside grid is provided.  It is " << this->CoarseningFactor);
    return;
  }

  if (!this->UseFFT && this->UseOriginalResolution)
  {
    bool pointsDontMatch = false;
    for (int ptId = 0; ptId < grid->GetNumberOfPoints(); ++ptId)
    {
      double center[3];
      grid->GetPoint(ptId, center);
      int id = field->FindPoint(center);
      for (int d = 0; d < this->Dimension; ++d)
      {
        if (std::abs(field->GetPoint(id)[d] - center[d]) > 1e-3)
        {
          pointsDontMatch = true;
          break;
        }
      }
      if (pointsDontMatch)
      {
        std::cout << "Field " << id << ": " << field->GetPoint(id)[0] << ", " << field->GetPoint(id)[1] << ", " << field->GetPoint(id)[2] << std::endl;
        std::cout << "Center " << ptId << ": " << center[0] << ", " << center[1] << ", " << center[2] << std::endl;
        std::cout << std::endl;
        vtkWarningMacro("Some grid points do not lie on the field points, which will result in a shifted result.");
        break;
      }
    }
  }
}

/**
 * the agorithm has two input ports
 * port 0 is the dataset of which the moments are computed
 * port 1 is the grid at whose locations the moments are computed. if not set, the original grid is
 * chosen
 */
int vtkPComputeMoments::FillInputPortInformation(int port, vtkInformation* info)
{
  if (port == 0)
  {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 0);
  }
  if (port == 1)
  {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  }
  return 1;
}

/**
 * the agorithm generates a field of vtkImageData storing the moments. It will
 * have numberOfFields scalar arrays in its pointdata it has the same dimensions
 * and topology as the second inputport
 */
int vtkPComputeMoments::FillOutputPortInformation(int, vtkInformation* info)
{
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
  return 1;
}

/**
 * This function moves the stencil to the current location, where the integration is supposed o be
 * performed
 * @param center: the location
 * @param field: the dataset
 * @param stencil: contains the locations at which the dataset is evaluated for the integration
 * @param numberOfIntegrationSteps: how fine the discrete integration done in each dimension
 * @return 0 if the stencil lies completely outside the field
 */
bool vtkPComputeMoments::CenterStencil(double center[3], vtkDataSet* field, vtkImageData* stencil,
  int numberOfIntegrationSteps, std::string nameOfPointData)
{
  // put the center to the point where the moments shall be calculated
  if (numberOfIntegrationSteps == 1)
  {
    stencil->SetOrigin(center);
  }
  else
  {
    double bounds[6];
    stencil->GetBounds(bounds);
    stencil->SetOrigin(center[0] - 0.5 * (bounds[1] - bounds[0]),
      center[1] - 0.5 * (bounds[3] - bounds[2]), center[2] - 0.5 * (bounds[5] - bounds[4]));
  }

  //  int subId = 0;
  //
  //  // interpolation of the field data at the integration points
  //  for (vtkIdType ptId = 0; ptId < stencil->GetNumberOfPoints(); ++ptId)
  //  {
  //    // find point coordinates
  //    double x[3];
  //    stencil->GetPoint(ptId, x);
  //
  //    // find cell
  //    double pcoords[3];
  //    double* weights = new double[field->GetMaxCellSize()];
  //    vtkIdType cellId = field->FindCell(x, NULL, -1, 1, subId, pcoords, weights);
  //    vtkCell* cell;
  //    if (cellId >= 0)
  //    {
  //      cell = field->GetCell(cellId);
  //    }
  //    else
  //    {
  //      cell = 0;
  //    }
  //    if (cell)
  //    {
  //      // Interpolate the point data
  //      stencil->GetPointData()->InterpolatePoint(
  //                                                field->GetPointData(), ptId, cell->PointIds,
  //                                                weights);
  //    }
  //    else
  //    {
  //      return (false);
  //    }
  //  }

  vtkNew<vtkPResampleWithDataSet> resample;
  resample->SetController(this->Controller);
  resample->SetInputData(stencil);
  resample->SetSourceData(field);
  resample->Update();

  //  vtkNew<vtkProbeFilter> resample;
  //  resample->SetInputData(stencil);
  //  resample->SetSourceData(field);
  //  resample->Update();

//  if (vtkImageData::SafeDownCast(resample->GetOutput())
//        ->GetPointData()
//        ->GetArray("vtkValidPointMask")
//        ->GetRange()[1] == 0)
//  {
//    return (false);
//  }

  stencil->GetPointData()->RemoveArray(nameOfPointData.c_str());
  stencil->GetPointData()->AddArray(vtkImageData::SafeDownCast(resample->GetOutput())
                                      ->GetPointData()
                                      ->GetArray(nameOfPointData.c_str()));

  //  if( this->Controller->GetLocalProcessId() == 0 && center[0] == 0 && center[1] == 0 )
  //  {
  //    std::ostream stream(std::cout.rdbuf());
  //    std::cout<<"stencil=";
  //    stencil->PrintSelf(stream, vtkIndent(0));
  //    std::cout<<"\n";
  //    std::cout<<"point="<<center[0]<<" "<<center[1]<<" range="<<stencil->GetScalarRange()[0]<<"
  //    "<<stencil->GetScalarRange()[1]<<" bounds="<<stencil->GetBounds()[0]<<"
  //    "<<stencil->GetBounds()[1]<<"\n";
  //    for (vtkIdType ptId = 0; ptId < stencil->GetNumberOfPoints(); ++ptId)
  //    {
  //      std::cout<<stencil->GetPointData()->GetArray(0)->GetTuple(ptId)[0]<<" ";
  //    }
  //    std::cout<<"\n";
  //  }

  return (true);
}

/**
 * This function handles the moment computation on the original resolution   * this is where all the
 * communication with the other procs happens
 * 1. It requires that the (partial) moments for all points on this grid have already been computed via the serial function
 * 2. it looks where points close to the boundary fall in the bounds of other procs and sends the
 * locations over as partly negative dimension-wise indizes of imageData
 * 3. each proc computes the parts of the momenrts in its domain and sends the results back
 * 4. in each home proc, the native and incoming moment parts are added up
 * the moments are the projections of the function to the monomial basis
 * they are evaluated using a numerical integration over the original dataset if it is structured
 * data
 * @param dimension: 2D or 3D
 * @param order: the maximal order up to which the moments are computed
 * @param fieldRank: 0 for scalar, 1 for vector and 2 for matrix
 * @param radiusIndex: index of this radius in the radii vector
 * @param field: the dataset of which the moments are computed
 * @param grid: the uniform grid on which the moments are computed
 * @param output: this vtkImageData has the same topology as grid and will
 * @param nameOfPointData: the name of the array in the point data of which the momens are computed.
 */
void vtkPComputeMoments::ComputeBoundaryOrigRes(int radiusIndex,
  vtkImageData* field, vtkImageData* grid, vtkImageData* output)
{
  int numProcs = this->Controller->GetNumberOfProcesses();
  double radius = this->Radii.at(radiusIndex);
  double gridBounds[6];
  grid->GetBounds(gridBounds);
  double fieldBounds[6];
  field->GetBounds(fieldBounds);
  double allFieldBounds[6 * numProcs + 6];
  vtkPMomentsHelper::GetAllBounds(allFieldBounds, field, this->Controller);

  // this vector contains a vector for each proc. with the centers + radii + similarity that are close to this proc's boundary
  std::vector<std::vector<std::vector<double> > > myBoundaryCenters = vtkPMomentsHelper::GetBoundaryCenters(allFieldBounds, grid, this->Dimension, radius, this->Controller);
  // exchange boundary centers between nodes. distance is commutative. so we can send and receive
  std::vector<std::vector< std::vector<double> > > foreignBoundaryCenters = vtkPMomentsHelper::ExchangeBoundaryInformation(3, myBoundaryCenters, this->Dimension, this->Controller);

  // compute the moments of the foreign centers
  std::vector<std::vector<std::vector<vtkMomentsTensor>> > foreignBoundaryMoments(numProcs);
  for (int p = 0; p < numProcs; ++p)
  {
    foreignBoundaryMoments.at(p).resize(foreignBoundaryCenters.at(p).size());
    for (int fc = 0; fc < foreignBoundaryCenters.at(p).size(); ++fc)
    {
//      // this would make the boundary zero
//      double center[3];
//      for (int d = 0; d < this-Dimension; ++d)
//      {
//        center[d] = foreignBoundaryCenters.at(p).at(fc).at(d);
//      }
//      if(vtkMomentsHelper::IsCloseToBoundary(center, radius, this->FieldGlobalBounds, this->Dimension))
//      {
//        continue;
//      }
      foreignBoundaryMoments.at(p).at(fc) = std::vector<vtkMomentsTensor>(this->Order + 1);
      for (int o = 0; o < this->Order + 1; o++)
      {
        foreignBoundaryMoments.at(p).at(fc).at(o) =
        vtkMomentsTensor(this->Dimension, o + this->FieldRank, this->FieldRank);
      }
      // determine the possibly negative indices i and j per this->Dimension of the foreign center in
      // this field.
      int dimPtId[this->Dimension];
      for (int d = 0; d < this->Dimension; ++d)
      {
        dimPtId[d] = round((foreignBoundaryCenters.at(p).at(fc).at(d) - gridBounds[2 * d]) /
			    grid->GetSpacing()[d]);
      }

      // compute part of the moments
      foreignBoundaryMoments.at(p).at(fc) =
      vtkMomentsHelper::allMomentsOrigResImageData(this->Dimension, this->Order,
                                                   this->FieldRank, radius, dimPtId, field, grid, this->NameOfPointData);
    }
  }


  // send the partly moments back home
  std::vector<std::vector<std::vector<double> > > foreignBoundaryMomentsFlat(numProcs);
  for (int p = 0; p < numProcs; ++p)
  {
    foreignBoundaryMomentsFlat.at(p).resize(foreignBoundaryCenters.at(p).size());
    for (int fc = 0; fc < foreignBoundaryCenters.at(p).size(); ++fc)
    {
      foreignBoundaryMomentsFlat.at(p).at(fc).resize(this->NumberOfDifferentBasisFunctions);
      int index = 0;
      for (int o = 0; o < this->Order + 1; o++)
      {
        vtkMomentsTensor fbm = foreignBoundaryMoments.at(p).at(fc).at(o);
        int size = 0;
        if (this->Dimension == 2)
        {
          size = pow(this->Dimension, this->FieldRank) * (o + 1);
        }
        else
        {
          size = pow(this->Dimension, this->FieldRank) * 0.5 * (o + 1) * (o + 2);
        }
        for (int i = 0; i < static_cast<int>(fbm.size()); ++i)
        {
          foreignBoundaryMomentsFlat.at(p).at(fc).at(index+fbm.getDifferentIndex(fbm.getIndices(i))) = fbm.get(i);
        }

        index += size;
      }
    }
  }

  std::vector<std::vector<std::vector<double> > > myBoundaryMomentsFlat
   = vtkPMomentsHelper::ExchangeBoundaryInformation(this->NumberOfDifferentBasisFunctions, foreignBoundaryMomentsFlat, this->Dimension, this->Controller);

  // add them to the field
  for (int p = 0; p < numProcs; ++p)
  {
    for (int mc = 0; mc < myBoundaryCenters.at(p).size(); ++mc)
    {
      double center[3];
      for (int d = 0; d < this->Dimension; ++d)
      {
        center[d] = myBoundaryCenters.at(p).at(mc).at(d);
      }
      int ptId = grid->FindPoint(center);
//      // this would make the boundary zero
//      if(vtkMomentsHelper::IsCloseToBoundary(center, radius, this->FieldGlobalBounds, dimension))
//      {
//        for (int i = 0; i < NumberOfBasisFunctions; ++i)
//        {
//          output->GetPointData()->GetArray(radiusIndex * NumberOfBasisFunctions + i) ->SetTuple1(ptId, 0);
//          continue;
//        }
//      }
//      for (int i = 0; i < NumberOfBasisFunctions; ++i)
//      {
//        output->GetPointData()
//        ->GetArray(radiusIndex * NumberOfBasisFunctions + i)
//        ->SetTuple1(ptId, output->GetPointData()->GetArray(radiusIndex * NumberOfBasisFunctions + i)->GetTuple1(ptId) + myBoundaryMomentsFlat.at(p).at(mc).at(i));
//      }
      // add the moments to the corresponding array
      int index = 0;
      for (int k = 0; k < this->Order + 1; ++k)
      {
        int size = 0;
        if (this->Dimension == 2)
        {
          size = pow(this->Dimension, this->FieldRank) * (k + 1);
        }
        else
        {
          size = pow(this->Dimension, this->FieldRank) * 0.5 * (k + 1) * (k + 2);
        }
        vtkMomentsTensor dummyTensor = vtkMomentsTensor(this->Dimension, k + this->FieldRank, this->FieldRank);
        for (int i = 0; i < static_cast<int>(dummyTensor.size()); ++i)
        {
          if ( vtkMomentsHelper::isOrdered(dummyTensor.getIndices(i), this->FieldRank) )
          {
            std::string name = vtkMomentsHelper::getFieldNameFromTensorIndices(this->Radii.at(radiusIndex), dummyTensor.getIndices(i), this->FieldRank);
            output->GetPointData()->GetArray(name.c_str())
            ->SetTuple1(ptId, output->GetPointData()->GetArray(name.c_str())->GetTuple1(ptId) + myBoundaryMomentsFlat.at(p).at(mc).at(index+dummyTensor.getDifferentIndex(dummyTensor.getIndices(i))));
          }
        }
        index += size;
      }
    }
  }
}

void vtkPComputeMoments::ComputeSampling(int radiusIndex, vtkImageData* field, vtkImageData* grid, vtkImageData* output)
{
  double currentRadius = this->Radii.at(radiusIndex);
  // vtkResampleWithDataSet needs to be called symmetrically from all ranks. otherwise there is a deadlock
  // therefore, we call CenterStencil() maxNumberOfPoints times even if it exceeds this rank's numberOfPoints

  // TODO: should it nor be grid->GetNumberOfPoints()? and also in centerstencil?
  // get maximum number of points per rank
  double numberOfPoints, maxNumberOfPoints;
  numberOfPoints = grid->GetNumberOfPoints();
  this->Controller->AllReduce(&numberOfPoints, &maxNumberOfPoints, 1, vtkCommunicator::MAX_OP);
  //    std::cout << numberOfPoints << " " << maxNumberOfPoints << " \n";

  vtkImageData* stencil = vtkImageData::New();
  vtkMomentsHelper::BuildStencil(stencil, currentRadius,
                                 this->NumberOfIntegrationSteps, this->Dimension, field, this->NameOfPointData);
  for (int ptId = 0; ptId < maxNumberOfPoints; ++ptId)
  {
    double center[3];
    if ( ptId < grid->GetNumberOfPoints() )
    {
      // Get the xyz coordinate of the point in the grid dataset
      grid->GetPoint(ptId, center);
      //      // this would make the boundary zero
      //        if (!vtkMomentsHelper::IsCloseToBoundary(center, currentRadius, this->FieldGlobalBounds, this->Dimension) && this->CenterStencil(
      //                                center, field, stencil, this->NumberOfIntegrationSteps, this->NameOfPointData))
      if (this->CenterStencil(center, field, stencil, this->NumberOfIntegrationSteps, this->NameOfPointData))
      {
        // get all the moments
        std::vector<vtkMomentsTensor> tensorVector =
        vtkMomentsHelper::allMoments(this->Dimension, this->Order, this->FieldRank,
                                     currentRadius, center, stencil, this->NameOfPointData);

        // std::vector< vtkMomentsTensor > orthonormalTensorVector = this->orthonormalizeMoments(
        // this->Dimension, tensorVector, this->Radii.at(radiusIndex), stencil );
        //            tensorVector = orthonormalTensorVector;
        //            if(ptId == 100)
        //            {
        //                for( int i = 0; i < static_cast<int>(tensorVector.size()); ++i )
        //                {
        //                    tensorVector.at(i).print();
        //                }
        //                std::vector< vtkMomentsTensor > orthonormalTensorVector =
        //                orthonormalizeMoments( this->Dimension, tensorVector,
        //                this->Radii.at(radiusIndex) ); for( int i = 0; i <
        //                static_cast<int>(tensorVector.size());
        //                ++i )
        //                {
        //                    orthonormalTensorVector.at(i).print();
        //                }
        //            }

        // put them into the corresponding array
        for (int k = 0; k < static_cast<int>(tensorVector.size()); ++k)
        {
          for (int i = 0; i < static_cast<int>(tensorVector.at(k).size()); ++i)
          {
            vtkSmartPointer<vtkDataArray> output_array = output->GetPointData() ->GetArray(vtkMomentsHelper::getFieldNameFromTensorIndices(this->Radii.at(radiusIndex), tensorVector.at(k).getIndices(i), this->FieldRank).c_str());
            output_array->SetTuple1(ptId, tensorVector.at(k).get(i));
          }
        }
      }
      else
      {
        for (int i = 0; i < NumberOfDifferentBasisFunctions; ++i)
        {
          vtkSmartPointer<vtkDataArray> output_array = output->GetPointData()->GetArray(radiusIndex * NumberOfDifferentBasisFunctions + i);
          output_array->SetTuple1(ptId, 0.0);
        }
      }
      // cout<<ptId<<" center="<<center[0]<<" "<<center[1]<<" "<<center[2]<<"
      // "<<outPD->GetArray(0)->GetTuple(ptId)[0]<<"\n";
    }
    else
    {
      grid->GetPoint(0, center);
      this->CenterStencil(center, field, stencil, this->NumberOfIntegrationSteps, this->NameOfPointData);
    }
  }
  stencil->Delete();
}
/**
 * This function handles the moment computation using FFT in place of naive integration
 * @param radiusIndex: index of this radius in the radii vector
 * @param field: the dataset of which the moments are computed
 * @param grid: the uniform grid on which the moments are computed
 * @param output: this vtkImageData has the same topology as grid and will
 */
void vtkPComputeMoments::ComputePFFTMoments(
  int radiusIndex,
  vtkImageData* field,
  vtkImageData* grid,
  vtkImageData* output)
{
  double currentRadius = this->Radii.at(radiusIndex);

  /* Getting info from controller */
  int procID = this->Controller->GetLocalProcessId();
  int numProcs = this->Controller->GetNumberOfProcesses();
  // std::cout << "procID: " << procId << ", " << "numProcs: " << numProcs << "\n";

  /* Getting local and global extent */
  int localExtent[6];
  field->GetExtent(localExtent);

  // Gather local extent into allExtents as 6 * procID + extentIndex
  // The global extent will be the last 6 values
  int allExtents[6 * numProcs + 6];
  this->Controller->AllGather(localExtent, &allExtents[0], 6);

  for (int d = 0; d < 3; ++d)
  {
    allExtents[6*numProcs+2*d] = allExtents[2*d];
    allExtents[6*numProcs+2*d+1] = allExtents[2*d+1];
  }

  for (int p = 1; p < numProcs; ++p)
  {
    for (int d = 0; d < 3; ++d)
    {
      allExtents[6*numProcs+2*d] = std::min(allExtents[6*numProcs+2*d],allExtents[6*p+2*d]);
      allExtents[6*numProcs+2*d+1] = std::max(allExtents[6*numProcs+2*d+1],allExtents[6*p+2*d+1]);
    }
  }

  // if( procId == 0 )
  // {
  //   for (int p = 0; p < numProcs + 1; p++)
  //   {
  //     for (int d = 0; d < 6; d++)
  //     {
  //       std::cout << allExtents[p*6+d] << " ";
  //     }
  //     std::cout << "\n";
  //   }
  // }

  for (size_t i = 0; i < this->FieldGlobalExtent.size(); i++)
  {
    this->FieldGlobalExtent[i] = allExtents[6 * numProcs + i];
  }

  double fieldSpacing[3];
  field->GetSpacing(fieldSpacing);
  std::vector<int> kernelExtent (6, 0);
  for (int i = 0; i < this->Dimension; i++)
  {
    kernelExtent[2 * i] = -round(currentRadius / fieldSpacing[i]);
    kernelExtent[2 * i + 1] = round(currentRadius / fieldSpacing[i]);
  }

  // std::cout << "field extent " << procID << ": [" << localExtent[0] << ", "
  //                                                 << localExtent[1] << ", "
  //                                                 << localExtent[2] << ", "
  //                                                 << localExtent[3] << ", "
  //                                                 << localExtent[4] << ", "
  //                                                 << localExtent[5] << "]" << std::endl;
  // std::cout << "FieldGlobalExtent " << procID << ": [" << this->FieldGlobalExtent[0] << ", "
  //                                                      << this->FieldGlobalExtent[1] << ", "
  //                                                      << this->FieldGlobalExtent[2] << ", "
  //                                                      << this->FieldGlobalExtent[3] << ", "
  //                                                      << this->FieldGlobalExtent[4] << ", "
  //                                                      << this->FieldGlobalExtent[5] << "]" << std::endl;
  // std::cout << "kernel extent " << procID << ": [" << kernelExtent[0] << ", "
  //                                                 << kernelExtent[1] << ", "
  //                                                 << kernelExtent[2] << ", "
  //                                                 << kernelExtent[3] << ", "
  //                                                 << kernelExtent[4] << ", "
  //                                                 << kernelExtent[5] << "]" << std::endl;

  /* Pad the field data */
  std::vector<int> paddedFieldGlobalExtent (6, 0);
  vtkSmartPointer<vtkImageData> paddedField =
    vtkPMomentsHelper::padPField(field, this->Dimension, this->FieldGlobalExtent, kernelExtent, paddedFieldGlobalExtent, this->NameOfPointData, this->Controller);

  // std::cout << "paddedField " << procID << ": [" << paddedField->GetExtent()[0] << ", "
  //                                                << paddedField->GetExtent()[1] << ", "
  //                                                << paddedField->GetExtent()[2] << ", "
  //                                                << paddedField->GetExtent()[3] << ", "
  //                                                << paddedField->GetExtent()[4] << ", "
  //                                                << paddedField->GetExtent()[5] << "]" << std::endl;
  // std::cout << "paddedFieldGlobalExtent " << procID << ": [" << paddedFieldGlobalExtent[0] << ", "
  //                                                << paddedFieldGlobalExtent[1] << ", "
  //                                                << paddedFieldGlobalExtent[2] << ", "
  //                                                << paddedFieldGlobalExtent[3] << ", "
  //                                                << paddedFieldGlobalExtent[4] << ", "
  //                                                << paddedFieldGlobalExtent[5] << "]" << std::endl;

  /* Start setting up FFT for padded field data */
  /* Use a basic grid: proc_map is the mapping between cartesian indices and MPI ranks */
  int *proc_map = (int*)malloc(sizeof(int) * numProcs);
  for (int i = 0; i < numProcs; i++)
  {
    proc_map[i] = i;
  }

  /* Choose a decomposition: pdim is the number of processors per dimension */
  int *pdim = (int*)malloc(sizeof(int) * this->Dimension);
  int *pdim_trans = (int*)malloc(sizeof(int) * this->Dimension);
  int r = powf(numProcs, 1.0 / (double)this->Dimension) + 0.5;

  int root = numProcs;
  while (root > r) root /= 2;

  // if (!s) printf("Processor grid: ");
  int ptot = 1;
  int i0 = this->Dimension-1;
  for (int i = 0; i < this->Dimension-1; ++i)
  {
    pdim[i] = (((ptot*root) > numProcs) ? 1 : root);
    // if (!s) printf("%d x ",pdim[i]);

    ptot *= pdim[i];
    pdim_trans[i0] = pdim[i];
    i0--;
  }
  pdim[this->Dimension-1] = numProcs / ptot;
  pdim_trans[0] = pdim[this->Dimension-1];
  // if (!s) printf("%d\n", pdim[nd-1]);

  /* Determine processor index */
  int *pidx = (int*)malloc(sizeof(int) * this->Dimension);
  int *pidx_trans = (int*)malloc(sizeof(int) * this->Dimension);
  int idx = procID;
  int j0 = 0;
  for (int i = this->Dimension-1; i >= 0; --i)
  {
    pidx[i] = idx % pdim[i];
    idx /= pdim[i];

    pidx_trans[j0] = pidx[i];
    j0++;
  }

  /* Get the global/local fft data array dimension */
  int *dim_glob = (int*)malloc(sizeof(int) * this->Dimension);
  int dim_local[3] = { 1, 1, 1 };
  for (int i = 0; i < this->Dimension; i++)
  {
    dim_glob[i] = paddedFieldGlobalExtent[2 * i + 1] - paddedFieldGlobalExtent[2 * i] + 1;
    dim_local[i] = dim_glob[i] / pdim_trans[i];
  }

  // std::cout << "dim_glob " << procID << ": [" << dim_glob[0] << ", " << dim_glob[1] << "]" << std::endl;
  // std::cout << "pdim " << procID << ": [" << pdim[0] << ", " << pdim[1] << "]" << std::endl;
  // std::cout << "dim_local " << procID << ": [" << dim_local[0] << ", " << dim_local[1] << "]" << std::endl;
  // std::cout << "pidx_trans " << procID << ": [" << pidx_trans[0] << ", " << pidx_trans[1] << "]" << std::endl;

  int voiext[6] = { 0, 0, 0, 0, 0, 0 };
  for (int i = 0; i < this->Dimension; i++)
  {
    voiext[2 * i] = pidx_trans[i] * dim_local[i];
    voiext[2 * i + 1] = pidx_trans[i] * dim_local[i] + dim_local[i] - 1;
  }

  // std::cout << "voiext " << procID << ": [" << voiext[0] << ", "
  //                                           << voiext[1] << ", "
  //                                           << voiext[2] << ", "
  //                                           << voiext[3] << ", "
  //                                           << voiext[4] << ", "
  //                                           << voiext[5] << "]" << std::endl;

  // const char* filename = "./paddedField.pvti";
  // vtkNew<vtkXMLPImageDataWriter> writer;
  // writer->SetInputData(paddedField);
  // int npieces = numProcs;
  // writer->SetNumberOfPieces(npieces);
  // int pperrank = npieces / numProcs;
  // int start = pperrank * procID;
  // int end = start + pperrank - 1;
  // writer->SetStartPiece(start);
  // writer->SetEndPiece(end);
  // writer->SetFileName(filename);
  // writer->SetUseSubdirectory(true);
  // writer->SetDataModeToAscii();
  // writer->UpdateInformation();
  // writer->Write();

  /* Resample to equal sized blocks on each rank */
  vtkNew<vtkImageData> voi_paddedField;
  voi_paddedField->SetOrigin(0, 0, 0);
  voi_paddedField->SetSpacing(paddedField->GetSpacing());
  voi_paddedField->SetExtent(voiext);

  vtkNew<vtkPResampleWithDataSet> resamplePaddedField;
  resamplePaddedField->MarkBlankPointsAndCellsOff();
  resamplePaddedField->SetController(this->Controller);
  resamplePaddedField->SetInputData(voi_paddedField);
  resamplePaddedField->SetSourceData(paddedField);
  resamplePaddedField->PassPointArraysOn();
  resamplePaddedField->UpdateInformation();
  resamplePaddedField->Update();
  vtkSmartPointer<vtkImageData> paddedFieldVOI = vtkImageData::SafeDownCast(resamplePaddedField->GetOutput());

  // std::cout << "paddedFieldVOI " << procID << ": [" << paddedFieldVOI->GetExtent()[0] << ", "
  //                                                << paddedFieldVOI->GetExtent()[1] << ", "
  //                                                << paddedFieldVOI->GetExtent()[2] << ", "
  //                                                << paddedFieldVOI->GetExtent()[3] << ", "
  //                                                << paddedFieldVOI->GetExtent()[4] << ", "
  //                                                << paddedFieldVOI->GetExtent()[5] << "]" << std::endl;

  // filename = "./paddedFieldVOI.pvti";
  // writer->SetFileName(filename);
  // writer->SetInputConnection(resamplePaddedField->GetOutputPort());
  // writer->Write();

  // paddedFieldLocalNumPts is the local size of the data array
  int paddedFieldLocalNumPts = paddedFieldVOI->GetNumberOfPoints();

  // Create the kissfft forward and inverse plan
  // kiss_fftnd_alloc( dim_glob, ndim_glob, forward/inverse, 0, 0 )
  // dim_glob: array of dimensions
  // ndim_glob: number of dimensions
  // inverse: 0 = forward; 1 = inverse
  dfft_plan moment_plan;
  dfft::create_plan(&moment_plan, this->Dimension, dim_glob, nullptr, nullptr, pdim,
                    pidx, 0, 0, 0, *this->MPI_Communicator, proc_map);

  /* Initialize & Execute plan on fieldFFT */
  std::vector<dfft::complex_t*> fieldFFT;
  vtkSmartPointer<vtkDataArray> paddedFieldVOIArray = paddedFieldVOI->GetPointData()->GetArray(this->NameOfPointData.c_str());
  int numberOfComponents = paddedFieldVOIArray->GetNumberOfComponents();
  for (int numComp = 0; numComp < numberOfComponents; numComp++)
  {
    fieldFFT.push_back(dfft::allocate_cpx(paddedFieldLocalNumPts));

    // Fill in values for fieldFFT
    for (int j = 0; j < paddedFieldLocalNumPts; j++)
    {
      RE(fieldFFT[numComp][j]) = paddedFieldVOIArray->GetTuple(j)[numComp];
      IM(fieldFFT[numComp][j]) = 0.0;
    }
    dfft::execute(fieldFFT[numComp], fieldFFT[numComp], 0, moment_plan);
  }

  // for (int i = 0; i < paddedFieldLocalNumPts; i++)
  // {
  //   std::cout << "P" << procID << ": " << RE(fieldFFT[0][i]) << ", " << IM(fieldFFT[0][i]) << std::endl;
  // }

  /* Setup imagedata for kernel */
  int paddedKernelExtent[6] = { 0, 0, 0, 0, 0, 0 };
  int paddedFieldVOIExtent[6];
  paddedFieldVOI->GetExtent(paddedFieldVOIExtent);
  for (int i = 0; i < this->Dimension; i++)
  {
    paddedKernelExtent[2 * i] = paddedFieldVOIExtent[2 * i] + kernelExtent[2 * i];
    paddedKernelExtent[2 * i + 1] = paddedFieldVOIExtent[2 * i + 1] + kernelExtent[2 * i];
  }

  vtkNew<vtkImageData> kernel;
  kernel->SetOrigin(0, 0, 0);
  kernel->SetSpacing(field->GetSpacing());
  kernel->SetExtent(paddedKernelExtent);

  // std::cout << "kernel padded extent " << procID << ": [" << kernel->GetExtent()[0] << ", "
  //                                                << kernel->GetExtent()[1] << ", "
  //                                                << kernel->GetExtent()[2] << ", "
  //                                                << kernel->GetExtent()[3] << ", "
  //                                                << kernel->GetExtent()[4] << ", "
  //                                                << kernel->GetExtent()[5] << "]" << std::endl;

  vtkNew<vtkDoubleArray> kernelArray;
  kernelArray->SetName("kernel");
  kernelArray->SetNumberOfComponents(1);
  kernelArray->SetNumberOfTuples(kernel->GetNumberOfPoints());

  kernel->GetPointData()->SetScalars(kernelArray);

  /* Start setting up FFT for kernel */
  dfft::complex_t* kernelFFT = dfft::allocate_cpx(paddedFieldLocalNumPts);

  /* Start setting up IFFT for the result */
  dfft::complex_t* result = dfft::allocate_cpx(paddedFieldLocalNumPts);

  vtkNew<vtkImageData> fft_output;
  this->BuildOutput(paddedFieldVOI, fft_output);

  /*
   * Initialize kernel for each basis function
   * Initialize kernelFFT from the kernel that's padded to the same size as padded field
   * Apply FFT on the kernelFFT
   * Initialize result from kernelFFT according to the Cross-Correlation Theorem
   * Apply IFFT on the result
   * Populate the output from the result with normalization
   */
  for (int i = 0; i < this->NumberOfBasisFunctions; i++)
  {
    double argument[3];
    double relArgument[3];
    kernelArray->Fill(0.0);

    std::vector<int> indices = vtkMomentsHelper::getTensorIndicesFromFieldIndex(
      i, this->Dimension, this->Order, this->FieldRank); // radiusIndex *
                                                         // this->NumberOfBasisFunctions + i); //
                                                         // given multiple radii this does not give
                                                         // correct answers
    if( vtkMomentsHelper::isOrdered(indices, this->FieldRank) )
    {
      int mrank = static_cast<int>(indices.size());

      // Fill in values for kernel
      for (int ptId = 0; ptId < paddedFieldLocalNumPts; ++ptId)
      {
        // TODO: Check if the output of getpoint need to add origin on top of it
        kernel->GetPoint(ptId, argument);
        for (int d = 0; d < 3; ++d)
        {
          relArgument[d] = 1. / currentRadius * argument[d];
        }
        if (vtkMath::Norm(relArgument) <= 1)
        //if (vtkMath::Norm(argument) <= currentRadius)
        {
          double faktor = 1;
          for (int k = 0; k < mrank - this->FieldRank; ++k)
          {
            faktor *= relArgument[indices.at(k)];
          }
          kernelArray->SetTuple1(ptId, faktor);
          // if (i == 0) std::cout << "P: " << argument[0] << ", " << argument[1] << ", " << argument[2] << std::endl;
          // if (i == 0) std::cout << "P: " << ptId << ", " << faktor << std::endl;
        }
      }

      // if (i == 0)
      // {
      //   filename = "./kernel.pvti";
      //   writer->SetFileName(filename);
      //   writer->SetInputData(kernel);
      //   writer->Write();
      // }

      // Initialize & Execute plan on kernelFFT
      vtkSmartPointer<vtkDataArray> kernel_array = kernel->GetPointData()->GetScalars();
      for (int j = 0; j < paddedFieldLocalNumPts; j++)
      {
        RE(kernelFFT[j]) = kernel_array->GetTuple1(j);
        IM(kernelFFT[j]) = 0.0;
      }
      dfft::execute(kernelFFT, kernelFFT, 0, moment_plan);

      // Multiply the 2 FFT together according to Cross-Correlation Theorem
      vtkMomentsTensor dummyTensor = vtkMomentsTensor(this->Dimension, mrank, this->FieldRank);
      int compIndex = dummyTensor.getFieldIndex(dummyTensor.getIndex(indices));
      for (int j = 0; j < paddedFieldLocalNumPts; j++)
      {
        // Cross-correction: complex conjugate the FFT of kernel
        RE(result[j]) = (RE(fieldFFT[compIndex][j]) * RE(kernelFFT[j]) +
          IM(fieldFFT[compIndex][j]) * IM(kernelFFT[j]));
        IM(result[j]) = (RE(fieldFFT[compIndex][j]) * (-IM(kernelFFT[j])) +
          RE(kernelFFT[j]) * IM(fieldFFT[compIndex][j]));
      }
      dfft::execute(result, result, 1, moment_plan);

      // FFTW/KissFFT/DFFTLIB performs unnormalized FFT & IFFT
      double normalize = 1.0;
      double radiusScaling = 1.0;
      double spacings = 1.0;
      for (int j = 0; j < this->Dimension; j++)
      {
        normalize *= (paddedFieldGlobalExtent[2 * j + 1] - paddedFieldGlobalExtent[2 * j] + 1);
        radiusScaling *= currentRadius;
        spacings *= field->GetSpacing()[j];
      }

      vtkSmartPointer<vtkDataArray> fft_output_array = fft_output->GetPointData()->GetArray(vtkMomentsHelper::getFieldNameFromTensorIndices(this->Radii.at(radiusIndex), indices, this->FieldRank).c_str());
      for (int j = 0; j < paddedFieldLocalNumPts; j++)
      {
        // Set the output array for each corresponding moment array
        fft_output_array->SetTuple1(j, RE(result[j]) / normalize / radiusScaling * spacings);
      }
    }
  }

  // filename = "./fft_output.pvti";
  // writer->SetFileName(filename);
  // writer->SetInputData(fft_output);
  // writer->Write();

  // Translate fft_output origin to match field origin
  vtkNew<vtkImageTranslateExtent> transOutput;
  transOutput->SetTranslation(this->FieldGlobalExtent[0] - paddedFieldGlobalExtent[0], this->FieldGlobalExtent[2] - paddedFieldGlobalExtent[2], this->FieldGlobalExtent[4] - paddedFieldGlobalExtent[4]);
  transOutput->SetInputData(fft_output);
  transOutput->Update();

  int gext[6] = { 0, 0, 0, 0, 0, 0 };
  for (int i = 0; i < this->Dimension; i++)
  {
    gext[2 * i] = paddedFieldGlobalExtent[2 * i] + this->FieldGlobalExtent[2 * i] - paddedFieldGlobalExtent[2 * i];
    gext[2 * i + 1] = paddedFieldGlobalExtent[2 * i + 1] + this->FieldGlobalExtent[2 * i] - paddedFieldGlobalExtent[2 * i];

    paddedFieldGlobalExtent[2 * i] = gext[2 * i];
    paddedFieldGlobalExtent[2 * i + 1] = gext[2 * i + 1];
  }

  // Manually set the global extent
  vtkNew<vtkTrivialProducer> trivialPO;
  trivialPO->SetOutput(transOutput->GetOutput());
  trivialPO->SetWholeExtent(gext);
  trivialPO->Update();
  vtkSmartPointer<vtkImageData> trivialOutput = vtkImageData::SafeDownCast(trivialPO->GetOutputDataObject(0));
  trivialOutput->SetOrigin(field->GetOrigin());

  // std::cout << "transOutput " << procID << ": [" << transOutput->GetOutput()->GetExtent()[0] << ", "
  //                                                << transOutput->GetOutput()->GetExtent()[1] << ", "
  //                                                << transOutput->GetOutput()->GetExtent()[2] << ", "
  //                                                << transOutput->GetOutput()->GetExtent()[3] << ", "
  //                                                << transOutput->GetOutput()->GetExtent()[4] << ", "
  //                                                << transOutput->GetOutput()->GetExtent()[5] << "]" << std::endl;
  // std::cout << "trivialPO " << procID << ": [" << trivialPO->GetWholeExtent()[0] << ", "
  //                                              << trivialPO->GetWholeExtent()[1] << ", "
  //                                              << trivialPO->GetWholeExtent()[2] << ", "
  //                                              << trivialPO->GetWholeExtent()[3] << ", "
  //                                              << trivialPO->GetWholeExtent()[4] << ", "
  //                                              << trivialPO->GetWholeExtent()[5] << "]" << std::endl;

  // filename = "./transOutput.pvti";
  // writer->SetFileName(filename);
  // writer->SetInputData(trivialOutput);
  // writer->Write();

  /* Resample back to dimension of field and put back duplication points */
  for (int i = 0; i < this->Dimension; i++)
  {
    voiext[2 * i + 1] += 1;
  }

  vtkNew<vtkImageData> voi_fftOutput;
  voi_fftOutput->SetOrigin(field->GetOrigin());
  voi_fftOutput->SetSpacing(field->GetSpacing());
  voi_fftOutput->SetExtent(voiext);

  vtkNew<vtkPResampleWithDataSet> resampleFFTOutput;
  resampleFFTOutput->MarkBlankPointsAndCellsOff();
  resampleFFTOutput->SetController(this->Controller);
  resampleFFTOutput->SetInputData(voi_fftOutput);
  resampleFFTOutput->SetSourceData(trivialOutput);
  resampleFFTOutput->PassPointArraysOn();
  resampleFFTOutput->UpdateInformation();
  resampleFFTOutput->Update();
  vtkSmartPointer<vtkImageData> fftOutputVOI = vtkImageData::SafeDownCast(resampleFFTOutput->GetOutput());

  // filename = "./fftOutputVOI.pvti";
  // writer->SetFileName(filename);
  // writer->SetInputData(fftOutputVOI);
  // writer->Write();

  /* Resample to grid structure if grid is defined */
  vtkNew<vtkPResampleWithDataSet> resample;
  resample->MarkBlankPointsAndCellsOff();
  resample->SetController(this->Controller);
  resample->SetInputData(grid);
  resample->SetSourceData(fftOutputVOI);
  resample->PassPointArraysOn();
  resample->UpdateInformation();
  resample->Update();
  vtkSmartPointer<vtkImageData> resampleData = vtkImageData::SafeDownCast(resample->GetOutput());

  // filename = "./resampleOutputData.pvti";
  // writer->SetFileName(filename);
  // writer->SetInputData(resampleData);
  // writer->Write();

  int outputNumPts = output->GetNumberOfPoints();
  for (int i = 0; i < this->NumberOfDifferentBasisFunctions; i++)
  {
    vtkSmartPointer<vtkDataArray> outputArray = output->GetPointData()->GetArray(radiusIndex * this->NumberOfDifferentBasisFunctions + i);
    vtkSmartPointer<vtkDataArray> resampleDataArray = resampleData->GetPointData()->GetArray(outputArray->GetName());
    for (vtkIdType j = 0; j < outputNumPts; j++)
    {
      outputArray->SetTuple1(j, resampleDataArray->GetTuple1(j));
    }
  }

  this->Controller->Barrier();

  /* Cleaning up the fft objects */
  for (int numComp = 0; numComp < numberOfComponents; numComp++)
  {
    dfft::free_cpx(fieldFFT[numComp]);
  }
  dfft::free_cpx(kernelFFT);
  dfft::free_cpx(result);

  dfft::destroy_plan(moment_plan);

  free(pidx);
  free(pidx_trans);
  free(pdim);
  free(dim_glob);

  free(proc_map);
}

void vtkPComputeMoments::ComputeVtkm(int radiusIndex, vtkImageData* field, vtkImageData* grid, vtkImageData* output)
{
  std::cout << "vtkPComputeMoments::ComputeVtkm \n";
  int numProcs = this->Controller->GetNumberOfProcesses();
  double radius = this->Radii.at(radiusIndex);
  double gridBounds[6];
  grid->GetBounds(gridBounds);
  double fieldBounds[6];
  field->GetBounds(fieldBounds);
  double allFieldBounds[6 * numProcs + 6];
  vtkPMomentsHelper::GetAllBounds(allFieldBounds, field, this->Controller);

  // compute (possibly partial) moments for all centers in this proc
  this->Superclass::ComputeVtkm(radiusIndex, grid, field, output);
}
/**
 * This Method switches between different ways to computes the moments.
 * @param radiusIndex: index of this radius in the radii vector
 * @param grid: the uniform grid on which the moments are computed
 * @param field: function of which the moments are computed
 * @param output: this vtkImageData has the same topology as grid and will contain numberOfFields
 * scalar fields, each containing one moment at all positions
 */
void vtkPComputeMoments::Compute(
  int radiusIndex, vtkImageData* grid, vtkImageData* field, vtkImageData* output)
{
  // std::cout << "vtkPComputeMoments::Compute \n";
  if (this->UseFFT)
  {
    if (!this->MPI_Communicator)
    {
      vtkErrorMacro("There is no MPI communicator set.");
    }
    this->CheckRanksValidity();    
    this->ComputePFFTMoments(radiusIndex, field, grid, output);
    normalizeMomentsImageData(radiusIndex, output);
    
  }
  else if (this->UseOriginalResolution)
  {
    if (this->UseGPU)
    {
      // compute (possibly partial) moments for all centers in this proc
      this->Superclass::ComputeVtkm(radiusIndex, grid, field, output);      
      // handle the points in vicinity of the boundary
      this->ComputeBoundaryOrigRes(radiusIndex, grid, field, output);
      normalizeMomentsImageData(radiusIndex, output);
    }
    else
    {
      // compute (possibly partial) moments for all centers in this proc
      this->Superclass::ComputeOrigRes(radiusIndex, grid, field, output);
      // handle the points in vicinity of the boundary
      this->ComputeBoundaryOrigRes(radiusIndex, grid, field, output);
    }
  }
  else
  {
    this->ComputeSampling(radiusIndex, field, grid, output);
  }
  //#ifdef MYDEBUG
  //    std::ostream stream(std::cout.rdbuf());
  //    std::cout<<"output=";
  //    output->PrintSelf(stream, vtkIndent(0));
  //    std::cout<<"\n";
  //#endif
}

/**
 * main executive of the program, reads the input, calles the
 * functions, and produces the utput.
 * @param inputVector: the input information
 * @param outputVector: the output information
 */
int vtkPComputeMoments::RequestData(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  if (!this->Controller)
  {
    vtkErrorMacro("There is no controller set.");
  }

  // get the info objects
  vtkInformation* fieldInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation* gridInfo = inputVector[1]->GetInformationObject(0);
  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkImageData* field = vtkImageData::SafeDownCast(fieldInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* grid;
  if (gridInfo)
  {
    grid = vtkImageData::SafeDownCast(gridInfo->Get(vtkDataObject::DATA_OBJECT()));
  }
  else
  {
    grid = field;
    gridInfo = fieldInfo;
  }
  vtkImageData* output = vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (field)
  {
    this->InterpretField(field);
    this->CheckValidity(field, grid);
    this->CheckGridValidity(field, grid, gridInfo);
    this->BuildOutput(grid, output);

    int radiiSize = static_cast<int>(this->Radii.size());
    for (int radiusIndex = 0; radiusIndex < radiiSize; ++radiusIndex)
    {
      this->Compute(radiusIndex, grid, field, output);
    }
  }
  return 1;
}

void vtkPComputeMoments::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Number of processes =  " << this->Controller->GetNumberOfProcesses() << "\n";
  os << indent << "ID of this process =  " << this->Controller->GetLocalProcessId() << "\n";
}
