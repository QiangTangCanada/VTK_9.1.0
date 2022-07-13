/*=========================================================================

 Program:   Visualization Toolkit
 Module:    vtkPMomentsHelper.cxx

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

 Copyright 2007. Los Alamos National Security, LLC.
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

#include "vtkDoubleArray.h"
#include "vtkMomentsHelper.h"
#include "vtkPMomentsHelper.h"
#include "vtkPointData.h"
#include "../ParallelDIY2/vtkPResampleWithDataSet.h"
#include "vtkImageConstantPad.h"
#include "vtkImageData.h"
#include "vtkImageReslice.h"
#include "vtkImageTranslateExtent.h"
#include "vtkMomentsTensor.h"
#include "vtkNew.h"
#include "vtkTrivialProducer.h"
#include <vector>

//------------------------------------------------------------------------------------------
vtkSmartPointer<vtkImageData> vtkPMomentsHelper::padPField(
                                                           vtkImageData* field,
                                                           int dimension,
                                                           std::vector<int> fieldGlobalExtent,
                                                           std::vector<int> kernelExtent,
                                                           std::vector<int>& paddedFieldGlobalExtent,
                                                           std::string nameOfPointData,
                                                           vtkMultiProcessController* /*controller*/)
{
  // Manually compute which end of the extent to pad
  // Only grow external boundaries and not boundaries between partitions
  // Also the final data array size needs to be a power of 2
  int lext[6];
  field->GetExtent(lext);

  // field->GetDimensions(dims); // returns local dimensions
  int dims[3] = { 0, 0, 0 };
  int k_dims[3] = { 0, 0, 0 };
  int max_dim = 0;
  for (int i = 0; i < dimension; i++)
  {
    dims[i] = fieldGlobalExtent[2 * i + 1] - fieldGlobalExtent[2 * i] + 1;
    k_dims[i] = kernelExtent[2 * i + 1] - kernelExtent[2 * i] + 1;
    max_dim = std::max(max_dim, dims[i]);
  }

  int global_origin_edge[3] = { 0, 0, 0 };
  for (int i = 0; i < dimension; i++)
  {
    int padSize_pattern = k_dims[i] / 2;
    if (lext[2 * i] == fieldGlobalExtent[2 * i])
    {
      lext[2 * i] -= padSize_pattern;
      global_origin_edge[i] = 1;
    }
    paddedFieldGlobalExtent[2 * i] = fieldGlobalExtent[2 * i] - padSize_pattern;

    int currentSize = dims[i] + padSize_pattern;
    int maxSize = max_dim + padSize_pattern;
    int padSize_power2 = pow(2, ceil(log2(maxSize))) - currentSize;
    if (lext[2 * i + 1] == fieldGlobalExtent[2 * i + 1])
    {
      lext[2 * i + 1] += padSize_power2;
    }
    paddedFieldGlobalExtent[2 * i + 1] = fieldGlobalExtent[2 * i + 1] + padSize_power2;
  }

  vtkNew<vtkImageData> pad;
  pad->SetOrigin( 0, 0, 0 );
  pad->SetSpacing( field->GetSpacing() );
  pad->SetExtent( lext );

  vtkDataArray* origArray = field->GetPointData()->GetArray(nameOfPointData.c_str());

  vtkNew<vtkDoubleArray> paddedArray;
  paddedArray->SetName( nameOfPointData.c_str() );
  paddedArray->SetNumberOfComponents( origArray->GetNumberOfComponents() );
  paddedArray->SetNumberOfTuples( pad->GetNumberOfPoints() );
  paddedArray->Fill(0.0);

  const int* tmp = field->GetDimensions();
  std::vector<int> origSize = std::vector<int>(tmp, tmp + 3);

  tmp = pad->GetDimensions();
  std::vector<int> paddedSize = std::vector<int>(tmp, tmp + 3);

  for (vtkIdType i = 0; i < field->GetNumberOfPoints(); i++)
  {
    std::vector<int> coord = vtkMomentsHelper::getCoord(i, origSize);
    for (size_t j = 0; j < coord.size(); j++)
    {
      if (global_origin_edge[j] == 1)
      {
        coord[j] += k_dims[j] / 2;
      }
    }

    vtkIdType index = vtkMomentsHelper::getArrayIndex(coord, paddedSize);
    paddedArray->SetTuple(index, origArray->GetTuple(i));
  }

  pad->GetPointData()->AddArray(paddedArray);

  // Translate to the origin
  vtkNew<vtkImageTranslateExtent> trans;
  trans->SetTranslation(-paddedFieldGlobalExtent[0], -paddedFieldGlobalExtent[2], -paddedFieldGlobalExtent[4]);
  trans->SetInputData(pad);
  trans->Update();

  int gext[6] = { 0, 0, 0, 0, 0, 0 };
  for (int i = 0; i < dimension; i++)
  {
    gext[2 * i] = paddedFieldGlobalExtent[2 * i] - paddedFieldGlobalExtent[2 * i];
    gext[2 * i + 1] = paddedFieldGlobalExtent[2 * i + 1] - paddedFieldGlobalExtent[2 * i];

    paddedFieldGlobalExtent[2 * i] = gext[2 * i];
    paddedFieldGlobalExtent[2 * i + 1] = gext[2 * i + 1];
  }

  // Manually set the global extent
  vtkNew<vtkTrivialProducer> tp;
  tp->SetOutput(trans->GetOutput());
  tp->SetWholeExtent(gext);
  // tp->UpdateInformation();
  tp->Update();

  vtkSmartPointer<vtkImageData> output;
  output.TakeReference(vtkImageData::SafeDownCast(tp->GetOutputDataObject(0)));
  output->SetOrigin(0,0,0);
  output->Register(nullptr);
  return output;
}

void vtkPMomentsHelper::GetAllBounds(double allBounds[], vtkImageData* field, vtkMultiProcessController* controller)
{
  double bounds[6];
  field->GetBounds(bounds);

  //int procId = controller->GetLocalProcessId();
  int numProcs = controller->GetNumberOfProcesses();
  //    std::cout<<"procId"<<procId<<"numProcs"<<numProcs<<"\n";

  controller->AllGather(bounds, &allBounds[0], 6);

  for (int d = 0; d < 3; ++d)
  {
    allBounds[6 * numProcs + 2 * d] = allBounds[2 * d];
    allBounds[6 * numProcs + 2 * d + 1] = allBounds[2 * d + 1];
  }

  for (int p = 1; p < numProcs; ++p)
  {
    for (int d = 0; d < 3; ++d)
    {
      allBounds[6 * numProcs + 2 * d] =
      std::min(allBounds[6 * numProcs + 2 * d], allBounds[6 * p + 2 * d]);
      allBounds[6 * numProcs + 2 * d + 1] =
      std::max(allBounds[6 * numProcs + 2 * d + 1], allBounds[6 * p + 2 * d + 1]);
    }
  }
  //    if( procId == 0 )
  //    {
  //      for (int p = 0; p < numProcs + 1; p++)
  //      {
  //        for (int d = 0; d < 6; d++)
  //        {
  //          std::cout<<allBounds[p*6+d]<<" ";
  //        }
  //        std::cout<<"\n";
  //      }
  //    }
}

void vtkPMomentsHelper::GetAllBoundsWithoutGhostCells(double allBounds[], vtkImageData* field, vtkMultiProcessController* controller)
{
  //int procId = controller->GetLocalProcessId();
  int numProcs = controller->GetNumberOfProcesses();
  vtkPMomentsHelper::GetAllBounds(allBounds, field, controller);
//  for (int p = 0; p < numProcs; ++p)
//  {
//    for (int d = 0; d < 3; ++d)
//    {
//      std::cout<<p<<" "<<d<<" "<<allBounds[6 * p + 2 * d]<<" "<<allBounds[6 * p + 2 * d + 1]<<endl;
//    }
//  }
//  std::cout<<"field->HasAnyGhostCells() "<<field->HasAnyGhostCells()<<"\n";
  if (!field->HasAnyGhostCells())
  {
    return;
  }

  // ghost cell problem
  double spacing[3];
  field->GetSpacing(spacing);
  for (int p = 0; p < numProcs; ++p)
  {
    for (int d = 0; d < 3; ++d)
    {
      if (allBounds[6 * numProcs + 2 * d] != allBounds[6 * p + 2 * d])
      {
        allBounds[6 * p + 2 * d] = allBounds[6 * p + 2 * d] + spacing[d];
      }
      if (allBounds[6 * numProcs + 2 * d + 1] != allBounds[6 * p + 2 * d + 1])
      {
        allBounds[6 * p + 2 * d + 1] = allBounds[6 * p + 2 * d + 1] - spacing[d];
      }
    }
  }
//    for (int p = 0; p < numProcs; ++p)
//    {
//      for (int d = 0; d < 3; ++d)
//      {
//        std::cout<<p<<" "<<d<<" "<<allBounds[6 * p + 2 * d]<<" "<<allBounds[6 * p + 2 * d + 1]<<endl;
//      }
//    }
}

std::vector<std::vector<std::vector<double> > > vtkPMomentsHelper::GetBoundaryCenters(double allBounds[], vtkImageData* field, int dimension, double radius, vtkMultiProcessController* controller)
{
  int procId = controller->GetLocalProcessId();
  int numProcs = controller->GetNumberOfProcesses();
  double localBounds[6];
  for (int d = 0; d < 6; ++d)
  {
    localBounds[d] = allBounds[6 * procId + d];
  }
  // this vector contains a vector for each proc. with the centers that are close to that proc's
  // boundary
  std::vector<std::vector<std::vector<double> > > myBoundaryCenters(numProcs);
  for (int ptId = 0; ptId < field->GetNumberOfPoints(); ++ptId)
  {
    double center[3];
    field->GetPoint(ptId, center);
    if (vtkMomentsHelper::IsCloseToBoundary(center, radius, localBounds, dimension))
    {
      for (int p = 0; p < numProcs; ++p)
      {
        if (p != procId)
        {
          bool isInThisProc = true;
          for (int d = 0; d < dimension; ++d)
          {
            if ((center[d] + radius - 1e-10 < allBounds[6 * p + 2 * d] &&
                 center[d] - radius - 1e-10 < allBounds[6 * p + 2 * d]) ||
                (center[d] + radius + 1e-10 > allBounds[6 * p + 2 * d + 1] &&
                 center[d] - radius + 1e-10 > allBounds[6 * p + 2 * d + 1]))
            {
              isInThisProc = false;
            }
          }
          if (isInThisProc)
          {
            myBoundaryCenters.at(p).push_back(std::vector<double>(3));
            for (int d = 0; d < dimension; ++d)
            {
              myBoundaryCenters.at(p).back().at(d) = center[d];
            }
          }
        }
      }
    }
  }
  return myBoundaryCenters;
}


std::vector<std::vector<std::vector<double> > > vtkPMomentsHelper::GetBoundaryInformation(double allBounds[], vtkImageData* localMaxData, int dimension, double radius, vtkMultiProcessController* controller)
{
  int procId = controller->GetLocalProcessId();
  int numProcs = controller->GetNumberOfProcesses();
  double localBounds[6];
  double globalBounds[6];
  for (int d = 0; d < 6; ++d)
  {
    localBounds[d] = allBounds[6 * procId + d];
    globalBounds[d] = allBounds[6 * numProcs + d];
  }
  // this vector contains a vector for each proc. with the centers that are close to that proc's
  // inner boundary (no global boundary points and no ghost cells)
  // we actually only need to exchange the ones with similarity > 0, but that may cause a deadlock in mpi send because it is not commutative
  std::vector<std::vector<std::vector<double> > > myBoundaryCenters(numProcs);
  for (int ptId = 0; ptId < localMaxData->GetNumberOfPoints(); ++ptId)
  {
    double center[3];
    localMaxData->GetPoint(ptId, center);
    if (vtkMomentsHelper::IsCloseToBoundary(center, radius, localBounds, dimension) && !vtkMomentsHelper::IsCloseToBoundary(center, radius, globalBounds, dimension) && localMaxData->GetPointData()->GetArray("vtkGhostType")->GetTuple1(ptId) == 0)
    {
//      if (procId == 0)
//      {
//        cout<<ptId<<endl;
//      }
      for (int p = 0; p < numProcs; ++p)
      {
        if (p != procId)
        {
          bool isInThisProc = true;
          for (int d = 0; d < dimension; ++d)
          {
            if ((center[d] + radius - 1e-10 < allBounds[6 * p + 2 * d] &&
                 center[d] - radius - 1e-10 < allBounds[6 * p + 2 * d]) ||
                (center[d] + radius + 1e-10 > allBounds[6 * p + 2 * d + 1] &&
                 center[d] - radius + 1e-10 > allBounds[6 * p + 2 * d + 1]))
            {
              isInThisProc = false;
            }
          }
          if (isInThisProc)
          {
            myBoundaryCenters.at(p).push_back(std::vector<double>(5));
            for (int d = 0; d < dimension; ++d)
            {
              myBoundaryCenters.at(p).back().at(d) = center[d];
            }
            myBoundaryCenters.at(p).back().at(3) = localMaxData->GetPointData()->GetArray("localMaxRadius")->GetTuple1(ptId);
            myBoundaryCenters.at(p).back().at(4) = localMaxData->GetPointData()->GetArray("localMaxValue")->GetTuple1(ptId);
          }
        }
      }
    }
  }
  return myBoundaryCenters;
}


// exchange boundary centers between nodes. distance is commutative. so we can send and receive
// only to the nodes that we share centers with
std::vector<std::vector<std::vector<double> > >  vtkPMomentsHelper::ExchangeBoundaryInformation(int numberOfInformation, std::vector<std::vector<std::vector<double> > > myBoundaryCenters, int /*dimension*/, vtkMultiProcessController* controller)
{
  int procId = controller->GetLocalProcessId();
  int numProcs = controller->GetNumberOfProcesses();
  const int MY_RETURN_VALUE_MESSAGE = 0x11;
  std::vector<std::vector<std::vector<double> > > foreignBoundaryCenters(numProcs);
  for (int p = 0; p < numProcs; ++p)
  {
    if (myBoundaryCenters.at(p).size() > 0)
    {
      int numMyBoundaryCenters = myBoundaryCenters.at(p).size();
      double myBoundaryCentersArray[numMyBoundaryCenters * numberOfInformation];
      for (int mc = 0; mc < numMyBoundaryCenters; ++mc)
      {
        for (int d = 0; d < numberOfInformation; ++d)
        {
          *(myBoundaryCentersArray + numberOfInformation * mc + d) = myBoundaryCenters.at(p).at(mc).at(d);
        }
      }
      //    for (int mc = 0; mc < numMyBoundaryCenters; ++mc)
      //    {
      //      std::cout<<"myBoundaryCentersArray: "<<procId<<" "<<p<<" "<<*(myBoundaryCentersArray+mc)<<","<<*(myBoundaryCentersArray+mc+1)<<"\n";
      //    }

      int numForeignBoundaryCenters;
      if (procId > p)
      {
        controller->Send(&numMyBoundaryCenters, 1, p, MY_RETURN_VALUE_MESSAGE);
        controller->Receive(&numForeignBoundaryCenters, 1, p, MY_RETURN_VALUE_MESSAGE);
      }
      else
      {
        controller->Receive(&numForeignBoundaryCenters, 1, p, MY_RETURN_VALUE_MESSAGE);
        controller->Send(&numMyBoundaryCenters, 1, p, MY_RETURN_VALUE_MESSAGE);
      }
      //    std::cout<<"numForeignCenters: "<<procId<<" "<<p<<" "<<numForeignBoundaryCenters<<"\n";

      double foreignBoundaryCentersArray[numForeignBoundaryCenters * numberOfInformation];
      if (procId > p)
      {
        controller->Send(
                         myBoundaryCentersArray, numMyBoundaryCenters * numberOfInformation, p, MY_RETURN_VALUE_MESSAGE);
        controller->Receive(
                            foreignBoundaryCentersArray, numForeignBoundaryCenters * numberOfInformation, p, MY_RETURN_VALUE_MESSAGE);
      }
      else
      {
        controller->Receive(
                            foreignBoundaryCentersArray, numForeignBoundaryCenters * numberOfInformation, p, MY_RETURN_VALUE_MESSAGE);
        controller->Send(
                         myBoundaryCentersArray, numMyBoundaryCenters * numberOfInformation, p, MY_RETURN_VALUE_MESSAGE);
      }
      foreignBoundaryCenters.at(p).resize(numForeignBoundaryCenters);
      for (int fc = 0; fc < numForeignBoundaryCenters; ++fc)
      {
        foreignBoundaryCenters.at(p).at(fc) = std::vector<double>(numberOfInformation);
        for (int d = 0; d < numberOfInformation; ++d)
        {
          foreignBoundaryCenters.at(p).at(fc).at(d) = *(foreignBoundaryCentersArray + numberOfInformation * fc + d);
        }
      }
    }
  }
  return foreignBoundaryCenters;
}
