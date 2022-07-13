/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkPSimilarityBalls.cxx

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
#include "vtkPSimilarityBalls.h"

#include "vtkDoubleArray.h"
#include "vtkImageData.h"

#include "vtkMultiProcessController.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkMPICommunicator.h"
#include "vtkImageTranslateExtent.h"
#include "vtkTrivialProducer.h"
#include "vtkUnsignedCharArray.h"

#include "vtkMomentsHelper.h"
#include "vtkPMomentsHelper.h"

/**
 * standard vtk new operator
 */
vtkStandardNewMacro(vtkPSimilarityBalls);
vtkSetObjectImplementationMacro(vtkPSimilarityBalls, Controller, vtkMultiProcessController);

/**
 * constructor setting defaults
 */
vtkPSimilarityBalls::vtkPSimilarityBalls()
{
  this->Controller = NULL;
  this->SetController(vtkMultiProcessController::GetGlobalController());
  this->MPI_Communicator = NULL;
  vtkMPICommunicator *comm = vtkMPICommunicator::SafeDownCast(this->Controller->GetCommunicator());
  this->MPI_Communicator = comm->GetMPIComm()->GetHandle();
}

/**
 * destructor
 */
vtkPSimilarityBalls::~vtkPSimilarityBalls()
{
  this->SetController(nullptr);
}

/**
 * Make sure that the user has not entered weird values.
 * @param similarityData: the similarity over the different radii
 * @param gridData: the grid for the balls
 */
void vtkPSimilarityBalls::CheckValidity(vtkImageData* similarityData, vtkImageData* gridData)
{
  double similarityBounds[6];
  similarityData->GetBounds(similarityBounds);
  double gridBounds[6];
  gridData->GetBounds(gridBounds);
  if (similarityData->HasAnyGhostCells())
  {
    for (int i = 0; i < 3; ++i)
    {
      similarityBounds[2 * i] = similarityBounds[2 * i] + similarityData->GetSpacing()[i];
      similarityBounds[2 * i + 1] = similarityBounds[2 * i + 1] - similarityData->GetSpacing()[i];
    }
  }
  if (gridData->HasAnyGhostCells())
  {
    for (int i = 0; i < 3; ++i)
    {
      gridBounds[2 * i] = gridBounds[2 * i] + gridData->GetSpacing()[i];
      gridBounds[2 * i + 1] = gridBounds[2 * i + 1] - gridData->GetSpacing()[i];
    }
  }
  for (int i = 0; i < 3; ++i)
  {
    if (similarityBounds[2 * i] < gridBounds[2 * i] - 1e-10 ||
        similarityBounds[2 * i + 1] > gridBounds[2 * i + 1] + 1e-10)
    {
      cout<<i<<" "<<similarityBounds[2 * i]<<" "<<gridBounds[2 * i]<<" "<<similarityBounds[2 * i + 1]<<" "<<gridBounds[2 * i + 1]<<endl;
      vtkErrorMacro("The grid is smaller than the similarity field, but should contain it. If there are more than one layer of ghost cells, this error might be thrown for no reason.");
      return;
    }
  }
}

int vtkPSimilarityBalls::RequestUpdateExtent(
                                            vtkInformation* req, vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  this->Superclass::RequestUpdateExtent(req, inputVector, outputVector);
  // needs ghost cells to get local maxima
  vtkInformation* momentsInfo = inputVector[0]->GetInformationObject(0);
  // does not need ghost cells
  vtkInformation* gridInfo = inputVector[1]->GetInformationObject(0);
  // inherits structure from moments
  vtkInformation* outInfo0 = outputVector->GetInformationObject(0);

  int ghostLevels = 1;
  momentsInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS(), ghostLevels);
  outInfo0->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS(), ghostLevels);

  if (gridInfo)
  {
    gridInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS(), ghostLevels);
  }

  return 1;
}

/** extraction of the local maxima of the similaity field. this avoids clutter in the visualization.
 * @param similarityData: the output of this algorithm. it has the topology of moments and will have
 * a number of scalar fields euqal to NumberOfRadii. each point contains the similarity of its
 * surrounding (of size radius) to the pattern
 * @param localMaxData: contains the similariy value and the corresponding radius if the similarity
 * field had a local maximum in space plus scale at the given point. It also stored the radius that
 * caused the maximum
 */
void vtkPSimilarityBalls::LocalMaxSimilarity(vtkImageData* similarityData,
                                            vtkImageData* localMaxData)
{
  localMaxData->CopyStructure(similarityData);
  localMaxData->GetPointData()->AddArray(similarityData->GetPointGhostArray());
  localMaxData->GetCellData()->AddArray(similarityData->GetCellGhostArray());

  vtkNew<vtkDoubleArray> localMaxValue;
  localMaxValue->SetName("localMaxValue");
  localMaxValue->SetNumberOfTuples(localMaxData->GetNumberOfPoints());
  for (int i = 0; i < localMaxData->GetNumberOfPoints(); ++i)
  {
    localMaxValue->SetTuple1(i, 0.0);
  }
  //    localMaxValue->Fill(0.0);
  localMaxData->GetPointData()->AddArray(localMaxValue);

  vtkNew<vtkDoubleArray> localMaxRadius;
  localMaxRadius->SetName("localMaxRadius");
  localMaxRadius->SetNumberOfTuples(localMaxData->GetNumberOfPoints());
  for (int i = 0; i < localMaxData->GetNumberOfPoints(); ++i)
  {
    localMaxRadius->SetTuple1(i, 0.0);
  }
  //    localMaxRadius->Fill(0.0);
  localMaxData->GetPointData()->AddArray(localMaxRadius);

//  // TODO repair the maximum thing and remove that
//  for (int ptId = 0; ptId < similarityData->GetNumberOfPoints(); ptId++)
//  {
//    localMaxData->GetPointData()
//    ->GetArray("localMaxValue")
//    ->SetTuple1(ptId, similarityData->GetPointData()->GetArray(0)->GetTuple1(ptId));
//    localMaxData->GetPointData()->GetArray("localMaxRadius")->SetTuple1(ptId, std::stod(similarityData->GetPointData()->GetArrayName(0)));
//  }
//  return;



  //  cout << "the following local maxima have been found:" << endl;

  bool isMax;
  int lessCounter;
  for (int radiusId = 0; radiusId < this->NumberOfRadii; radiusId++)
  {
//    std::string fieldName = similarityData->GetPointData()->GetArrayName(radiusId);
//    std::string indexD = fieldName.substr(0, fieldName.find("index"));
//    std::string radiusD = indexD.substr(0, indexD.find("radius"));
    double radius = std::stod(similarityData->GetPointData()->GetArrayName(radiusId));
    for (int ptId = 0; ptId < similarityData->GetNumberOfPoints(); ptId++)
    {
      // if the point is not on an edge, compare it to its neighbors
      if (vtkMomentsHelper::isEdge(this->Dimension, ptId, similarityData))
      {
        isMax = false;
      }
      else
      {
        isMax = true;
        lessCounter = 0;
        for (int r = -1; r <= 1; r++)
        {
          if (radiusId + r > 0 &&
              radiusId + r < similarityData->GetPointData()->GetNumberOfArrays())
          {
            if (!(r == 0) &&
                similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(ptId) <
                similarityData->GetPointData()->GetArray(radiusId + r)->GetTuple1(ptId))
            {
              isMax = false;
              break;
            }
          }
        }
        if (this->KindOfMaxima < 2)
        {
          for (int i = -1; i <= 1; i++)
          {
            for (int j = -1; j <= 1; j++)
            {
              if (this->Dimension == 2)
              {
                if (!(i == 0 && j == 0) &&
                    similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(ptId) <
                    similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(
                                                                                  ptId + i + j * similarityData->GetDimensions()[0]))
                {
                  lessCounter++;
                }
              }
              else
              {
                for (int k = -1; k <= 1; k++)
                {
                  if (!(i == 0 && j == 0 && k == 0) &&
                      similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(ptId) <
                      similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(ptId + i +
                                                                                    j * similarityData->GetDimensions()[0] +
                                                                                    k * similarityData->GetDimensions()[0] *
                                                                                    similarityData->GetDimensions()[1]))
                  {
                    lessCounter++;
                  }
                }
              }
            }
          }
          if ((KindOfMaxima == 0 && lessCounter > 0) || (KindOfMaxima == 1 && lessCounter > 2))
          {
            isMax = false;
          }
        }
      }

      if (isMax &&
          similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(ptId) >=
          localMaxData->GetPointData()->GetArray("localMaxValue")->GetTuple1(ptId))
      {
        localMaxData->GetPointData()
        ->GetArray("localMaxValue")
        ->SetTuple1(ptId, similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(ptId));
        localMaxData->GetPointData()->GetArray("localMaxRadius")->SetTuple1(ptId, radius);
        //      cout << "similarity=" <<
        //      localMaxData->GetPointData()->GetArray("localMaxValue")->GetTuple1(ptId) << "
        //      radius=" << radius << " location=" << localMaxData->GetPoint(ptId)[0] << " " <<
        //      localMaxData->GetPoint(ptId)[1] << " " << localMaxData->GetPoint(ptId)[2] << endl;
      }
    }
  }
}

/**
 * This method draws a sphere (full and hollow) around the local maxima.
 * This is where all the communication with the other procs happens
 * 1. it computes the (partial) balls for all points on this grid
 * 2. it looks where points close to the boundary fall in the bounds of other procs and sends the
 * locations over as partly negative dimension-wise indizes of imageData
 * 3. each proc draws the parts of the balls in its domain
 * @param localMaxData: contains the similarity value and the corresponding radius if the similarity
 * field at a local maximum in space plus scale at the given point.
 * @param gridData: the grid for the balls
 * @param ballsData: a solid ball drawn around the local maxima
 * @param spheresData: an empty sphere drawn around the local maxima
 */
void vtkPSimilarityBalls::Balls(vtkImageData* localMaxData,
                               vtkImageData* gridData,
                               vtkImageData* ballsData,
                               vtkImageData* spheresData)
{
  ballsData->CopyStructure(gridData);
  if (gridData->HasAnyGhostCells())
  {
    ballsData->GetPointData()->AddArray(gridData->GetPointGhostArray());
    ballsData->GetCellData()->AddArray(gridData->GetCellGhostArray());
  }

  vtkNew<vtkDoubleArray> ballsArray;
  ballsArray->SetName("balls");
  ballsArray->SetNumberOfTuples(gridData->GetNumberOfPoints());
  for (int i = 0; i < gridData->GetNumberOfPoints(); ++i)
  {
    ballsArray->SetTuple1(i, 0.0);
  }
  //    localMaxValue->Fill(0.0);

  spheresData->CopyStructure(gridData);
  if (gridData->HasAnyGhostCells())
  {
    spheresData->GetPointData()->AddArray(gridData->GetPointGhostArray());
    spheresData->GetCellData()->AddArray(gridData->GetCellGhostArray());
  }
  vtkNew<vtkDoubleArray> spheresArray;
  spheresArray->SetName("balls");
  spheresArray->SetNumberOfTuples(gridData->GetNumberOfPoints());
  for (int i = 0; i < gridData->GetNumberOfPoints(); ++i)
  {
    spheresArray->SetTuple1(i, 0.0);
  }
  //    localMaxValue->Fill(0.0);

  int gridDims[3];
  gridData->GetDimensions(gridDims);
  double localBounds[6];
  gridData->GetBounds(localBounds);
  // int procId = this->Controller->GetLocalProcessId();
  int numProcs = this->Controller->GetNumberOfProcesses();
  // get all bounds in array procId * 6 + boundIndex. last row contains the global bounds
  double allBounds[6 * numProcs + 6];
  vtkPMomentsHelper::GetAllBoundsWithoutGhostCells(allBounds, localMaxData, this->Controller);
  double globalBounds[6];
  for (int d = 0; d < 6; ++d)
  {
    globalBounds[d] = allBounds[6 * numProcs + d];
  }
  // draw balls for the local centers
  for (int maxId = 0; maxId < localMaxData->GetNumberOfPoints(); maxId++)
  {
    if (localMaxData->GetPointData()->GetArray("localMaxValue")->GetTuple1(maxId) > 0)
    {
      double radius = localMaxData->GetPointData()->GetArray("localMaxRadius")->GetTuple1(maxId);
      double similarity = localMaxData->GetPointData()->GetArray("localMaxValue")->GetTuple1(maxId);
      int ptId = gridData->FindPoint(localMaxData->GetPoint(maxId));
      double center[3];
      gridData->GetPoint(ptId, center);
      if(!vtkMomentsHelper::IsCloseToBoundary(center, radius, globalBounds, this->Dimension))
      {
        this->Superclass::Draw(center, radius, similarity, gridData, localBounds, ballsArray, spheresArray);
      }
    }
  }

  // TODO it might speed things up to only send the actual maxima
  // this vector contains a vector for each proc. with the centers + radii + similarity that are close to this proc's boundary
  std::vector<std::vector<std::vector<double> > > myBoundaryInformation = vtkPMomentsHelper::GetBoundaryInformation(allBounds, localMaxData, this->Dimension, localMaxData->GetPointData()->GetArray("localMaxRadius")->GetRange()[1], this->Controller);
//  if (procId == 0)
//  {
//    cout<<"my ";
//    for (int fc = 0; fc < myBoundaryInformation.at(1).size(); ++fc)
//    {
//      cout<<" "<<myBoundaryInformation.at(1).at(fc)[0]<<" "<<myBoundaryInformation.at(1).at(fc)[1]<<" "<<myBoundaryInformation.at(1).at(fc)[2]<<" "<<myBoundaryInformation.at(1).at(fc)[3]<<" "<<myBoundaryInformation.at(1).at(fc)[4]<<" / ";
//    }
//    cout<<endl;
//  }

  // this vector contains a vector for each proc. with the centers + radii + similarity that are close to that proc's boundary
  std::vector<std::vector< std::vector<double> > > foreignBoundaryInformaion = vtkPMomentsHelper::ExchangeBoundaryInformation(5, myBoundaryInformation, this->Dimension, this->Controller);
//  if (procId == 1)
//  {
//    cout<<"foreign ";
//    for (int fc = 0; fc < foreignBoundaryInformaion.at(0).size(); ++fc)
//    {
//      cout<<" "<<foreignBoundaryInformaion.at(0).at(fc)[0]<<" "<<foreignBoundaryInformaion.at(0).at(fc)[1]<<" / ";
//    }
//    cout<<endl;
//  }
  // draw balls for the foreign centers
  for (int p = 0; p < numProcs; ++p)
  {
    //    cout<<"myBoundaryInformation: "<<procId<<" "<<p<<" "<<myBoundaryInformation.at(p).size()<<endl;
    //    cout<<"foreignBoundaryInformaion: "<<procId<<" "<<p<<" "<<foreignBoundaryInformaion.at(p).size()<<endl;

    for (std::size_t fc = 0; fc < foreignBoundaryInformaion.at(p).size(); ++fc)
    {
      double center[3];
      for (int d = 0; d < 3; ++d)
      {
        center[d] = foreignBoundaryInformaion.at(p).at(fc).at(d);
      }
      double radius = foreignBoundaryInformaion.at(p).at(fc).at(3);
      double similarity = foreignBoundaryInformaion.at(p).at(fc).at(4);
//      if (procId == 1 && p == 0)
//      {
//        cout<<globalBounds[0]<<" "<<globalBounds[1]<<" "<<globalBounds[2]<<" "<<globalBounds[3]<<" "<<radius<<" "<<center[0]<<" "<<center[1]<<" "<<vtkMomentsHelper::IsCloseToBoundary(center, radius, globalBounds, this->Dimension)<<" / ";
//      }
      if(!vtkMomentsHelper::IsCloseToBoundary(center, radius, globalBounds, this->Dimension))
      {

        this->Superclass::Draw(center, radius, similarity, gridData, localBounds, ballsArray, spheresArray);
      }
    }
  }
  ballsData->GetPointData()->AddArray(ballsArray);
  spheresData->GetPointData()->AddArray(spheresArray);
}

/**
 * main executive of the program, reads the input, calles the
 * functions, and produces the utput.
 * @param inputVector: the input information
 * @param outputVector: the output information
 */
int vtkPSimilarityBalls::RequestData(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  if (!this->Controller)
  {
    vtkErrorMacro("There is no controller set.");
  }

  // get the info objects
  vtkInformation* similarityInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation* gridInfo = inputVector[1]->GetInformationObject(0);

  vtkInformation* outInfo0 = outputVector->GetInformationObject(0);
  vtkInformation* outInfo1 = outputVector->GetInformationObject(1);
  vtkInformation* outInfo2 = outputVector->GetInformationObject(2);

  // get the input and output
  vtkImageData* similarityData =
  vtkImageData::SafeDownCast(similarityInfo->Get(vtkDataObject::DATA_OBJECT()));
  if (similarityData)
  {
    this->Superclass::InterpretSimilarityData(similarityData);
    vtkImageData* localMaxData =
    vtkImageData::SafeDownCast(outInfo0->Get(vtkDataObject::DATA_OBJECT()));
//    std::cout<<"localMaxData->HasAnyGhostCells() "<<localMaxData->HasAnyGhostCells()<<"\n";
    this->LocalMaxSimilarity(similarityData, localMaxData);

    if (this->ProduceBalls)
    {
      vtkImageData* gridData;
      if (gridInfo)
      {
        gridData =
        vtkImageData::SafeDownCast(gridInfo->Get(vtkDataObject::DATA_OBJECT()));
      }
      else
      {
        gridData = similarityData;
      }

      vtkImageData* ballsData = vtkImageData::SafeDownCast(outInfo1->Get(vtkDataObject::DATA_OBJECT()));
      vtkImageData* spheresData =
      vtkImageData::SafeDownCast(outInfo2->Get(vtkDataObject::DATA_OBJECT()));

      this->CheckValidity(similarityData, gridData);
      this->Balls(localMaxData, gridData, ballsData, spheresData);
    }
  }
  return 1;
}

void vtkPSimilarityBalls::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Number of processes =  " << this->Controller->GetNumberOfProcesses() << "\n";
  os << indent << "ID of this process =  " << this->Controller->GetLocalProcessId() << "\n";
}
