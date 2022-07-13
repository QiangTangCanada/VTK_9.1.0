/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkSimilarityBalls.cxx

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

#include "vtkSimilarityBalls.h"
#include "vtkMomentsHelper.h"

#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "vtk_eigen.h"
#include VTK_EIGEN(Dense)
#include <vector>

/**
 * standard vtk new operator
 */
vtkStandardNewMacro(vtkSimilarityBalls);

/**
 * constructor setting defaults
 */
vtkSimilarityBalls::vtkSimilarityBalls()
{
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfOutputPorts(3);
  this->KindOfMaxima = 0;
  this->ProduceBalls = 1;
}

/**
 * destructor
 */
vtkSimilarityBalls::~vtkSimilarityBalls() {}

/** standard vtk print function
 * @param os: the way how to print
 * @param indent: how far to the right the text shall appear
 */
void vtkSimilarityBalls::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

/**
 * the agorithm has two input ports
 * port 0 is the similarityData, which is a vtkImageData and the 0th output of vtkMomentInvariants
 * port 1 a finer grid that is used for the drawing of the circles and balls, which is optional
 */
int vtkSimilarityBalls::FillInputPortInformation(int port, vtkInformation* info)
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
 * the agorithm generates 3 outputs, all are vtkImageData
 * the first has the topology of the similarityData
 * it contains the similarity value together with the corresponding radius at the local maxima. All
 * other points are zero. the latter two have the topology of the second input the radius of the
 * local maxima is used to draw a circle around the location with the value of the similarity the
 * radius of the local maxima is used to draw a ball around the location with the value of the
 * similarity
 */
int vtkSimilarityBalls::FillOutputPortInformation(int, vtkInformation* info)
{
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
  return 1;
}

int vtkSimilarityBalls::RequestInformation(vtkInformation*,
                                            vtkInformationVector** inputVector,
                                            vtkInformationVector* outputVector)
{
  vtkInformation* gridInfo = inputVector[1]->GetInformationObject(0);
  // solid ball have shape of grid
  vtkInformation* outInfo1 = outputVector->GetInformationObject(1);
  // hollow sphere have shape of grid
  vtkInformation* outInfo2 = outputVector->GetInformationObject(2);
  if (gridInfo)
  {
    outInfo1->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
                  gridInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
                  6);
    outInfo2->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
                  gridInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
                  6);
  }
  return 1;
}

/**
 * standard vtk pipeline function?
 */
int vtkSimilarityBalls::RequestUpdateExtent(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* similarityInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation* gridInfo = inputVector[1]->GetInformationObject(0);
  // maxima have shape of similarity
  vtkInformation* outInfo0 = outputVector->GetInformationObject(0);
  // solid ball have shape of grid
  vtkInformation* outInfo1 = outputVector->GetInformationObject(1);

  if (gridInfo)
  {
    similarityInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
                        outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT()),
                        6);
    gridInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
                  outInfo1->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT()),
                  6);
  }
  return 1;
}

/**
 * Find out the dimension and the date type of the similarityData dataset.
 * @param similarityData: function of which the moments are computed
 */
void vtkSimilarityBalls::InterpretSimilarityData(vtkImageData* similarityData)
{
  // number of radii
  this->NumberOfRadii = 0;
  for (int i = 0; i < similarityData->GetPointData()->GetNumberOfArrays(); ++i)
  {
    std::string name = similarityData->GetPointData()->GetArrayName(i);
    //    std::cout<<"name="<<name <<" \n";
    try
    {
      std::stod(name);
    }
    catch(...)
    {
      continue;
    }
    this->NumberOfRadii = i+1;
  }
  if (this->NumberOfRadii == 0)
  {
    vtkErrorMacro("The similarityData does not contain any pointdata with radii.");
    return;
  }

  // dimension
  double bounds[6];
  similarityData->GetBounds(bounds);
  if (bounds[5] - bounds[4] < 1e-10)
  {
    this->Dimension = 2;
  }
  else
  {
    this->Dimension = 3;
  }
}

/**
 * Make sure that the user has not entered weird values.
 * @param similarityData: the similarity over the different radii
 * @param gridData: the grid for the balls
 */
void vtkSimilarityBalls::CheckValidity(vtkImageData* similarityData, vtkImageData* gridData)
{
  double similarityBounds[6];
  similarityData->GetBounds(similarityBounds);
  double gridBounds[6];
  gridData->GetBounds(gridBounds);
  for (int i = 0; i < 3; ++i)
  {
    if (similarityBounds[2 * i] < gridBounds[2 * i] ||
      similarityBounds[2 * i + 1] > gridBounds[2 * i + 1])
    {
      cout<<i<<" "<<similarityBounds[2 * i]<<" "<<gridBounds[2 * i]<<" "<<similarityBounds[2 * i + 1]<<" "<<gridBounds[2 * i + 1]<<endl;
      vtkErrorMacro("The grid is smaller than the similarity field, but should contain it.");
      return;
    }
  }
}

/** extraction of the local maxima of the similaity field. this avoids clutter in the visualization.
 * @param similarityData: the output of this algorithm. it has the topology of moments and will have
 * a number of scalar fields euqal to NumberOfRadii. each point contains the similarity of its
 * surrounding (of size radius) to the pattern
 * @param localMaxData: contains the similariy value and the corresponding radius if the similarity
 * field had a local maximum in space plus scale at the given point. It also stored the radius that
 * caused the maximum
 */
void vtkSimilarityBalls::LocalMaxSimilarity(vtkImageData* similarityData,
  vtkImageData* localMaxData)
{
  localMaxData->CopyStructure(similarityData);

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

  //  cout << "the following local maxima have been found:" << endl;

  bool isMax;
  int lessCounter;
  for (int radiusId = 0; radiusId < this->NumberOfRadii; radiusId++)
  {
    std::string fieldName = similarityData->GetPointData()->GetArrayName(radiusId);
    std::string indexD = fieldName.substr(0, fieldName.find("index"));
    std::string radiusD = indexD.substr(0, indexD.find("radius"));
    double radius = std::stod(radiusD);
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
 * This method adds a sphere of (full and hollow) around the center
 * @param center: centroid of the ball
 * @param radius: radius of the ball
 * @param similarity: similarity value (intensity/color) of the ball
 * @param gridData: the grid for the balls
 * @param bounds: global bounds where no balls shall be drawn
 * @param ballsArray: array for the solid ball drawn around the local maxima
 * @param spheresArray: array for the empty sphere drawn around the local maxima
 */
void vtkSimilarityBalls::Draw(double center[3], double radius, double similarity, vtkImageData* gridData, double bounds[6], vtkDoubleArray* ballsArray, vtkDoubleArray* spheresArray)
{
  if (similarity == 0)
  {
    return;
  }
  // determine the indices i and j per dimension of the center in this dataset.
  int dimPtId[3];
  double spacing[3];
  int numberOfPixels[3];
  for (int d = 0; d < this->Dimension; ++d)
  {
    spacing[d] = gridData->GetSpacing()[d] - 1e-10;
    numberOfPixels[d] = radius / spacing[d];
    dimPtId[d] = (center[d] - bounds[2 * d]) / spacing[d];
  }
  int gridDims[3];
  gridData->GetDimensions(gridDims);

//  if (center[0] >= 8-1e-10 && center[0] <=9 && center[1] >= 8-1e-10 && center[1] <= 9 && bounds[0] > 8 && bounds[2] == 0)
//  {
//    cout<<center[0]<< " "<<center[1]<< " "<<dimPtId[0]<< " "<<dimPtId[1]<< " "<<bounds[0]<< " "<<bounds[2]<< " "<<endl;
//  }

  // draw this part
  for (int i = -numberOfPixels[0]; i <= numberOfPixels[0]; ++i)
  {
    if (dimPtId[0] + i < 0 || dimPtId[0] + i >= gridDims[0])
    {
      continue;
    }
    for (int j = -numberOfPixels[1]; j <= numberOfPixels[1]; ++j)
    {
      if (dimPtId[1] + j < 0 || dimPtId[1] + j >= gridDims[1])
      {
        continue;
      }
      if (this->Dimension == 2)
      {
        if (pow(i * spacing[0], 2) + pow(j * spacing[1], 2) < pow(radius, 2))
        {
          //                cout<<i<< " "<<j<< " "<<dimPtId[0] + i<< " "<<dimPtId[1] + j<< " "<<dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0]<< " "<<gridData->GetNumberOfPoints()<<endl;
          if (ballsArray->GetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0]) < similarity)
          {
            ballsArray->SetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0], similarity);
          }
          if (ballsArray->GetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0]) < similarity)
          {
            ballsArray->SetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0], similarity);
          }
          if (pow((std::abs(i) + 1) * spacing[0], 2) +
              pow((std::abs(j) + 1) * spacing[1], 2) >
              pow(radius, 2) &&
              spheresArray->GetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0]) < similarity)
          {
            spheresArray->SetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0], similarity);
          }
        }
      }
      else
      {
        for (int k = -numberOfPixels[2]; k <= numberOfPixels[2]; ++k)
        {
          if (dimPtId[2] + k < 0 || dimPtId[2] + k >= gridDims[2])
          {
            continue;
          }
          if (pow(i * spacing[0], 2) + pow(j * spacing[1], 2) + pow(k * spacing[2], 2) < pow(radius, 2))
          {
            if (ballsArray->GetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0] + (dimPtId[2] + k) * gridData->GetDimensions()[0] * gridData->GetDimensions()[1]) <
                similarity)
            {
              ballsArray->SetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0] + (dimPtId[2] + k) * gridData->GetDimensions()[0] * gridData->GetDimensions()[1],
                                    similarity);
            }
            if (pow((std::abs(i) + 1) * gridData->GetSpacing()[0], 2) +
                pow((std::abs(j) + 1) * gridData->GetSpacing()[1], 2) +
                pow((std::abs(k) + 1) * gridData->GetSpacing()[2], 2) >
                pow(radius, 2) &&
                spheresArray->GetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0] + (dimPtId[2] + k) * gridData->GetDimensions()[0] * gridData->GetDimensions()[1]) < similarity)
            {
              spheresArray->SetTuple1(dimPtId[0] + i + (dimPtId[1] + j) * gridData->GetDimensions()[0] + (dimPtId[2] + k) * gridData->GetDimensions()[0] * gridData->GetDimensions()[1],
                                      similarity);
            }
          }
        }
      }
    }
  }
}

/**
 * This method draws a sphere (full and hollow) around the local maxima.
 * @param localMaxData: contains the similarity value and the corresponding radius if the similarity
 * field at a local maximum in space plus scale at the given point.
 * @param gridData: the grid for the balls
 * @param ballsData: a solid ball drawn around the local maxima
 * @param spheresData: an empty sphere drawn around the local maxima
 */
void vtkSimilarityBalls::Balls(vtkImageData* localMaxData,
  vtkImageData* gridData,
  vtkImageData* ballsData,
  vtkImageData* spheresData)
{
  ballsData->CopyStructure(gridData);
  vtkNew<vtkDoubleArray> ballsArray;
  ballsArray->SetName("balls");
  ballsArray->SetNumberOfTuples(gridData->GetNumberOfPoints());
  for (int i = 0; i < gridData->GetNumberOfPoints(); ++i)
  {
    ballsArray->SetTuple1(i, 0.0);
  }
  //    localMaxValue->Fill(0.0);

  spheresData->CopyStructure(gridData);
  vtkNew<vtkDoubleArray> spheresArray;
  spheresArray->SetName("balls");
  spheresArray->SetNumberOfTuples(gridData->GetNumberOfPoints());
  for (int i = 0; i < gridData->GetNumberOfPoints(); ++i)
  {
    spheresArray->SetTuple1(i, 0.0);
  }
  //    localMaxValue->Fill(0.0);

  double bounds[6];
  gridData->GetBounds(bounds);

  for (int maxId = 0; maxId < localMaxData->GetNumberOfPoints(); maxId++)
  {
    if (localMaxData->GetPointData()->GetArray("localMaxValue")->GetTuple1(maxId) > 0)
    {
      double radius = localMaxData->GetPointData()->GetArray("localMaxRadius")->GetTuple1(maxId);
      double similarity = localMaxData->GetPointData()->GetArray("localMaxValue")->GetTuple1(maxId);
      int ptId = gridData->FindPoint(localMaxData->GetPoint(maxId));
      double center[3];
      gridData->GetPoint(ptId, center);
      if(!vtkMomentsHelper::IsCloseToBoundary(center, radius, bounds, this->Dimension))
      {
        this->Draw(center, radius, similarity, gridData, bounds, ballsArray, spheresArray);
      }
    }
  }
  ballsData->GetPointData()->AddArray(ballsArray);
  spheresData->GetPointData()->AddArray(spheresArray);

  //    if( !isEdge(ptId, similarityData) )
  //    {
  //        for (int i = -1; i <= 1; i++)
  //        {
  //            if (radiusId + i > 0
  //                && radiusId + i < similarityData->GetPointData()->GetNumberOfArrays())
  //            {
  //                for (int j = -1; j <= 1; j++)
  //                {
  //                    for (int k = -1; k <= 1; k++)
  //                    {
  //                        if (!(i == 0 && j == 0 && k == 0)
  //                            &&
  //                            similarityData->GetPointData()->GetArray(radiusId)->GetTuple1(ptId)
  //                            <
  //                            similarityData->GetPointData()->GetArray(radiusId+i)->GetTuple1(ptId+j+k*similarityData->GetDimensions()[0]))
  //                        {
  //                            isMax = false;
  //                        }
  //                    }
  //                }
  //            }
  //        }
  //    }
}

/** main executive of the program, reads the input, calles the functions, and produces the utput.
 * @param request: ?
 * @param inputVector: the input information
 * @param outputVector: the output information
 */
int vtkSimilarityBalls::RequestData(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
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
    this->InterpretSimilarityData(similarityData);
    vtkImageData* localMaxData =
    vtkImageData::SafeDownCast(outInfo0->Get(vtkDataObject::DATA_OBJECT()));
    LocalMaxSimilarity(similarityData, localMaxData);

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
