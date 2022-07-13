/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkMomentInvariants.cxx

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

#include "vtkMomentInvariants.h"
#include "vtkMomentsHelper.h"
#include "vtkMomentsTensor.h"
#include "vtkMomentInvariantData.h"

#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "vtk_eigen.h"
#include VTK_EIGEN(Dense)
#include VTK_EIGEN(Eigenvalues)


#include <list>
#include <vector>


#ifndef M_PI
#define M_PI vtkMath::Pi()
#endif

namespace
{
/** helper function that generates a 2D rotation matrix.
 * @param angle: angle in radiant
 * @return2D rotation matrix around angle
 */
Eigen::MatrixXd getRotMat(double angle)
{
  Eigen::Matrix2d rotMat;
  rotMat(0, 0) = cos(angle);
  rotMat(0, 1) = -sin(angle);
  rotMat(1, 0) = sin(angle);
  rotMat(1, 1) = cos(angle);
  return rotMat;
}

/** this function generates the rotation matrix that rotates the first dominantContraction into the
 * x-axis and if applicable, the second one into the x-y-halfplane with positive z
 * @param dominantContractions: the vectors used for the normalization
 * @return 2D or 3D rotation matrix
 */
Eigen::MatrixXd getRotMat(std::vector<vtkMomentsTensor>& dominantContractions,
  unsigned int dimension)
{
  if (dimension == 2)
  {
    if (dominantContractions.at(0).getVector().norm() < 1e-3)
    {
      return Eigen::Matrix2d::Identity();
    }
    Eigen::Matrix2d rotMat =
      getRotMat(-atan2(dominantContractions.at(0).get(1), dominantContractions.at(0).get(0)));
    if (std::abs((rotMat * dominantContractions.at(0).getVector())[1]) > 1e-3)
    {
      vtkGenericWarningMacro(
        "rotMat=" << rotMat << "gedreht=" << rotMat * dominantContractions.at(0).getVector());
      vtkGenericWarningMacro("rotation not successful.");
    }
    return rotMat;
  }
  else
  {
    if (dominantContractions.at(0).getVector().norm() < 1e-3)
    {
      return Eigen::Matrix3d::Identity();
    }
    if (dominantContractions.at(0).size() == 1)
    {
      return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d axis1 = dominantContractions.at(0).getVector();
    axis1.normalize();
    axis1 = axis1 + Eigen::Vector3d::UnitX();
    axis1.normalize();
    Eigen::Vector3d axis2 = dominantContractions.at(0).getVector();
    axis2.normalize();
    Eigen::Matrix3d rotMat1, rotMat2;
    rotMat1 = Eigen::AngleAxisd(M_PI, axis1);
    if (dominantContractions.size() == 1)
    {
      return rotMat1;
    }
    if (dominantContractions.at(1).getVector().norm() < 1e-3 ||
      (Eigen::Vector3d(dominantContractions.at(0).getVector())
          .cross(Eigen::Vector3d(dominantContractions.at(1).getVector())))
          .norm() < 1e-3)
    {
      return rotMat1;
    }
    rotMat2 = Eigen::AngleAxisd(-atan2((rotMat1 * dominantContractions.at(1).getVector())[2],
                                  (rotMat1 * dominantContractions.at(1).getVector())[1]),
      Eigen::Vector3d::UnitX());
    if (std::abs((rotMat1 * dominantContractions.at(0).getVector())[1]) > 1e-3 ||
      std::abs((rotMat1 * dominantContractions.at(0).getVector())[2]) > 1e-3 ||
      std::abs((rotMat2 * rotMat1 * dominantContractions.at(1).getVector())[2]) > 1e-3 ||
      (rotMat2 * rotMat1 * dominantContractions.at(1).getVector())[1] < -1e-3)
    {
      vtkGenericWarningMacro("Rotation not successful.");
      vtkGenericWarningMacro(
        "rotMat1=" << rotMat1 << "gedreht=" << rotMat1 * dominantContractions.at(0).getVector());
      vtkGenericWarningMacro(
        "rotMat1=" << rotMat1 << "gedreht=" << rotMat1 * dominantContractions.at(1).getVector());
      vtkGenericWarningMacro("rotMat2="
        << rotMat2 << "gedreht=" << rotMat2 * rotMat1 * dominantContractions.at(1).getVector());
      vtkGenericWarningMacro("rotation not successful.");
    }
    return rotMat2 * rotMat1;
  }
}
}

vtkStandardNewMacro(vtkMomentInvariants);

/**
 * constructior setting defaults
 */
vtkMomentInvariants::vtkMomentInvariants()
{
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfOutputPorts(4);  

  this->Dimension = 0;
  this->FieldRank = 0;

  // default settings
  this->Order = 2;
  this->Radii = std::vector<double>(0);
  this->UseOriginalResolution = true;
  this->NumberOfIntegrationSteps = 0;
  this->AngleResolution = 100;
  this->Eps = 1e-2;
  this->NameOfPointData = "no name set by user";
  this->NumberOfFields = 0;
  this->NumberOfDifferentFields = 0;
  this->NumberOfBasisFunctions = 0;
  this->NumberOfDifferentBasisFunctions = 0;
  this->RadiusPattern = std::numeric_limits<double>::max();

  this->CenterPattern[0] = 0.0;
  this->CenterPattern[1] = 0.0;
  this->CenterPattern[2] = 0.0;

  this->IsTranslation = 0;
  this->IsScaling = 0;
  this->IsRotation = 1;
  this->IsReflection = 0;
  this->TranslationFactor = nullptr;

  this->invariantMethod = "Normalization";
  this->luThreshold = 1e-5;
  this->minimumNonZeroOrder = -1;
  this->minimumNonZeroOrderThreshold = 0;
}

vtkMomentInvariants::~vtkMomentInvariants()
{
  delete[] this->TranslationFactor;
}

/**
 * the agorithm has two input ports
 * port 1 is the pattern, which is a vtkImageData of scalar, vector, or matrix type
 * port 0 is the output of computeMoments, which is vtkImageData
 */
int vtkMomentInvariants::FillInputPortInformation(int port, vtkInformation* info)
{
  if (port == PatternPort)
  {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 0);
  }
  if (port == MomentPort)
  {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 0);
  }
  return 1;
}

/**
 * the agorithm generates 4 outputs, all are vtkImageData
 * the first two have the topology of the momentData
 * a field storing the similarity to the pattern for all radii in a scalar field each
 * the normalized moments of the field
 * the latter two have extent 0, they only have 1 point in each field
 * the moments of the pattern
 * the first standard position of the normalized moments of the pattern
 */
int vtkMomentInvariants::FillOutputPortInformation(int, vtkInformation* info)
{
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
  return 1;
}

/** standard vtk print function
 * @param os: the way how to print
 * @param indent: how far to the right the text shall appear
 */
void vtkMomentInvariants::PrintSelf(ostream& os, vtkIndent indent)
{
  os << indent << "vtkMomentInvariants::PrintSelf\n";
  os << indent << "Dimension =  " << this->Dimension << "\n";
  os << indent << "FieldRank =  " << this->FieldRank << "\n";
  os << indent << "Order =  " << this->Order << "\n";
  os << indent << "Radii =  ";
  for (int i = 0; i < static_cast<int>(this->Radii.size()); ++i)
  {
    os << std::scientific << setprecision(10) << this->Radii.at(i) << " ";
  }
  os << "\n";
  os << indent << "NumberOfIntegrationSteps =  " << this->NumberOfIntegrationSteps << "\n";
  os << indent << "UseOriginalResolution? " << this->UseOriginalResolution << "\n";
  os << indent << "NumberOfFields =  " << this->NumberOfFields << "\n";
  os << indent << "NumberOfBasisFunctions =  " << this->NumberOfBasisFunctions << "\n";
  os << indent << "IsTranslation =  " << this->IsTranslation << "\n";
  os << indent << "IsScaling =  " << this->IsScaling << "\n";
  os << indent << "IsRotation =  " << this->IsRotation << "\n";
  os << indent << "IsReflection =  " << this->IsReflection << "\n";
  os << indent << "AngleResolution =  " << this->AngleResolution << "\n";
  os << indent << "Eps =  " << this->Eps << "\n";
  os << indent << "RadiusPattern =  " << this->RadiusPattern << "\n";
  os << indent << "CenterPattern =  " << this->CenterPattern[0] << " " << this->CenterPattern[1]
     << " " << this->CenterPattern[2] << "\n";
  os << indent << "NumberOfOrientationsToCompare =  " << this->MomentsPatternNormal.size() << "\n";
  os << indent << "Invariant Algorithm: " << invariantMethod << endl;
  if (invariantMethod == "Generator") {
    os << indent << "threshold for choosing the minimum nonzero order =  " << minimumNonZeroOrderThreshold << endl;
    os << indent << "minimum nonzero order =  " << minimumNonZeroOrder  << endl;
  }

  cout << "momentsPattern" << endl;
  for (int k = 0; k < static_cast<int>(this->MomentsPattern.size()); ++k)
  {
    this->MomentsPattern.at(k).print();
  }
  if (this->IsTranslation)
  {
    cout << "momentsPatternTNormal" << endl;
    for (int k = 0; k < static_cast<int>(this->MomentsPatternTNormal.size()); ++k)
    {
      this->MomentsPatternTNormal.at(k).print();
    }
  }
  if (this->IsScaling)
  {
    cout << "momentsPatternTSNormal" << endl;
    for (int k = 0; k < static_cast<int>(this->MomentsPatternTSNormal.size()); ++k)
    {
      this->MomentsPatternTSNormal.at(k).print();
    }
  }
  
  printInvariants();
  
  this->Superclass::PrintSelf(os, indent);
}

/**
 * normalization with respect to outer transaltion, i.e. the result will be invariant to adding a
 * constant
 * the translational factor is evaluated with the same stencil as the moments
 * @param moments: the moments at one point stored in a vector of tensors
 * @param radius: the integration radius over which the moments were computed
 * @param isTranslation: if normalization w.tr.t. translation is desired by the user
 * @param stencil: the points in this stencil are used to numerically approximate the integral
 * @return the translationally normalized moments
 */
std::vector<vtkMomentsTensor> vtkMomentInvariants::NormalizeT(
  std::vector<vtkMomentsTensor>& moments,
  double radius,
  bool isTranslation,
  vtkImageData* stencil)
{
  /** normalization of the pattern with respect to translation */
  std::vector<vtkMomentsTensor> momentsNormal = moments;
  if (isTranslation)
  {
    for (int k = 0; k < static_cast<int>(moments.size()); ++k)
    {
      for (int i = 0; i < static_cast<int>(moments.at(k).size()); ++i)
      {
        if (this->Dimension == 2)
        {
          momentsNormal.at(k).set(i,
            moments.at(k).get(i) -
              moments.at(0).get(moments.at(k).getFieldIndices(i)) /
				  vtkMomentsHelper::translationFactor(Dimension, radius, 0, 0, 0, stencil) *
				  vtkMomentsHelper::translationFactor(Dimension, radius,
                  moments.at(k).getOrders(i).at(0),
                  moments.at(k).getOrders(i).at(1),
                  0,
                  stencil));
//          std::cout<<" p="<<moments.at(k).getOrders(i).at(0)<<" q="<<moments.at(k).getOrders(i).at(1)<<" factorAnalytic="<< vtkMomentsHelper::translationFactorAnalytic(radius, 2, moments.at(k).getOrders(i).at(0), moments.at(k).getOrders(i).at(1), 0)<<" factor="<< vtkMomentsHelper::translationFactor(radius, moments.at(k).getOrders(i).at(0), moments.at(k).getOrders(i).at(1), 0, stencil)<<"\n";
        }
        else
        {
          momentsNormal.at(k).set(i,
            moments.at(k).get(i) -
              moments.at(0).get(moments.at(k).getFieldIndices(i)) /
				  vtkMomentsHelper::translationFactor(Dimension, radius, 0, 0, 0, stencil) *
				  vtkMomentsHelper::translationFactor(Dimension, radius,
                  moments.at(k).getOrders(i).at(0),
                  moments.at(k).getOrders(i).at(1),
                  moments.at(k).getOrders(i).at(2),
                  stencil));
        }
      }
    }
  }
  return momentsNormal;
}

/**
 * normalization with respect to outer transaltion, i.e. the result will be invariant to adding a
 * constant
 * the translational factor is evaluated with the same stencil as the moments
 * @param moments: the moments at one point stored in a vector of tensors
 * @param radiusIndex: the index to the integration radius over which the moments were computed
 * @param isTranslation: if normalization w.tr.t. translation is desired by the user
 * @return the translationally normalized moments
 */
std::vector<vtkMomentsTensor> vtkMomentInvariants::NormalizeT(
  std::vector<vtkMomentsTensor>& moments,
  int radiusIndex,
  bool isTranslation)
{
  /** normalization of the moments with respect to translation */
  std::vector<vtkMomentsTensor> momentsNormal = moments;
  if (isTranslation)
  {
    for (int k = 0; k < static_cast<int>(moments.size()); ++k)
    {
      for (int i = 0; i < static_cast<int>(moments.at(k).size()); ++i)
      {
        if (this->Dimension == 2)
        {
          momentsNormal.at(k).set(i,
            moments.at(k).get(i) -
              moments.at(0).get(moments.at(k).getFieldIndices(i)) /
                this->GetTranslationFactor(radiusIndex, 0, 0, 0) *
                this->GetTranslationFactor(radiusIndex,
                  moments.at(k).getOrders(i).at(0),
                  moments.at(k).getOrders(i).at(1),
                  0));
        }
        else
        {
          momentsNormal.at(k).set(i,
            moments.at(k).get(i) -
              moments.at(0).get(moments.at(k).getFieldIndices(i)) /
                this->GetTranslationFactor(radiusIndex, 0, 0, 0) *
                this->GetTranslationFactor(radiusIndex,
                  moments.at(k).getOrders(i).at(0),
                  moments.at(k).getOrders(i).at(1),
                  moments.at(k).getOrders(i).at(2)));
        }
      }
    }
  }
  return momentsNormal;
}

/**
 * normalization with respect to outer transaltion, i.e. the result will be invariant to adding a
 * constant
 * the translational factor is evaluated from the analytic formula
 * @param moments: the moments at one point stored in a vector of tensors
 * @param radius: the integration radius over which the moments were computed
 * @param isTranslation: if normalization w.tr.t. translation is desired by the user
 * @return the translationally normalized moments
 */
std::vector<vtkMomentsTensor> vtkMomentInvariants::NormalizeTAnalytic(
  std::vector<vtkMomentsTensor>& moments,
  double radius,
  bool isTranslation)
{
  /** normalization of the pattern with respect to translation */
  std::vector<vtkMomentsTensor> momentsNormal = moments;
  if (isTranslation)
  {
    for (int k = 0; k < static_cast<int>(moments.size()); ++k)
    {
      for (int i = 0; i < static_cast<int>(moments.at(k).size()); ++i)
      {
        if (this->Dimension == 2)
        {
          momentsNormal.at(k).set(i,
            moments.at(k).get(i) -
              moments.at(0).get(moments.at(k).getFieldIndices(i)) /
                vtkMomentsHelper::translationFactorAnalytic(radius, 2, 0, 0, 0) *
                vtkMomentsHelper::translationFactorAnalytic(radius,
                  2,
                  moments.at(k).getOrders(i).at(0),
                  moments.at(k).getOrders(i).at(1),
                  0));
//          std::cout<<" p="<<moments.at(k).getOrders(i).at(0)<<" q="<<moments.at(k).getOrders(i).at(1)<<" factor="<< vtkMomentsHelper::translationFactorAnalytic(radius,
//          2,
//          moments.at(k).getOrders(i).at(0),
//          moments.at(k).getOrders(i).at(1),
//          0)<<"\n";
        }
        else
        {
          momentsNormal.at(k).set(i,
            moments.at(k).get(i) -
              moments.at(0).get(moments.at(k).getFieldIndices(i)) /
                vtkMomentsHelper::translationFactorAnalytic(radius, 3, 0, 0, 0) *
                vtkMomentsHelper::translationFactorAnalytic(radius,
                  3,
                  moments.at(k).getOrders(i).at(0),
                  moments.at(k).getOrders(i).at(1),
                  moments.at(k).getOrders(i).at(2)));
        }
      }
    }
  }
  return momentsNormal;
}

/**
 * normalization with respect to outer scaling, i.e. the result will be invariant to multiplying a
 * constant
 * @param moments: the moments at one point stored in a vector of tensors
 * @param isScaling: if normalization w.tr.t. scalin is desired by the user
 * @return the scale normalized moments
 */
std::vector<vtkMomentsTensor> vtkMomentInvariants::NormalizeS(
  std::vector<vtkMomentsTensor>& moments,
  bool isScaling,
  double radius)
{
  /** normalization of the pattern with respect to scaling */
  std::vector<vtkMomentsTensor> momentsNormal = moments;
  if (isScaling)
  {
    for (int k = 0; k < static_cast<int>(momentsNormal.size()); ++k)
    {
      for (int i = 0; i < static_cast<int>(moments.at(k).size()); ++i)
      {
        momentsNormal.at(k).set(i, moments.at(k).get(i) / pow(radius, k + this->Dimension));
      }
    }
    double norm = 0;
    for (int k = 0; k < static_cast<int>(momentsNormal.size()); ++k)
    {
      norm += momentsNormal.at(k).norm();
    }
    if (norm > 1e-10)
    {
      for (int k = 0; k < static_cast<int>(momentsNormal.size()); ++k)
      {
        for (int i = 0; i < static_cast<int>(momentsNormal.at(k).size()); ++i)
        {
          momentsNormal.at(k).set(i, momentsNormal.at(k).get(i) / norm);
        }
      }
    }
    norm = 0;
    for (int k = 0; k < static_cast<int>(momentsNormal.size()); ++k)
    {
      norm += momentsNormal.at(k).norm();
    }
    if (std::abs(norm - 1) > 1e-10 && norm > 1e-10)
    {
      vtkErrorMacro("The norm is not one after normalization, but " << norm);
    }
  }
  return momentsNormal;
}

/**
 * this functions reads out the parameters from the pattern and checks if they assume reasonable
 * values
 * @param pattern: the pattern that we will look for
 */
void vtkMomentInvariants::InterpretPattern(vtkImageData* pattern)
{
  // dimension
  double bounds[6];
  pattern->GetBounds(bounds);
  if (bounds[5] - bounds[4] < 1e-10)
  {
    this->Dimension = 2;
  }
  else
  {
    this->Dimension = 3;
  }

  // radius
  for (int d = 0; d < this->Dimension; ++d)
  {
    this->RadiusPattern = std::min(this->RadiusPattern, 0.5 * (bounds[2 * d + 1] - bounds[2 * d]));
  }

  // center
  for (int d = 0; d < 3; ++d)
  {
    this->CenterPattern[d] = 0.5 * (bounds[2 * d + 1] + bounds[2 * d]);
  }

  if (pattern->GetPointData()->GetNumberOfArrays() == 0)
  {
    vtkErrorMacro("The pattern does not contain any pointdata.");
    return;
  }
  if (this->NameOfPointData == "no name set by user")
  {
    this->NameOfPointData = pattern->GetPointData()->GetArrayName(0);
  }
  if (pattern->GetPointData()->GetArray(this->NameOfPointData.c_str()) == NULL)
  {
    vtkErrorMacro("The pattern does not contain an array by the set name in NameOfPointData.");
    return;
  }

  // FieldRank, i.e. scalars, vectors, or matrices
  int numberOfComponents =
    pattern->GetPointData()->GetArray(this->NameOfPointData.c_str())->GetNumberOfComponents();
  if (numberOfComponents == 1)
  {
    this->FieldRank = 0;
  }
  else if (numberOfComponents == 2 || numberOfComponents == 3)
  {
    this->FieldRank = 1;
  }
  else if (numberOfComponents == 4 || numberOfComponents == 6 || numberOfComponents == 9)
  {
    this->FieldRank = 2;
  }
  else
  {
    vtkErrorMacro("pattern pointdata's number of components does not correspond to 2D or 3D "
                  "scalars, vectors, or matrices.");
    return;
  }

  // if UseOriginalResolution is used, a point needs to be in the center of the pattern
  if (this->UseOriginalResolution)
  {
    if (pattern->FindPoint(this->CenterPattern) < 0)
    {
      vtkErrorMacro(
          "Could not find point in the center of the pattern. Make sure your spacings are all greater than 0.");
    }
    for (int d = 0; d < this->Dimension; d++)
    {
      if (pattern->GetDimensions()[d] % 2 != 1)
      {
        vtkErrorMacro(
          "If UseOriginalResolution is used, a point needs to be in the center of "
          "the pattern. Resample the pattern with an odd dimension. pattern->GetDimensions()["
          << d << "] is " << pattern->GetDimensions()[d]);
        return;
      }
      if (this->CenterPattern[d] - this->RadiusPattern < bounds[2 * d] - 1e-10 ||
        this->CenterPattern[d] + this->RadiusPattern > bounds[2 * d + 1] + 1e-10)
      {
        std::cout << "Center: " << this->CenterPattern[0] << ", " << this->CenterPattern[1] << ", " << this->CenterPattern[2] << std::endl;
        std::cout << "Bounds: " << bounds[0] << ", " << bounds[1] << ", " << bounds[2] << ", " << bounds[3] << ", " << bounds[4] << ", " << bounds[5] << std::endl;
        std::cout << "Radius: " << this->RadiusPattern << std::endl;

        vtkErrorMacro("The ball around the center with the radius exceeds the bounds.");
        return;
      }
    }
  }
}

/**
 * this functions reads out the parameters from the momentData and checks if they assume reasonable
 * values and if they match the ones from the pattern
 * @param moments: the moment data
 */
void vtkMomentInvariants::InterpretField(vtkImageData* moments)
{
//  std::cout << "this->Dimension = " << this->Dimension << "\n";

  // actual number of fields
  for (int i = 0; i < moments->GetPointData()->GetNumberOfArrays(); ++i)
  {
    std::string name = moments->GetPointData()->GetArrayName(i);
    // std::cout<<"name="<<name<<" "<<strncmp(name.c_str(), "radius", 6 ) <<" \n";
    if (strncmp(name.c_str(), "radius", 6 ) == 0)
    {
      this->NumberOfDifferentFields = i+1;
    }
  }
//    std::cout << "this->NumberOfDifferentFields = " << this->NumberOfDifferentFields << "\n";

  // fieldrank
  std::string name = moments->GetPointData()->GetArrayName(0);
  int rank = static_cast<int>(name.length() - 1 - name.find("x"));
  if (rank < 0 || rank > 2)
  {
    vtkErrorMacro("the field rank of the moments must be 0, 1, or 2.");
  }
  if ((rank == 0 && this->FieldRank != 0) || (rank == 1 && this->FieldRank != 1) ||
    (rank == 2 && this->FieldRank != 2))
  {
    vtkErrorMacro("field rank of pattern and field must match.");
  }
//  std::cout << "this->FieldRank =" << this->FieldRank << "\n";

  // order
  name = moments->GetPointData()->GetArrayName(this->NumberOfDifferentFields - 1);
  char buffer = name.back();
  if (!(buffer == '2' || buffer == '1' || buffer == 'x'))
  {
    //    std::cout<<"name="<<name<<" buffer="<<buffer <<"end \n";
    vtkErrorMacro("index of the last moment field must end with 1 or 2.");
  }
  if ((buffer == '2' && this->Dimension == 2) || (buffer == '1' && this->Dimension == 3))
  {
    vtkErrorMacro("the dimensions of the pattern and the field must match.");
  }
  name = moments->GetPointData()->GetArrayName(this->NumberOfDifferentFields - 1);
  this->Order = name.length() - 1 - name.find("x") - this->FieldRank;
  // std::cout << "this->Order = " << this->Order << "\n";

  // numberOfBasisFunctions
  this->NumberOfBasisFunctions = 0;
  this->NumberOfDifferentBasisFunctions = 0;

  for (unsigned int k = 0; k < this->Order + 1; ++k)
  {
    this->NumberOfBasisFunctions += pow(this->Dimension, k + this->FieldRank);
    if (this->Dimension == 2)
    {
      this->NumberOfDifferentBasisFunctions += pow(this->Dimension, this->FieldRank) * (k + 1);
    }
    else
    {
      this->NumberOfDifferentBasisFunctions += pow(this->Dimension, this->FieldRank) * 0.5 * (k + 1) * (k + 2);
    }
  }
//  std::cout << "this->NumberOfBasisFunctions=" << this->NumberOfBasisFunctions << "\n";
//  std::cout << "this->NumberOfDifferentBasisFunctions=" << this->NumberOfDifferentBasisFunctions << "\n";

  // radii
  this->Radii = std::vector<double>(0);
  for (int i = 0; i < this->NumberOfDifferentFields; ++i)
  {
    name = moments->GetPointData()->GetArrayName(i);
    char buffer2[40];
    name.copy(buffer2, name.find("index") - name.find("s") - 1, name.find("s") + 1);
    double radius = std::atof(buffer2);
    radius = double(int(radius * 1e6)) / 1e6;
    //    std::cout<<"name="<<name <<"end \n";
    //    std::cout<<"buffer2="<<buffer2 <<"end \n";
    //    std::cout<<std::scientific<< setprecision(10)<<"radius="<<radius <<"\n";
    if (this->Radii.size() == 0 || radius != this->Radii.back())
    {
      this->Radii.push_back(radius);
    }
  }
//    std::cout << "this->Radii.size()=" << this->Radii.size() << "\n";

  // number of basis functions
  this->NumberOfFields = this->NumberOfBasisFunctions * this->Radii.size();
  if (this->NumberOfDifferentFields % this->Radii.size() == 0)
  {
    if (!(unsigned(this->NumberOfDifferentBasisFunctions) == unsigned(this->NumberOfDifferentFields / this->Radii.size())))
    {
//      std::cout << "this->NumberOfFields =" << this->NumberOfFields << "\n";
//      std::cout << "this->Radii.size()=" << this->Radii.size() << "\n";
//      std::cout << "this->NumberOfBasisFunctions =" << this->NumberOfBasisFunctions << "\n";
      vtkErrorMacro("the number of fields in moments has to be the product of the number of radii "
                    "and the numberOfBasisFunctions.");
    }
  }
  else
  {
    vtkErrorMacro("the number of fields in moments has to be a multiple of the number of radii.");
  }
}

/** calculation of the dominant contraction
 * there can be multiple dominant contractions due to EV or 3D
 * dominantContractions.at( i ) contains 1 vector in 2D and 2 in 3D
 * dominantContractions.size() = 1 if no EV, 2 if 1 EV, 4 if 2EV are chosen
 * if no contraction was found dominantContractions.size() = 0
 * if only one contraction was found in 3D dominantContractions.at(i).size() = 1 instead of 2
 * @param momentsPattern: the moments of the pattern
 * @return the dominant contractions, i.e. the biggest vectors that can be used for the normalizaion
 * w.r.t. rotation
 */
std::vector<std::vector<vtkMomentsTensor> > vtkMomentInvariants::CalculateDominantContractions(
  std::vector<vtkMomentsTensor>& momentsPatterLocal)
{
  /** calculation of the products */
  std::list<vtkMomentsTensor> contractions;
  for (int k = 0; k < static_cast<int>(momentsPatterLocal.size()); ++k)
  {
    contractions.push_back(momentsPatterLocal.at(k));
  }
  for (std::list<vtkMomentsTensor>::iterator it = contractions.begin(); it != contractions.end();
       ++it)
  {
    if (it->norm() > 1e-3 && it->getRank() > 0) // prevents infinite powers of zero order
    {
      for (int k = 0; k < static_cast<int>(momentsPatterLocal.size()); ++k)
      {
        // cout<<"it->getRank()="<<it->getRank()<<" k="<<k<<endl;
        if (momentsPatterLocal.at(k).norm() > 1e-3 &&
          it->getRank() <= momentsPatterLocal.at(k).getRank() &&
          it->getRank() + momentsPatterLocal.at(k).getRank() < momentsPatterLocal.back().getRank())
        {
          contractions.push_back(vtkMomentsTensor::tensorProduct(*it, momentsPatterLocal.at(k)));
        }
      }
    }
  }

  /** calculation of the contractions */
  for (std::list<vtkMomentsTensor>::iterator it = contractions.begin(); it != contractions.end();
       ++it)
  {
    if (it->getRank() > 2)
    {
      std::vector<vtkMomentsTensor> contractionsTemp = it->contractAll();
      for (int i = 0; i < static_cast<int>(contractionsTemp.size()); ++i)
      {
        contractions.push_back(contractionsTemp.at(i));
      }
    }
  }

  /** calculation of the eigenvectors */
  for (std::list<vtkMomentsTensor>::iterator it = contractions.begin(); it != contractions.end();
       ++it)
  {
    if (it->getRank() == 2 && it->norm() > this->Eps)
    {
      std::vector<vtkMomentsTensor> contractionsTemp = it->eigenVectors();
      for (int i = 0; i < static_cast<int>(contractionsTemp.size()); ++i)
      {
        contractions.push_back(contractionsTemp.at(i));
      }
    }
  }

  /** calculation of the dominant contraction */
  std::vector<std::vector<vtkMomentsTensor> > dominantContractions(1);
  dominantContractions.at(0).push_back(vtkMomentsTensor(
    this->Dimension, momentsPatterLocal.at(0).getRank(), momentsPatterLocal.at(0).getFieldRank()));
  for (std::list<vtkMomentsTensor>::iterator it = contractions.begin(); it != contractions.end();
       ++it)
  {
    if (it->getRank() == 1)
    {
      // cout << "vectors ";
      // it->print();

      if (dominantContractions.at(0).at(0).norm() < it->norm())
      {
        dominantContractions.at(0).at(0) = *it;
      }
    }
  }
  // zero check
  if (dominantContractions.at(0).at(0).norm() < this->Eps)
  {
    // cout << "all contractions up to this rank are zero.";
    return std::vector<std::vector<vtkMomentsTensor> >(0);
    // vtkGenericWarningMacro( "all contractions up to this rank are zero." );
  }

  if (dominantContractions.at(0).at(0).getContractionInfo().size() % 2 == 1)
  {
    dominantContractions.push_back(dominantContractions.at(0));
    dominantContractions.back().at(0).otherEV();
  }

  if (this->Dimension == 3)
  {
    for (int i = 0; i < static_cast<int>(dominantContractions.size()); ++i)
    {
      dominantContractions.at(i).push_back(
        vtkMomentsTensor(this->Dimension, 1, momentsPatterLocal.at(0).getFieldRank()));
      for (std::list<vtkMomentsTensor>::iterator it = contractions.begin();
           it != contractions.end();
           ++it)
      {
        if (it->getRank() == 1)
        {
          if (Eigen::Vector3d(dominantContractions.at(i).at(0).getVector())
                .cross(Eigen::Vector3d(dominantContractions.at(i).at(1).getVector()))
                .norm() <
            Eigen::Vector3d(Eigen::Vector3d(dominantContractions.at(i).at(0).getVector()))
              .cross(Eigen::Vector3d(it->getVector()))
              .norm())
          {
            dominantContractions.at(i).at(1) = *it;
          }
        }
      }

      // zero or dependence check
      if (Eigen::Vector3d(dominantContractions.at(i).at(0).getVector())
            .cross(Eigen::Vector3d(dominantContractions.at(i).at(1).getVector()))
            .norm() < 1e-2 * dominantContractions.at(i).at(0).getVector().norm())
      {
        // cout << "all contractions up to this rank are linearly dependent." << endl;
        for (int j = 0; j < static_cast<int>(dominantContractions.size()); ++j)
        {
          dominantContractions.at(j).resize(1);
        }
        return dominantContractions;
      }
    }

    int size = static_cast<int>(dominantContractions.size());
    for (int i = 0; i < size; ++i)
    {
      if (dominantContractions.at(i).at(1).getContractionInfo().size() % 2 == 1)
      {
        dominantContractions.push_back(dominantContractions.at(i));
        dominantContractions.back().at(1).otherEV();
      }
    }
  }

  // check if reproduction was successsful
  for (int i = 0; i < static_cast<int>(dominantContractions.size()); ++i)
  {
    std::vector<vtkMomentsTensor> reproducedContractions =
      ReproduceContractions(dominantContractions.at(i), momentsPatterLocal);
    for (int j = 0; j < static_cast<int>(dominantContractions.at(i).size()); ++j)
    {
      if ((dominantContractions.at(i).at(j).getVector() - reproducedContractions.at(j).getVector())
            .norm() > 1e-3)
      {
        reproducedContractions.at(j)
          .rotate(getRotMat(reproducedContractions, this->Dimension))
          .print();
        vtkGenericWarningMacro("reproduction fails.");
      }
    }
  }
  return dominantContractions;
}

/** the dominant contractions are stored as a vector of integers that encode which tensors were
 * multiplied and contracted to form them. This function applies these excat instructions to the
 * moments in the field. That way, these can be normalized in the same way as the pattern was, which
 * is crucial for the comparison.
 * @param dominantContractions: the vectors that can be used for the normalization of this
 * particular pattern, i.e. the ones that are nt zero or linearly dependent
 * @param moments: the moments at one point
 */
std::vector<vtkMomentsTensor> vtkMomentInvariants::ReproduceContractions(
  std::vector<vtkMomentsTensor>& dominantContractions,
  std::vector<vtkMomentsTensor>& moments)
{
  std::vector<vtkMomentsTensor> reproducedTensors(dominantContractions.size());
  for (int i = 0; i < static_cast<int>(dominantContractions.size()); ++i)
  {
    vtkMomentsTensor reproducedTensor =
      moments.at(dominantContractions.at(i).getProductInfo().at(0) - moments.at(0).getFieldRank());

    for (int j = 1; j < static_cast<int>(dominantContractions.at(i).getProductInfo().size()); ++j)
    {
      reproducedTensor = vtkMomentsTensor::tensorProduct(reproducedTensor,
        moments.at(
          dominantContractions.at(i).getProductInfo().at(j) - moments.at(0).getFieldRank()));
    }
    reproducedTensors.at(i) =
      reproducedTensor.contract(dominantContractions.at(i).getContractionInfo());
  }
  return reproducedTensors;
}

/** normalization of the pattern with respect to rotation and reflection
 * @param dominantContractions: the vectors used for the normalization
 * @param isRotation: if the user wants normalization w.r.t rotation
 * @param isReflection: if the user wants normalization w.r.t reflection
 * @param moments: the moments at a given point
 */
std::vector<vtkMomentsTensor> vtkMomentInvariants::NormalizeR(
  std::vector<vtkMomentsTensor>& dominantContractions,
  bool isRotation,
  bool isReflection,
  std::vector<vtkMomentsTensor>& moments)
{
  if (isRotation || isReflection)
  {
    std::vector<vtkMomentsTensor> momentsNormal = moments;
    // determine rotation matrix to move dominantContraction to positive real axis
    std::vector<vtkMomentsTensor> reproducedContraction =
      this->ReproduceContractions(dominantContractions, moments);
    Eigen::MatrixXd rotMat = getRotMat(reproducedContraction, this->Dimension);
    // rotate the tensors
    for (int k = 0; k < static_cast<int>(moments.size()); ++k)
    {
      momentsNormal.at(k) = moments.at(k).rotate(rotMat);
    }

    return momentsNormal;
  }
  else
  {
    return moments;
  }
}

/** if no dominant contractions could be found to be non-zero, the algorithm defaults back to looking
 * for all possible orientations of the given template the parameter AngleResolution determines what
 * "everywhere" means in 2D, we divide phi=[0,...,2Pi] into that many equidistant steps in 3D, we
 * divide phi=[0,...,2Pi] into that many equidistant steps and theta=[0,...,Pi] in half that many
 * steps to determine the rotation axis. Then, we use anther AngleResolution different rotation
 * angles in [0,...,2Pi] to cover all positions
 * @param momentsPatterNormal: this contains all orientations of the moments of the pattern. during
 * the detection later, we will compare the moments of the field to all these version of the pattern
 * @param momentsPatternTranslationalNormal: this contains the moments that are not invariant to
 * orientation yet
 */
void vtkMomentInvariants::LookEverywhere(
  std::vector<std::vector<vtkMomentsTensor> >& momentsPatternNormal,
  std::vector<vtkMomentsTensor>& momentsPatternTranslationalNormal)
{
  std::vector<vtkMomentsTensor> rotatedMoments = momentsPatternTranslationalNormal;
  //    cout << "rotatedMoments" << endl;
  //    for( int k = 0; k < static_cast<int>(rotatedMoments.size()); ++k )
  //    {
  //        rotatedMoments.at( k ).print();
  //    }
  if (this->Dimension == 2)
  {
    for (int i = 0; i < this->AngleResolution; ++i)
    {
      Eigen::Matrix2d rotMat = getRotMat(2 * M_PI / this->AngleResolution * i);
      for (int k = 0; k < static_cast<int>(momentsPatternTranslationalNormal.size()); ++k)
      {
        rotatedMoments.at(k) = momentsPatternTranslationalNormal.at(k).rotate(rotMat);
      }
      momentsPatternNormal.push_back(rotatedMoments);
    }
  }
  else
  {
    Eigen::Matrix3d rotMat;
    for (int i = 0; i < this->AngleResolution; ++i)
    {
      double phi = 2 * M_PI / this->AngleResolution * i;
      for (int j = 0; j < this->AngleResolution / 2; ++j)
      {
        double theta = M_PI / this->AngleResolution * j;
        Eigen::Vector3d axis(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
        for (int l = 0; l < this->AngleResolution; ++l)
        {
          rotMat = Eigen::AngleAxisd(2 * M_PI / this->AngleResolution * l, axis);
          for (int k = 0; k < static_cast<int>(momentsPatternTranslationalNormal.size()); ++k)
          {
            rotatedMoments.at(k) = momentsPatternTranslationalNormal.at(k).rotate(rotMat);
          }
          momentsPatternNormal.push_back(rotatedMoments);
        }
      }
    }
  }
}

/** if only one dominant contraction could be found to be non-zero, but no second one to be linearly
 * independent from the first one, the algorithm, will rotate the first contraction to the x-axis
 * and then look for all possible orientations of the given template around this axis. In principal,
 * it reduces the 3D problem to a 2D problem. the parameter AngleResolution determines what
 * "everywhere" means. We divide phi=[0,...,2Pi] into that many equidistant steps
 * @param dominantContractions: the vectors used for the normalization
 * @param momentsPatternNormal: this contains all orientations of the moments of the pattern. during
 * the detection later, we will compare the moments of the field to all these version of the pattern
 */
void vtkMomentInvariants::LookEverywhere(
  std::vector<std::vector<vtkMomentsTensor> >& dominantContractions,
  std::vector<std::vector<vtkMomentsTensor> >& momentsPatternNormal)
{
  std::vector<vtkMomentsTensor> rotatedMoments = momentsPatternNormal.at(0);
  Eigen::Matrix3d rotMat;
  for (int i = 0; i < static_cast<int>(dominantContractions.size()); ++i)
  {
    Eigen::Vector3d axis = dominantContractions.at(i).at(0).getVector().normalized();
    for (int j = 1; j < this->AngleResolution; ++j)
    {
      rotMat = Eigen::AngleAxisd(2 * M_PI / this->AngleResolution * j, axis.normalized());
      for (int k = 0; k < static_cast<int>(momentsPatternNormal.at(0).size()); ++k)
      {
        rotatedMoments.at(k) = momentsPatternNormal.at(i).at(k).rotate(rotMat);
      }
      momentsPatternNormal.push_back(rotatedMoments);
    }
  }
}

/** calculation of the moments of the pattern and its invariants.
 * we choose, which contractions (dominantContractions) can be used for the normalization of this
 * particular pattern, i.e. the ones that are not zero or linearly dependent. They will later be
 * used for the normalization of the field moments, too.
 * @param dominantContractions: the vectors that can be used for the normalization of this
 * particular pattern, i.e. the ones that are not zero or linearly dependent
 * @param momentsPatternNormal: the moment invariants of the pattern
 * @param pattern: the pattern
 * @param originalMomentsPattern: the moments of the pattern
 * @param normalizedMomentsPattern: the normalized moments of the pattern. It
 * visualizes how the standard position of this particular pattern looks like
 */
void vtkMomentInvariants::HandlePatternNormalization(
  std::vector<std::vector<vtkMomentsTensor> >& dominantContractions,
  vtkImageData* pattern,
  vtkImageData* originalMomentsPattern,
  vtkImageData* normalizedMomentsPattern)
{
  // calculation of the moments of the pattern
  if (this->UseOriginalResolution)
  {
    vtkNew<vtkImageData> stencil;
    stencil->CopyStructure(pattern);
    if (this->Dimension == 2)
    {
    stencil->SetSpacing(stencil->GetSpacing()[0] / this->RadiusPattern,
                        stencil->GetSpacing()[1] / this->RadiusPattern,
                        1);
    }
    else
    {
      stencil->SetSpacing(stencil->GetSpacing()[0] / this->RadiusPattern,
                          stencil->GetSpacing()[1] / this->RadiusPattern,
                          stencil->GetSpacing()[2] / this->RadiusPattern);
    }

    std::vector<int> dimPtId(this->Dimension);
    for (int d = 0; d < this->Dimension; ++d)
    {
      dimPtId[d] = (pattern->GetDimensions()[d]) / 2;
//      std::cout << "pattern->GetDimensions()[d]="<<pattern->GetDimensions()[d] << " dimPtId[d]="<<dimPtId[d]<< "\n";
    }
    this->MomentsPattern = vtkMomentsHelper::allMomentsOrigResImageData(this->Dimension,
      this->Order, this->FieldRank, this->RadiusPattern, &dimPtId[0], pattern, pattern, this->NameOfPointData);

    // fill in the symmetric values that were not computed redundantly
    for (int o = 0; o < this->Order + 1; o++)
    {
      this->MomentsPattern.at(o).fillUsingSymmetry();
    }

    // normalize the moments of the pattern w.r.t translation
    this->MomentsPatternTNormal =
      this->NormalizeT(this->MomentsPattern, 1, this->IsTranslation, stencil);
//     this->NormalizeTAnalytic(this->MomentsPattern, 1, this->IsTranslation);
  }
  else
  {
    vtkNew<vtkImageData> stencil;
    vtkMomentsHelper::BuildStencil(stencil,
      this->RadiusPattern,
      this->NumberOfIntegrationSteps,
      this->Dimension,
      pattern,
      this->NameOfPointData);
    vtkMomentsHelper::CenterStencil(
      this->CenterPattern, pattern, stencil, this->NumberOfIntegrationSteps, this->NameOfPointData);
    this->MomentsPattern = vtkMomentsHelper::allMoments(this->Dimension,
      this->Order,
      this->FieldRank,
      this->RadiusPattern,
      this->CenterPattern,
      stencil,
      this->NameOfPointData);

    // fill in the symmetric values that were not computed redundantly
    for (int o = 0; o < this->Order + 1; o++)
    {
      this->MomentsPattern.at(o).fillUsingSymmetry();
    }

    // normalize the moments of the pattern w.r.t translation
    if (this->Dimension == 2)
    {
      stencil->SetSpacing(stencil->GetSpacing()[0] / this->RadiusPattern,
                          stencil->GetSpacing()[1] / this->RadiusPattern,
                          1);
    }
    else
    {
      stencil->SetSpacing(stencil->GetSpacing()[0] / this->RadiusPattern,
                          stencil->GetSpacing()[1] / this->RadiusPattern,
                          stencil->GetSpacing()[2] / this->RadiusPattern);
    }
    this->MomentsPatternTNormal =
      this->NormalizeT(this->MomentsPattern, 1, this->IsTranslation, stencil);
  }

  this->MomentsPatternTSNormal =
    this->NormalizeS(this->MomentsPatternTNormal, this->IsScaling, 1);

  if (invariantMethod == "Normalization" && (this->IsRotation || this->IsReflection))
  {
    // calculation of the dominant contraction
    dominantContractions = this->CalculateDominantContractions(this->MomentsPatternTSNormal);
    // no dominant contraction could be found?
    if (dominantContractions.size() == 0)
    {
      this->LookEverywhere(this->MomentsPatternNormal, this->MomentsPatternTSNormal);
    }
    else
    {
      for (int i = 0; i < static_cast<int>(dominantContractions.size()); ++i)
      {
        std::vector<vtkMomentsTensor> reproducedContractions =
          this->ReproduceContractions(dominantContractions.at(i), this->MomentsPatternTSNormal);
        for (int j = 0; j < static_cast<int>(dominantContractions.at(i).size()); ++j)
        {
          reproducedContractions.at(j).rotate(getRotMat(reproducedContractions, this->Dimension));
        }
      }

      // normalization of the pattern
      for (int i = 0; i < static_cast<int>(dominantContractions.size()); ++i)
      {
        this->MomentsPatternNormal.push_back(this->NormalizeR(dominantContractions.at(i),
          this->IsRotation,
          this->IsReflection,
          this->MomentsPatternTSNormal));
      }
      // 3D and only one dominant contraction could be found
      if (this->Dimension == 3 && dominantContractions.at(0).size() == 1)
      {
        this->LookEverywhere(dominantContractions, this->MomentsPatternNormal);
      }
      for (int i = 0; i < static_cast<int>(dominantContractions.size()); ++i)
      {
        if (!this->IsRotation)
        {
          for (int k = 0; k < static_cast<int>(this->MomentsPatternNormal.back().size()); ++k)
          {
            for (int j = 0; j < static_cast<int>(this->MomentsPatternNormal.back().at(k).size());
                 ++j)
            {
              this->MomentsPatternNormal.back().at(k).set(j,
                pow(-1,
                  this->MomentsPatternNormal.back().at(k).getIndexSum(j).at(this->Dimension - 1)) *
                  this->MomentsPatternNormal.back().at(k).get(j));
            }
          }
        }
        if (this->IsRotation && this->IsReflection)
        {
          this->MomentsPatternNormal.push_back(this->MomentsPatternNormal.back());
          for (int k = 0; k < static_cast<int>(MomentsPatternNormal.back().size()); ++k)
          {
            for (int j = 0; j < static_cast<int>(MomentsPatternNormal.back().at(k).size()); ++j)
            {
              this->MomentsPatternNormal.back().at(k).set(j,
                pow(-1,
                  this->MomentsPatternNormal.back().at(k).getIndexSum(j).at(this->Dimension - 1)) *
                  this->MomentsPatternNormal.back().at(k).get(j));
            }
          }
        }
      }
    }
  }
  else
  {
    this->MomentsPatternNormal.push_back(this->MomentsPatternTSNormal);
  }

  // store moments as output
  originalMomentsPattern->SetOrigin(this->CenterPattern);
  originalMomentsPattern->SetExtent(0, 0, 0, 0, 0, 0);
  for (int i = 0; i < this->NumberOfBasisFunctions; ++i)
  {
    vtkNew<vtkDoubleArray> array;
    std::string fieldName = "radius" + std::to_string(this->RadiusPattern) + "index" +
      vtkMomentsHelper::getTensorIndicesFromFieldIndexAsString(
        i, this->Dimension, this->Order, this->FieldRank)
        .c_str();
    array->SetName(fieldName.c_str());
    array->SetNumberOfTuples(1);
    originalMomentsPattern->GetPointData()->AddArray(array);
  }
  for (unsigned int k = 0; k < this->Order + 1; ++k)
  {
    for (int i = 0; i < static_cast<int>(this->MomentsPattern.at(k).size()); ++i)
    {
      originalMomentsPattern->GetPointData()
        ->GetArray(vtkMomentsHelper::getFieldNameFromTensorIndices(this->RadiusPattern,
          this->MomentsPattern.at(k).getIndices(i), this->FieldRank).c_str())
        ->SetTuple1(0, this->MomentsPattern.at(k).get(i));
    }
  }

  if (invariantMethod == "Normalization") {
    normalizedMomentsPattern->SetOrigin(this->CenterPattern);
    normalizedMomentsPattern->SetExtent(0, 0, 0, 0, 0, 0);
    for (int i = 0; i < this->NumberOfBasisFunctions; ++i)
      {
	vtkNew<vtkDoubleArray> array;
	std::string fieldName = "radius" + std::to_string(this->RadiusPattern) + "index" +
	  vtkMomentsHelper::getTensorIndicesFromFieldIndexAsString(
								   i, this->Dimension, this->Order, this->FieldRank)
	  .c_str();
	array->SetName(fieldName.c_str());
	array->SetNumberOfTuples(1);
	normalizedMomentsPattern->GetPointData()->AddArray(array);
      }

    for (unsigned int k = 0; k < this->Order + 1; ++k)
      {
	for (int i = 0; i < static_cast<int>(this->MomentsPatternNormal.at(0).at(k).size()); ++i)
	  {
	    normalizedMomentsPattern->GetPointData()
	      ->GetArray(vtkMomentsHelper::getFieldNameFromTensorIndices(this->RadiusPattern,
									 this->MomentsPatternNormal.at(0).at(k).getIndices(i), this->FieldRank).c_str())
	      ->SetTuple1(0, this->MomentsPatternNormal.at(0).at(k).get(i));
	  }
      }
  }
}

/** main part of the pattern detection
 * the moments of the field at each point are normalized and compared to the moments of the pattern
 * @param dominantContractions: the dominant contractions, i.e. vectors for the normalization w.r.t.
 * rotation
 * @param moments: the moments of the field
 * @param normalizedMoments: the moment invariants of the field
 * @param pattern: the pattern
 * @param similarityFields: the output of this algorithm. it has the topology of moments and will
 * have a number of scalar fields euqal to NumberOfRadii. each point contains the similarity of its
 * surrounding (of size radius) to the pattern
 */
void vtkMomentInvariants::HandleFieldNormalization(
  std::vector<std::vector<vtkMomentsTensor> >& dominantContractions,
  vtkImageData* moments,
  vtkImageData* normalizedMoments,
  vtkImageData* pattern,
  vtkImageData* similarityFields)
{
  normalizedMoments->CopyStructure(moments);
  similarityFields->CopyStructure(moments);

  // vector of arrays for the moments. the name is the tensor indices
  for (int r = 0; r < static_cast<int>(this->Radii.size()); ++r)
  {
    vtkNew<vtkDoubleArray> similarity;
    similarity->SetName(std::to_string(this->Radii.at(r)).c_str());
    similarity->SetNumberOfTuples(moments->GetNumberOfPoints());
    similarityFields->GetPointData()->AddArray(similarity);

    for (int i = 0; i < this->NumberOfBasisFunctions; ++i)
    {
      vtkNew<vtkDoubleArray> array;
      std::string fieldName = "radius" + std::to_string(this->Radii.at(r)) + "index" +
        vtkMomentsHelper::getTensorIndicesFromFieldIndexAsString(
          i, this->Dimension, this->Order, this->FieldRank)
          .c_str();
      array->SetName(fieldName.c_str());
      array->SetNumberOfTuples(moments->GetNumberOfPoints());
      normalizedMoments->GetPointData()->AddArray(array);
    }
  }

  // prepare the translational factors. They will be reused for all points
  this->BuildTranslationalFactorArray(pattern);
//  std::cout<<GetTranslationFactor(0,0,0,0)<<" "<<GetTranslationFactor(0,2,0,0)<<" "<<GetTranslationFactor(0,0,2,0)<<"\n";
  // read the tensor vector, normalize it, compute similarity, and store it in the output
  for (int r = 0; r < static_cast<int>(this->Radii.size()); ++r)
  {
    for (int j = 0; j < moments->GetNumberOfPoints(); ++j)
    {
      // read the moment vector
      std::vector<vtkMomentsTensor> tensorVector(this->Order + 1);
      for (unsigned int k = 0; k < this->Order + 1; ++k)
      {
        tensorVector.at(k) =
          vtkMomentsTensor(this->Dimension, k + this->FieldRank, this->FieldRank);
        for (int i = 0; i < static_cast<int>(tensorVector.at(k).size()); ++i)
        {
          tensorVector.at(k).set(i,
            moments->GetPointData()
              ->GetArray(vtkMomentsHelper::getFieldNameFromTensorIndices(this->Radii.at(r),
                tensorVector.at(k).getIndices(i), this->FieldRank).c_str())
              ->GetTuple(j)[0]);
        }
      }
      // normalize the moment vector
      std::vector<vtkMomentsTensor> tensorVectorTNormal =
        this->NormalizeT(tensorVector, r, this->IsTranslation);

      std::vector<vtkMomentsTensor> tensorVectorTSNormal =
        this->NormalizeS(tensorVectorTNormal, this->IsScaling, 1);
      std::vector<vtkMomentsTensor> tensorVectorNormal = tensorVectorTSNormal;
      if (dominantContractions.size() > 0)
      {
        tensorVectorNormal = this->NormalizeR(
          dominantContractions.at(0), this->IsRotation, this->IsReflection, tensorVectorTSNormal);
      }
      // compute similarity to pattern
      double distance = std::numeric_limits<double>::max();
      for (int i = 0; i < static_cast<int>(this->MomentsPatternNormal.size()); ++i)
      {
        double distanceTemp = 0;
        for (unsigned int k = 0; k < this->Order + 1; ++k)
        {
          distanceTemp += vtkMomentsTensor::tensorDistance(
            this->MomentsPatternNormal.at(i).at(k), tensorVectorNormal.at(k));
        }
        distance = std::min(distance, distanceTemp);
      }
      if (distance == 0)
      {
        similarityFields->GetPointData()
        ->GetArray(std::to_string(this->Radii.at(r)).c_str())
        ->SetTuple1(j, 1e10);
      }
      else
      {
        similarityFields->GetPointData()
        ->GetArray(std::to_string(this->Radii.at(r)).c_str())
        ->SetTuple1(j, 1. / distance);
      }
      // store normalized moments in the output
      for (unsigned int k = 0; k < this->Order + 1; ++k)
      {
        for (int i = 0; i < static_cast<int>(tensorVectorNormal.at(k).size()); ++i)
        {
          normalizedMoments->GetPointData()
            ->GetArray(vtkMomentsHelper::getFieldNameFromTensorIndices(this->Radii.at(r),
              tensorVectorNormal.at(k).getIndices(i), this->FieldRank).c_str())
            ->SetTuple1(j, tensorVectorNormal.at(k).get(i));
        }
      }
    }
  }
}

/**
 * Make sure that the user has not entered weird values.
 * @param pattern: the pattern that we will look for
 */
void vtkMomentInvariants::CheckValidity(vtkImageData* pattern)
{
  if (!pattern)
  {
    vtkErrorMacro("A pattern needs to be provided through SetInputData().");
    return;
  }
  if (this->NumberOfIntegrationSteps < 0)
  {
    vtkErrorMacro("The number of integration steps must be >= 0.");
    return;
  }
  if ( this->UseOriginalResolution && this->NumberOfIntegrationSteps > 0)
  {
    vtkErrorMacro("The NumberOfIntegrationSteps are ignored and shuld be 0 because UseOriginalResolution is enabled");
    return;
  }
  if (!this->UseOriginalResolution && this->NumberOfIntegrationSteps == 0)
  {
    vtkErrorMacro("UseOriginalResolution is disabled. Therefore a stencil with NumberOfIntegrationSteps will be produced for the integration, but NumberOfIntegrationSteps is 0.");
    return;
  }
}

/**
 * this computes the translational factors necessary for normalization w.r.t. translation
 * we have radius and then p,q,r
 * @param stencil: the points in this stencil are used to numerically approximate the integral
 */
void vtkMomentInvariants::BuildTranslationalFactorArray(vtkImageData* pattern)
{
  if (this->IsTranslation)
  {
    delete[] this->TranslationFactor;
    this->TranslationFactor = new double[this->Radii.size() *
      int(pow((this->Order + 1) + this->FieldRank, this->Dimension))];
    for (int radiusIndex = 0; radiusIndex < static_cast<int>(this->Radii.size()); ++radiusIndex)
    {
      // prepare stencil
      vtkNew<vtkImageData> stencil;
      if (this->UseOriginalResolution)
      {
        stencil->CopyStructure(pattern);
        if (this->Dimension == 2)
        {
          stencil->SetSpacing(stencil->GetSpacing()[0] / this->RadiusPattern,
            stencil->GetSpacing()[1] / this->RadiusPattern, 1);
        }
        else
        {
          stencil->SetSpacing(stencil->GetSpacing()[0] / this->RadiusPattern,
            stencil->GetSpacing()[1] / this->RadiusPattern,
            stencil->GetSpacing()[2] / this->RadiusPattern);
        }
      }
      else
      {
        vtkMomentsHelper::BuildStencil(stencil,
                                       1,
                                       this->NumberOfIntegrationSteps,
                                       this->Dimension,
                                       pattern,
                                       this->NameOfPointData);
      }
      // compute factor
      for (unsigned int p = 0; p < (this->Order + 1); ++p)
      {
        for (unsigned int q = 0; q < (this->Order + 1) - p; ++q)
        {
          if (this->Dimension == 2)
          {
//            this->SetTranslationFactor(radiusIndex,
//              p,
//              q,
//              0,vtkMomentsHelper::translationFactorAnalytic(this->Radii.at(radiusIndex),2, p, q, 0));
//            std::cout<<" p="<<p<<" q="<<q<<" translationFactorAnalytic="<<vtkMomentsHelper::translationFactorAnalytic(this->Radii.at(radiusIndex),2, p, q, 0)<<" translationFactor="<<vtkMomentsHelper::translationFactor(1, p, q, 0, stencil)<<"\n";
            this->SetTranslationFactor(radiusIndex,
                                       p,
                                       q,
                                       0, vtkMomentsHelper::translationFactor(Dimension, 1, p, q, 0, stencil));
          }
          else
          {
            for (unsigned int r = 0; r < (this->Order + 1) - p - q; ++r)
            {
              this->SetTranslationFactor(radiusIndex,
                p,
                q,
                r,
					 vtkMomentsHelper::translationFactor(Dimension, 1, p, q, r, stencil));
            }
          }
        }
      }
    }
  }
}

/**
 * This function computes homogeneous invariants for a specific order
 * @param generator random generator
 * @param homogeneousInvariant a list of independent homogeneous invariants
 * @param order the maximum order of homogeneous invariants
 */
std::list<vtkMomentsPolynomial> vtkMomentInvariants::computeHomogeneousInvariant(default_random_engine & generator, std::list<vtkMomentsPolynomial> & homogeneousInvariant, unsigned order) const {
  std::forward_list<vtkMomentsPolynomial> polys;
  std::list<vtkMomentsIndex> variables = vtkMomentsIndex::generateVariables(this->Dimension, order, this->FieldRank);
  std::list<polyDerivativePair> pairs;
  list<vtkMomentsPolynomial> invariants;
 

  unsigned maxNumber = vtkMomentInvariants::getInvariantNumberVec(Dimension, FieldRank, order)[1];

  // if (((Dimension == 2) && (order + FieldRank > 0)) || ((Dimension == 3) && (order + FieldRank > 1))) {
    for (unsigned exp = 1; true; exp++) 
      if (((order + this->FieldRank) * exp) % 2 == 0) {
	// cout << "order: " << order << " exp: " << exp << endl;
	vtkMomentsTensorSimple tensor(this->Dimension, order + this->FieldRank, this->FieldRank);
	tensor.tensorPow(exp);
	if (computeInvariant(tensor, invariants, polys, variables, pairs, generator, (int)maxNumber))
	  break;
      }    
  // }
  // else
  //  for (unsigned exp = 1; exp <= 4; exp++) {
  //   if (((order + this->FieldRank) * exp) % 2 == 0) {
  //     cout << "order: " << order << "exp = " << exp << endl;
      
  //     vtkMomentsTensorSimple tensor(this->Dimension, order + this->FieldRank, this->FieldRank);
  //     tensor.tensorPow(exp);
  //     computeInvariant(tensor, invariants, polys, variables, pairs, generator);
  //   }    
  // }
  
  for (auto & pair:pairs) 
    homogeneousInvariant.push_back(pair.first);

  return invariants;
}

/**
 * This function computes the generator invariants based on the moments
 * of the pattern. It looks for the the order at which moments are non-zero, and uses this order to construct mixed order invariants.
 * @param[in] variables a list of moments up to Order
 * @param[in] values values of moments
 * @param[in,out] invariants a list of independent invariants
 * @param[in] homogeneousInvariants homogeneous invariants computed using the generoator algorithm
 * @param[in] mixedInvariants mixed invariants computed using the generoator algorithm
 * @return a vector of moment invariants computed on the pattern moments.
 */  
std::vector<double> vtkMomentInvariants::handlePatternGenerator(list<vtkMomentsIndex> variables,
								indexValueMap & values,
								list<vtkMomentsPolynomial> & invariants,
								const list<list<vtkMomentsPolynomial>> & homogeneousInvariants,
								const list<list<list<vtkMomentsPolynomial>>> & mixedInvariants,
								vtkImageData* normalizedMomentsPattern) 
{
  for (auto & val : values)
    val.second = MomentsPatternTSNormal[val.first.getMomentRank()].get(val.first.getIndices());

  vector<double> avgMoments(Order + 1, 0);
  unsigned i;
  unsigned numMoment = 0;
  
  if (FieldRank == 0) 
    i = 1;
  else 
    i = 0;

  for (auto & var : variables) {
    if (var.getMomentRank() == i) {
      avgMoments[i] += fabs(values.at(var));
      numMoment++;
    }
    else if (numMoment > 0){
      avgMoments[i++] /= numMoment;
      avgMoments[i] += fabs(values.at(var)); 
      numMoment = 1;
    }
  }

  avgMoments[i] /= numMoment;
  
  minimumNonZeroOrderThreshold = accumulate(avgMoments.cbegin(), avgMoments.cend(), 0.0);

  if (FieldRank == 0) 
    minimumNonZeroOrderThreshold /= Order;
  else 
    minimumNonZeroOrderThreshold /= (Order + 1);  

  // cout << "minimumNonZeroOrderThreshold = " << minimumNonZeroOrderThreshold << endl;
  // for (auto & avg : avgMoments)
  //   cout << avg << endl;
  
  unsigned o = 0;
  if (FieldRank == 0)
    o = 1;
  // else if (FieldRank == 1)
  //   o = 1;

  for (; o <= Order; o++)
    if (avgMoments[o] > minimumNonZeroOrderThreshold) {
      minimumNonZeroOrder = o;
      break;
    }

  if (minimumNonZeroOrder == -1) 
    vtkErrorMacro("All moments are zero.");

  if (invariants.empty())
    for (auto & homoInvOrder : homogeneousInvariants)
      for (auto & homoInv : homoInvOrder)
	invariants.push_back(homoInv);

  for (auto & xs : mixedInvariants)
    for (auto & ys : xs)
      for (auto & mixedInv : ys)
	if (mixedInv.getMaximumTensorOrder() == minimumNonZeroOrder ||
	    mixedInv.getMinimumTensorOrder() == minimumNonZeroOrder)
	  invariants.push_back(mixedInv);

  for (unsigned i = 0; i <= Order; i++) {
    list<vtkMomentsPolynomial> polys;
    
    for (auto & inv : invariants)
      if (inv.getMaximumTensorOrder() == i)
	polys.push_back(inv);
    
    if (!polys.empty())
      this->generatorInvariants.push_back(polys);    
  }

  // store normalizedMomentsPattern
  normalizedMomentsPattern->SetOrigin(this->CenterPattern);
  normalizedMomentsPattern->SetExtent(0, 0, 0, 0, 0, 0);

  vector<double> descriptor(invariants.size());

  auto itDesc = descriptor.begin();
  auto itInv = invariants.cbegin();

  for (; itDesc != descriptor.end(); ++itDesc, ++itInv)  {
    vtkNew<vtkDoubleArray> array;
    string fieldName = vtkMomentInvariants::getArrayNameFromInvariant(this->RadiusPattern, *itInv);
    array->SetName(fieldName.c_str());
    array->SetNumberOfTuples(1);
    array->SetTuple1(0,(*itInv).assignValueAndNormalize(values));
    normalizedMomentsPattern->GetPointData()->AddArray(array);
    (*itDesc) = (*itInv).assignValueAndNormalize(values);    
  }
    
  return descriptor;
}

/**
 * This function computes the invariants for the field using the the invariants computed in handlePatternGenerator. 
 * Then it compute the similarity between the value of the invariants of the pattern and the ones of the field.
 * @param[in] invariants a list of independent invariants
 * @param[in] values values of moments
 * @param[in] patternDesc a vector of moment invariants computed on the pattern moments.
 * @param[in] momentData moment data
 * @param[in] pattern pattern data   
 * @param[out] similarityFields the similarities between the pattern and the field
 */
void vtkMomentInvariants::handleFieldGenerator(std::list<vtkMomentsPolynomial> & invariants, indexValueMap & values, const vector<double> & patternDesc, vtkImageData* moments, vtkImageData* pattern, vtkImageData* similarityFields, vtkImageData* normalizedMomentsFields)
{
  normalizedMomentsFields->CopyStructure(moments);
  similarityFields->CopyStructure(moments);
  double minDist = 100000000000;
  int minIdx = -1;

  // prepare the translational factors. They will be reused for all points
  this->BuildTranslationalFactorArray(pattern);

  for (int r = 0; r < static_cast<int>(this->Radii.size()); ++r)
    {
      //initialize normalizedMomentsFields
      for (auto & inv : invariants) {
      	vtkNew<vtkDoubleArray> array;
      	string fieldName = vtkMomentInvariants::getArrayNameFromInvariant(Radii[r],inv);
      	array->SetName(fieldName.c_str());
      	array->SetNumberOfTuples(moments->GetNumberOfPoints());
      	normalizedMomentsFields->GetPointData()->AddArray(array);
      }

      // initialize similarityFields
      vtkNew<vtkDoubleArray> similarity;
      similarity->SetName(std::to_string(this->Radii.at(r)).c_str());
      similarity->SetNumberOfTuples(moments->GetNumberOfPoints());
      similarityFields->GetPointData()->AddArray(similarity);
      
      for (int j = 0; j < moments->GetNumberOfPoints(); ++j)
	{
	  // read the moment vector
	  std::vector<vtkMomentsTensor> tensorVector(this->Order + 1);
	  for (unsigned int k = 0; k < this->Order + 1; ++k)
	    {
	      tensorVector.at(k) =
		vtkMomentsTensor(this->Dimension, k + this->FieldRank, this->FieldRank);
	      for (int i = 0; i < static_cast<int>(tensorVector.at(k).size()); ++i)
		  tensorVector.at(k).set(i,
					 moments->GetPointData()
					 ->GetArray(vtkMomentsHelper::getFieldNameFromTensorIndices(this->Radii.at(r),
												    tensorVector.at(k).getIndices(i), this->FieldRank).c_str())
					 ->GetTuple(j)[0]);
	    }
	  
	  // normalize the moment vector
	  std::vector<vtkMomentsTensor> tensorVectorTNormal =
	    this->NormalizeT(tensorVector, r, this->IsTranslation);

	  std::vector<vtkMomentsTensor> tensorVectorTSNormal =
	    this->NormalizeS(tensorVectorTNormal, this->IsScaling, 1);
	  std::vector<vtkMomentsTensor> tensorVectorNormal = tensorVectorTSNormal;
	  
	  for (auto & val:values) 
	    val.second = tensorVectorNormal[val.first.getMomentRank()].get(val.first.getIndices());

	  //store normalized moments to normalizedMomentsFields
	  vector<double> descriptor(invariants.size());

	  auto itDesc = descriptor.begin();
	  auto itInv = invariants.cbegin();

	  for (; itDesc != descriptor.end(); ++itDesc, ++itInv) {
	    normalizedMomentsFields
	      ->GetPointData()
	      ->GetArray(vtkMomentInvariants::getArrayNameFromInvariant(Radii[r], *itInv).c_str())
	      ->SetTuple1(j, (*itInv).assignValueAndNormalize(values));
	    (*itDesc) = (*itInv).assignValueAndNormalize(values);
	  }

	  // compute similarity 
	  double distance = 0;
	  auto it1 = patternDesc.cbegin();
	  auto it2 = descriptor.cbegin();
	  for (; it1 != patternDesc.cend(); ++it1, ++it2)
	    distance += (*it1 - *it2) * (*it1 - *it2);

	  if (distance < minDist) {
	    minDist = distance;
	    minIdx =  j;
	  }

	  if (distance == 0)
	    similarityFields->GetPointData()
	      ->GetArray(std::to_string(this->Radii.at(r)).c_str())
	      ->SetTuple1(j, 1e10);
	  else
	    similarityFields->GetPointData()
	      ->GetArray(std::to_string(this->Radii.at(r)).c_str())
	      ->SetTuple1(j, 1. / distance);
	} // j
    }  // r
}

/**
 * This function computes the invariants using the algorithm proposed by Langbein. 
 * For the sake of reducing computation time, there are a factor cap(4) and a rank cap (14).
 * This, of course, prevents the algorithm from finding 
 * all possible invariants. This is also a drawback of the algorithm.
 * The function stops when the number of invariants found reaches its theoretical maximum.
 * @return a list of Langbein invariants
 */
std::list<vtkMomentsPolynomial> vtkMomentInvariants::computeLangbeinInvariants()  {
  list<vtkMomentsIndex> variables = createVariables();
  forward_list<vtkMomentsPolynomial> polys;
  list<polyDerivativePair> pairs;
  list<vtkMomentsPolynomial> output;
  default_random_engine generator;
  unsigned maxRank = Order + FieldRank;

  int constant;
  if (Dimension == 2) {
    if (Order + FieldRank == 0)
      constant = 0;
    else
      constant = 1;
  }
  else if (Dimension == 3) {
    if (Order + FieldRank == 0)
      constant = 0;
    else if (Order + FieldRank == 1)
      constant = 2;
    else
      constant = 3;
  }
  else
    vtkErrorMacro("Dimension is not 2 nor 3.");


  unsigned i = 0;
  if (FieldRank == 0) {
    i = 1;
    vtkMomentsTensorSimple tensor = vtkMomentsTensorSimple(Dimension, 0, 0);
    if (computeInvariant(tensor, output, polys, variables, pairs, generator, variables.size() - constant))
      return output;
  }

  for (; true; i++) {
    string str = to_string(i);
    vector<unsigned> rankVec(str.size());

    unsigned firstRank = str.front() - '0';
    if (firstRank > maxRank || firstRank < FieldRank)
      continue;

    for (unsigned j = 0; j < str.size(); j++)
      rankVec[j] = str[j] - '0';

    bool flag = true;

    for (auto it = next(rankVec.cbegin()); it != rankVec.cend(); ++it) 
      if (*it < *prev(it) || *it > maxRank || *it < FieldRank) {
	flag = false;	
	break;
      }
    
    if (flag) {
      vtkMomentsTensorSimple tensor(Dimension, rankVec.front(), FieldRank);
	
      for (auto it = next(rankVec.cbegin()); it != rankVec.cend(); ++it)
	tensor.tensorProduct(vtkMomentsTensorSimple(Dimension, *it, FieldRank));

      if (computeInvariant(tensor, output, polys, variables, pairs, generator, variables.size() - constant)) 
	break;
    } // flag
  } // i  

  return output;
}

/**
 * This function computes the Langbein invariants based on the moments
 * of the pattern. 
 * @param[in] invariants a list of Langbein invariants
 * @param values values of moments
 * @return a vector of moment invariants computed on the pattern moments. 
 */  
std::vector<double> vtkMomentInvariants::handlePatternLangbeinInvariants(const std::list<vtkMomentsPolynomial> & invariants, indexValueMap & values, vtkImageData* normalizedMomentsPattern) {
  for (auto & val:values)
    val.second = MomentsPatternTSNormal[val.first.getMomentRank()].get(val.first.getIndices());

  // store normalizedMomentsPattern
  normalizedMomentsPattern->SetOrigin(this->CenterPattern);
  normalizedMomentsPattern->SetExtent(0, 0, 0, 0, 0, 0);
  vector<double> descriptor(invariants.size());

  auto itDesc = descriptor.begin();
  auto itInv = invariants.cbegin();

  for (; itDesc != descriptor.end(); ++itDesc, ++itInv) {
    vtkNew<vtkDoubleArray> array;
    string fieldName = vtkMomentInvariants::getArrayNameFromInvariant(this->RadiusPattern, *itInv);
    array->SetName(fieldName.c_str());
    array->SetNumberOfTuples(1);
    array->SetTuple1(0,(*itInv).assignValueAndNormalize(values));
    normalizedMomentsPattern->GetPointData()->AddArray(array);
    (*itDesc) = (*itInv).assignValueAndNormalize(values);
  }
  
  return descriptor;
}

/**
 * This helper function inserts a mixed invariants to the invariant list. 
 * If the mixed one is independent to the rest, it will be inserted to the list.
 * Otherwise, it will be discarded.
 * @param invariants a list of invariants
 * @param variables a list of moments up to Order
 * @param polys a set of polynomials that corresponds to the invariants
 * @param pairs a list of moment invariants and corresponding derivatives w.r.t. moments
 * @param generator a random generator
 * @param o1 order of the first moment
 * @param exp1 exponent of the first moment factor
 * @param o2 order of the second moment
 * @param exp2 exponent of the second moment factor
 * @param maxNum theoretical maximum number of invariants
 * @return a boolean indicating whether or not the number of invariant has reached its theoretical maximum
 */
bool vtkMomentInvariants::computeMixedInvariantOrderPairHelper(list<vtkMomentsPolynomial> & invariants, const list<vtkMomentsIndex> & variables, forward_list<vtkMomentsPolynomial> & polys, list<polyDerivativePair> & pairs, default_random_engine & generator, unsigned o1, unsigned exp1, unsigned o2, unsigned exp2, unsigned maxNum) const {
  if (((o1 + FieldRank) * exp1 + (o2 + FieldRank) * exp2) % 2 == 0) {
    vtkMomentsTensorSimple product(Dimension, o1 + FieldRank, FieldRank), tensor2(Dimension, o2 + FieldRank, FieldRank);
    
    product.tensorPow(exp1);
    tensor2.tensorPow(exp2);
    product.tensorProduct(tensor2);
    
    return computeInvariant(product, invariants, polys, variables, pairs, generator, (int)maxNum);    
  }
  else
    return false;
}

/** 
 * This funcion searches for independent mixed invariants for a pair of order.
 * @param homoInv a list of homogeneous invariants
 * @param variables a list of moments
 * @param o1 order of the first moment
 * @param o2 order of the second moment
 * @return a list of independent mixed invariants
 */
list<vtkMomentsPolynomial> vtkMomentInvariants::computeMixedInvariantsOrderPair(const list<vtkMomentsPolynomial> & homoInv , const list<vtkMomentsIndex> & variables, unsigned o1, unsigned o2) const {
  
  unsigned exp1, exp2;
  
  if ((o1 + FieldRank) % 2 == 0)
    exp1 = 1;
  else
    exp1 = 2;

  if ((o2 + FieldRank) % 2 == 0)
    exp2 = 1;
  else
    exp2 = 2;

  unsigned rank1 = (o1 + FieldRank) * exp1, rank2 = (o2 + FieldRank) * exp2;

  vtkMomentsTensorSimple product(Dimension, o1 + FieldRank, FieldRank), tensor2(Dimension, o2 + FieldRank, FieldRank);
 
  product.tensorPow(exp1);
  tensor2.tensorPow(exp2);
  product.tensorProduct(tensor2);

  if (rank1 > rank2) {
    unsigned rem = (rank1 - rank2) / 2;
    for (unsigned i = 0; i < rem; i++)
      product.contract(0,1);
	  
    for (unsigned i = rank2; i > 0; i--) 
      product.contract(0,i);	  
  }
  else {
    unsigned rem = (rank2 - rank1) / 2;
	  
    for (unsigned i = rank1; i > 0; i--) 
      product.contract(0,i);

    for (unsigned i = 0; i < rem; i++)
      product.contract(0,1);
  }

  vtkMomentsPolynomial mixedPoly(product);

  unsigned o = max(o1,o2);

  if (o < 1)
    vtkErrorMacro("vtkMomentInvariants::computeMixedInvariantsOrderPair: the maximum order < 1.");

  vector<int> num = vtkMomentInvariants::getInvariantNumberVec((int)Dimension, (int)FieldRank, (int)o);
  vector<int> num0 = vtkMomentInvariants::getInvariantNumberVec((int)Dimension, (int)FieldRank, (int)(o - 1));
  unsigned numExpectedMixed = (unsigned)num[4] - (unsigned)num0[4];
  if (min(o1,o2) + FieldRank == 1)
    numExpectedMixed = 2;

  default_random_engine generator;
  list<polyDerivativePair> pairs;
  forward_list<vtkMomentsPolynomial> polys;
  list<vtkMomentsPolynomial> invariants;
  
  for (auto & inv : homoInv)
    pairs.push_back(computeDerivative(variables,inv));
  
  pairs.push_back(computeDerivative(variables, mixedPoly));
  polys.push_front(mixedPoly);

  invariants.push_back(mixedPoly);

  exp1 = 1;
  exp2 = 1;
  while (true) {	  
    if (computeMixedInvariantOrderPairHelper(invariants, variables, polys, pairs, generator, o1, exp1, o2, exp2, numExpectedMixed) ||
	computeMixedInvariantOrderPairHelper(invariants, variables, polys, pairs, generator, o1, exp1 + 1, o2, exp2, numExpectedMixed) ||
	computeMixedInvariantOrderPairHelper(invariants, variables, polys, pairs, generator, o1, exp1, o2, exp2 + 1, numExpectedMixed))
      break;
	  
    exp1++;
    exp2++;
  }

  return invariants;

}

/**
 * This function computes the mixed invariants for every order pairs
 * The outtermost list sorts the mixed invariants by Order
 * The middle list sorts the mixed invariants by the maximum order of a mixed invariant
 * The innermost list sorts the mixed invariants by the minium order of a mixed invariant
 * @param HomoInvariants a list of homogeneous invariants
 * @param variables a list of moments
 * @return mixed invariants
 */ 
list<list<list<vtkMomentsPolynomial>>> vtkMomentInvariants::computeMixedInvariant(list<vtkMomentsPolynomial> & HomoInvariants, const list<vtkMomentsIndex> & variables) const {
  list<list<list<vtkMomentsPolynomial>>> mixedInvariants;
  unsigned o2 = 0;
  if (FieldRank == 0)
    o2 = 1;
  
  if (Dimension == 2) {
    
    for (; o2 <= Order; o2++) {
      
      list<list<vtkMomentsPolynomial>> mixedInvariantsMaxOrder;
      unsigned o1 = 0;
      if (FieldRank == 0)
	o1 = 1;
      for (; o1 < o2; o1++) {
	list<vtkMomentsPolynomial> mixedInvariantsMinOrder;
	unsigned exp1, exp2;
	
	if ((o1 + FieldRank) % 2 == 0)
	  exp1 = 1;
	else
	  exp1 = 2;

	if ((o2 + FieldRank) % 2 == 0)
	  exp2 = 1;
	else
	  exp2 = 2;

	if ((o1 + FieldRank) % 2 != 0 && (o2 + FieldRank) % 2 != 0) {
	  exp1 = 1;
	  exp2 = 1;
	}

	unsigned rank1 = (o1 + FieldRank) * exp1, rank2 = (o2 + FieldRank) * exp2;
	
	vtkMomentsTensorSimple product(Dimension, o1 + FieldRank, FieldRank), tensor2(Dimension, o2 + FieldRank, FieldRank);

	product.tensorPow(exp1);
	tensor2.tensorPow(exp2);
	product.tensorProduct(tensor2);

	if (rank1 > rank2) {
	  unsigned rem = (rank1 - rank2) / 2;
	  for (unsigned i = 0; i < rem; i++)
	    product.contract(0,1);
	  
	  for (unsigned i = rank2; i > 0; i--) 
	    product.contract(0,i);	  
	}
	else {
	  unsigned rem = (rank2 - rank1) / 2;
	  
	  for (unsigned i = rank1; i > 0; i--) 
	    product.contract(0,i);

	  for (unsigned i = 0; i < rem; i++)
	    product.contract(0,1);
	}
	
	vtkMomentsPolynomial poly(product);

	mixedInvariantsMinOrder.push_back(poly);
	mixedInvariantsMaxOrder.push_back(mixedInvariantsMinOrder);
      }

      if (!mixedInvariantsMaxOrder.empty())
	mixedInvariants.push_back(mixedInvariantsMaxOrder);
    }
    
  }
  else if (Dimension == 3) {
    for (; o2 <= Order; o2++) {
      
      list<list<vtkMomentsPolynomial>> mixedInvariantsMaxOrder;
      unsigned o1 = 0;
      if (FieldRank == 0)
	o1 = 1;
      for (; o1 < o2; o1++) {
	list<vtkMomentsPolynomial> mixedInvariantsMinOrder = computeMixedInvariantsOrderPair(HomoInvariants, variables, o1, o2);
	mixedInvariantsMaxOrder.push_back(mixedInvariantsMinOrder);
      }

      if (!mixedInvariantsMaxOrder.empty())
	mixedInvariants.push_back(mixedInvariantsMaxOrder);
    }
  }
  else
    vtkErrorMacro("vtkMomentInvariants::computeMixedInvariant: Dimension which is not 2 or 3 is not supported");

  return mixedInvariants;
}

/**
 * The maximum number of invariants can be computed theoretically.
 * This function provides the expected number of invariants.
 * @param d dimension
 * @param fr field rank
 * @param o order
 * @return a vector of length 5. They are moments of order o, existing homo. inv. of order o, moments up to order o_m, theor. existing inv. up to order o_m and mixed inv. needed
 */
vector<int> vtkMomentInvariants::getInvariantNumberVec(int d, int fr, int o) {
  vector<int> num(5);

  // hard coded part
  if (fr == 0 && o == 0) {
    num[0] = 1;
    num[1] = 1;
    num[2] = 1;
    num[3] = 1;
    num[4] = 0;

    return num;
  }

  if (d ==  3 && fr == 0 && o == 1) {
    num[0] = 3;
    num[1] = 1;
    num[2] = 4;
    num[3] = 2;
    num[4] = 0;

    return num;
  } 

  if (d == 3 && fr == 1 && o == 0) {
    num[0] = 3;
    num[1] = 1;
    num[2] = 3;
    num[3] = 1;
    num[4] = 0;

    return num;
  }

  // compute the maximum number based on certain rules
  if (d == 2) {
    switch (fr) {
    case 0:
      num[0] = o + 1;      
      num[2] = (o + 1) * (o + 2) / 2;      
      num[4] = o - 1;
      break;
    case 1:
      num[0] = 2 * (o + 1);      
      num[2] = (o + 1) * (o + 2);      
      num[4] = o;
      break;
    case 2:
      num[0] = 4 * (o + 1);
      num[2] = 2 * (o + 1) * (o + 2);      
      num[4] = o;
      break;
    }
    num[1] = num[0] - 1;
    num[3] = num[2] - 1;
  }
  else if (d == 3) {
    switch (fr) {
    case 0:
      num[0] = (o + 1) * (o + 2) / 2;      
      num[2] = (o + 1) * (o + 2) * (o + 3)/ 6;      
      num[4] = 3 * o - 4;
      break;
    case 1:
      num[0] = 3 * (o + 1) * (o + 2) / 2;      
      num[2] = (o + 1) * (o + 2) * (o + 3) / 2;      
      num[4] = 3 * o - 1;
      break;
    case 2:
      num[0] = 9 * (o + 1) * (o + 2) / 2;      
      num[2] = 3 * (o + 1) * (o + 2) * (o + 3) / 2;      
      num[4] = 3 * o;
      break;
    }
    num[1] = num[0] - 3;
    num[3] = num[2] - 3;
  }
  else {
    cerr << "getInvariantNumberVec: dimension is neither 2 nor 3\n";
  }

  return num;
}

/**
 * This function creates moments up to the Order.
 * The moments are sorted, e.g. M0,M00,M01,M11...
 * @return a list of moments
 */
list<vtkMomentsIndex> vtkMomentInvariants::createVariables() const {  
  list<vtkMomentsIndex> variables;
  
  for (unsigned i = 0; i <= Order; i++) {
    list<vtkMomentsIndex> variables_temp = vtkMomentsIndex::generateVariables(Dimension, i, FieldRank);
    variables.splice(variables.cend(),variables_temp);
  }

  return variables;
}

/**
 * This function assigns random values to moments.
 * @param variables a list of moments
 * @param generator a random number generator
 * @return a map whose key is moment and value is the value of a moment.
 */  
indexValueMap vtkMomentInvariants::generateValues(const list<vtkMomentsIndex> variables, default_random_engine & generator) {
  uniform_real_distribution<double> distribution(-1.0,1.0);
  indexValueMap map;

  for (auto & var:variables)
    map.emplace(var,distribution(generator));

  return map;
}

/**
 * This function computes derivatives of a given polynomial with repsect to given moments
 * @param variables a list of moments
 * @param poly a the polynomial 
 * @return the derivatives with respect to moments. 
 */
polyDerivativePair vtkMomentInvariants::computeDerivative(const list<vtkMomentsIndex> variables, const vtkMomentsPolynomial & poly) {
  vector<vtkMomentsPolynomial> derivatives(variables.size(), poly);

  auto itDer = derivatives.begin();
  auto itVar = variables.cbegin();

  for (;itDer != derivatives.end(); ++itDer, ++itVar)
    (*itDer).takeDerivative(*itVar);

  return polyDerivativePair(poly, derivatives);
}

/**
 * This function tries to insert an invariant to a list of invariants. 
 * If this invariant is indepedent to the invariants in the list,
 * it will be inserted. 
 * @param generator a random number generator
 * @param indInv derivatives of already found independent polynimals
 * @param variables a list of moments
 * @param poly a polynomial that is going to be added
 * @return a boolean indicating whether or not the polynomial is added
 */
bool vtkMomentInvariants::addIndependentInvariants(default_random_engine & generator,
			      list<polyDerivativePair> & indInv,
			      const list<vtkMomentsIndex> variables,
			      const vtkMomentsPolynomial & poly ) const{
  
  polyDerivativePair pair = computeDerivative(variables,poly);
  Eigen::MatrixXf mat(variables.size(),indInv.size() + 1);
  
  for (unsigned i = 0; i < 5; i++) {
    indexValueMap map = vtkMomentInvariants::generateValues(variables,generator);
    unsigned j = 0;
    for (auto it = indInv.cbegin(); it != indInv.cend(); ++it, j++)
      for (unsigned k = 0; k < variables.size(); k++) 
	mat(k,j) = (*it).second.at(k).assignValue(map);
    
    for (unsigned k = 0; k < variables.size(); k++)
    	mat(k,j) = pair.second.at(k).assignValue(map);


    Eigen::FullPivLU<Eigen::MatrixXf> lu(mat);
    lu.setThreshold(luThreshold);


    if (lu.rank() > indInv.size()) {
      indInv.push_back(pair);
      return true;
    }
  }

  return false;
  
}

/**
 * Given a tensor, this function computes the contractions of it and then creates corresponding polynomials.
 * The function inserts those polynomials (invariants) to a list of independent invariants.
 * If the number of invariants reaches its theoretical maximum, then it will not compute the rest polynomials.
 * @param tensor the input tensor
 * @param invariants a list of indenpent polynomials
 * @param polys a set of existing polynomials
 * @param variables a list of moments
 * @param pairs derivatives of already found independent polynomials
 * @param generator a random number generator
 * @param maxNumber theoretical maximum number of independent polynomials
 * @return a boolean indicating whether or not the maximum number has been reached.
 */ 
bool vtkMomentInvariants::computeInvariant(vtkMomentsTensorSimple & tensor, list<vtkMomentsPolynomial> & invariants, forward_list<vtkMomentsPolynomial> & polys, const list<vtkMomentsIndex> & variables, list<polyDerivativePair> & pairs, default_random_engine & generator, int maxNumber) const {
  if (tensor.getRank() == 0) {
    vtkMomentsPolynomial poly(tensor);
      
    if (polys.empty()) {      
      if (!pairs.empty()) {
	cerr << "computeInvariant error: pairs is not empty.\n";
	exit(0);
      }
      polys.push_front(poly);
      pairs.push_front(vtkMomentInvariants::computeDerivative(variables, poly));
      invariants.push_back(poly);
    }    
    else {
      auto itPolys = polys.cbefore_begin();
      while (true)
	if (next(itPolys) == polys.cend() || poly < *(next(itPolys))) {
	  polys.insert_after(itPolys, poly);
	  if(addIndependentInvariants(generator, pairs, variables, poly)) 
	    invariants.push_back(poly);
	  break;
	}
	else if (poly == *next(itPolys)) 
	  break;
	else
	  ++itPolys;
    }
  }
  else {
    for (unsigned i = 1; i < tensor.getRank(); ++i) {
      vtkMomentsTensorSimple contractedTensor(tensor, (unsigned)0, i);
      if (computeInvariant(contractedTensor,invariants, polys, variables, pairs, generator, maxNumber))
	return true;
    }
  }

  if (invariants.size() == maxNumber && maxNumber != -1)
    return true;
  else
    return false;  
}


/**
 * This function chooses an invariant method based on variable invariantMethod to perform pattern detection.
 * @param[in] momentsInfo moment information
 * @param[in] pattern pattern data
 * @param[in] momentData moment data
 * @param[out] similarityFields the similarities between the pattern and the field
 * @param[out] normalizedMomentsField normalized moment data
 * @param[out] originalMomentsPattern moments of pattern
 * @param[out] normalizedMomentsPattern normalized moments of pattern
 */
void vtkMomentInvariants::momentInvariantsPatternDetection(vtkInformation* momentsInfo, vtkImageData* pattern, vtkImageData* momentData, vtkImageData* similarityFields, vtkImageData* normalizedMomentsField, vtkImageData* originalMomentsPattern, vtkImageData* normalizedMomentsPattern) {
  std::vector<std::vector<vtkMomentsTensor> > dominantContractions;
  
  this->HandlePatternNormalization(dominantContractions, pattern, originalMomentsPattern, normalizedMomentsPattern);
  
  if (invariantMethod == "Normalization" && momentsInfo) 
    this->HandleFieldNormalization(dominantContractions, momentData, normalizedMomentsField, pattern, similarityFields);  
  else if (invariantMethod == "Generator" || invariantMethod == "LangbeinInvariants") {   
    if (!(this->IsRotation || this->IsReflection) && momentsInfo) 
      this->HandleFieldNormalization(dominantContractions, momentData, normalizedMomentsField, pattern, similarityFields);    
    else {
      list<vtkMomentsIndex> variables = createVariables();
      indexValueMap values;
      for (auto & var : variables) {
	values.emplace(var,0);
	this->patternMoments.emplace(var,0);
      }

      for (auto & val:this->patternMoments)
	val.second = MomentsPatternTSNormal[val.first.getMomentRank()].get(val.first.getIndices());

      if (invariantMethod == "Generator") {
	list<list<vtkMomentsPolynomial>> homoInvStructure;
	list<list<list<vtkMomentsPolynomial>>> mixedInvStructure;
 	bool recomputeFlag = false;

	switch (Dimension) {
	case 2:
	  switch (FieldRank) {
	  case 0:
	    if (Order > generator2DScalarOrder6.order) 	  
	      recomputeFlag = true;	
	    else {
	      homoInvStructure = generator2DScalarOrder6.getHomogeneousInvariants(Order);
	      mixedInvStructure = generator2DScalarOrder6.getMixedInvariants(Order);	  
	    }	  
	    break;
	  case 1:
	    if (Order > generator2DVectorOrder5.order) 	  
	      recomputeFlag = true;	
	    else {
	      homoInvStructure = generator2DVectorOrder5.getHomogeneousInvariants(Order);
	      mixedInvStructure = generator2DVectorOrder5.getMixedInvariants(Order);	  
	    }	  
	    break;
	  case 2:
	    if (Order > generator2DMatrixOrder4.order) 	  
	      recomputeFlag = true;	
	    else {
	      homoInvStructure = generator2DMatrixOrder4.getHomogeneousInvariants(Order);
	      mixedInvStructure = generator2DMatrixOrder4.getMixedInvariants(Order);	  
	    }	  
	    break;
	  default:
	    vtkErrorMacro("Generator Invaraints: FieldRank > 3 || FieldRank < 0.");
	  }
	  break;      
	case 3:
	  switch (FieldRank) {
	  case 0:
	    if (Order > generator3DScalarOrder4.order) 	  
	      recomputeFlag = true;	
	    else {
	      homoInvStructure = generator3DScalarOrder4.getHomogeneousInvariants(Order);
	      mixedInvStructure = generator3DScalarOrder4.getMixedInvariants(Order);	  
	    }	  
	    break;
	  case 1:
	    if (Order > generator3DVectorOrder3.order) 	  
	      recomputeFlag = true;	
	    else {
	      homoInvStructure = generator3DVectorOrder3.getHomogeneousInvariants(Order);
	      mixedInvStructure = generator3DVectorOrder3.getMixedInvariants(Order);	  
	    }	  
	    break;
	  case 2:
	    if (Order > generator3DMatrixOrder2.order) 	  
	      recomputeFlag = true;	
	    else {
	      homoInvStructure = generator3DMatrixOrder2.getHomogeneousInvariants(Order);
	      mixedInvStructure = generator3DMatrixOrder2.getMixedInvariants(Order);	  
	    }	  
	    break;
	  default:
	    vtkErrorMacro("Generator Invaraints: FieldRank > 3 || FieldRank < 0.");
	  }
	  break;
	default:
	  vtkErrorMacro("Generator Invaraints: Dimension is neither 2 nor 3.");
	} // switch	

	if (recomputeFlag) {
	  cout << "Computing the generator invariants from scratch could take a long time.\n";
	  
	  homoInvStructure.clear();
	  mixedInvStructure.clear();
	    
	  list<vtkMomentsPolynomial> homogenerousInvariants;
	  default_random_engine generator;

	  // time_t begin,end;
	  // time(&begin);
	  int i = 0;
	  if (FieldRank == 0) {
	    i = 1;
	    homogenerousInvariants.push_front(vtkMomentsTensorSimple(Dimension, 0, FieldRank));
	    homoInvStructure.push_front(homogenerousInvariants);
	  }

	  for (; i <= Order; i++) 
	    homoInvStructure.push_back(computeHomogeneousInvariant(generator, homogenerousInvariants, i));

	  mixedInvStructure = computeMixedInvariant(homogenerousInvariants, variables);
	  // time(&end);
	  // cout << difftime(end,begin) << " seconds\n";
	}

	list<vtkMomentsPolynomial> invariants;
	auto patternDesc = handlePatternGenerator(variables, values, invariants, homoInvStructure, mixedInvStructure, normalizedMomentsPattern);
	
	if (momentsInfo)
	  handleFieldGenerator(invariants, values, patternDesc, momentData, pattern, similarityFields, normalizedMomentsField);
      }
      else {
	bool recomputeFlag = false;

	// cout << "LangbeinInvariants" << endl;
    
	switch (Dimension) {
	case 2:
          switch (FieldRank) {
	  case 0:
	    if (Order > langbein2DScalarOrder6.order) 	  
	      recomputeFlag = true;	
	    else 
	      langbeinInvariants = langbein2DScalarOrder6.getLangbeinInvariants(Order);
	    break;
	  case 1:
	    if (Order > langbein2DVectorOrder5.order) 	  
	      recomputeFlag = true;	
	    else 
	      langbeinInvariants = langbein2DVectorOrder5.getLangbeinInvariants(Order);
	    break;
	  case 2:
	    if (Order > langbein2DMatrixOrder4.order) 	  
	      recomputeFlag = true;	
	    else 
	      langbeinInvariants = langbein2DMatrixOrder4.getLangbeinInvariants(Order);
	    break;
	  default:
	    vtkErrorMacro("Langbein Invaraints: FieldRank > 3 || FieldRank < 0.");
	  }
	  break;      
	case 3:
	  switch (FieldRank) {
	  case 0:
	    if (Order > langbein3DScalarOrder4.order) 	  
	      recomputeFlag = true;	
	    else 
	      langbeinInvariants = langbein3DScalarOrder4.getLangbeinInvariants(Order);
	    break;
	  case 1:
	    if (Order > langbein3DVectorOrder3.order) 	  
	      recomputeFlag = true;	
	    else 
	      langbeinInvariants = langbein3DVectorOrder3.getLangbeinInvariants(Order);
	    break;
	  case 2:
	    if (Order > langbein3DMatrixOrder2.order) 	  
	      recomputeFlag = true;	
	    else 
	      langbeinInvariants = langbein3DMatrixOrder2.getLangbeinInvariants(Order);
	    break;
	  default:
	    vtkErrorMacro("Langbein Invaraints: FieldRank > 3 || FieldRank < 0.");
	  }
	  break;
	default:
	  vtkErrorMacro("Langbein Invaraints: Dimension is neither 2 nor 3.");
	}

	if (recomputeFlag) {
	  cout << "Computing the Langbein invariants from scratch could take a long time.\n";
	  time_t begin,end;
	  time(&begin);
	  langbeinInvariants = computeLangbeinInvariants();
	  time(&end);
	  cout << difftime(end,begin) << " seconds\n";
	}
	

	auto patternDesc = handlePatternLangbeinInvariants(langbeinInvariants, values, normalizedMomentsPattern);

	if (momentsInfo)
	  handleFieldGenerator(langbeinInvariants,values,patternDesc, momentData, pattern, similarityFields, normalizedMomentsField);
      }
    } 
  }
  else     
    vtkErrorMacro("The invariant method name has to be 'Normalization', 'Generator' or 'LangbeinInvariants'. It is " + invariantMethod);
  
}

/**
 * Print invariants and values for generator and Langbein algorithm
 */
void vtkMomentInvariants::printInvariants()
{
  if (this->invariantMethod == "Normalization")
  {
    cout << "momentsPatternNormal" << endl;
    for (int i = 0; i < static_cast<int>(this->MomentsPatternNormal.size()); ++i)
    {
      cout << "momentsPatternNormal.at(" << i << ")" << endl;
      for (int k = 0; k < static_cast<int>(this->MomentsPatternNormal.at(i).size()); ++k)
      {
        this->MomentsPatternNormal.at(i).at(k).print();
      }
    }
  }

  if (this->invariantMethod == "Generator") {
    cout << "Generator Invariants:\n";
    
    for (auto & xs : this->generatorInvariants) {
      cout << "Order " << xs.front().getMaximumTensorOrder() << ":\n";

      for (auto & x : xs)
	cout << x.printTensorConcise() << " = " << x.assignValueAndNormalize(this->patternMoments) << endl;
	// cout << x.printTensorConcise() << " = " << x << endl;
      
	
    }
  }
  else if (this->invariantMethod == "LangbeinInvariants") {
    cout << "Langbein Invariants:\n";

    for (auto & inv : langbeinInvariants)
      cout << inv.printTensorConcise() << " = " << inv.assignValueAndNormalize(this->patternMoments) << endl;
      // cout << inv.printTensorConcise() << " = " << inv << endl;
  }
}

/**
 * This function creates a name for an invariant. The name is used to for storing to vtkImageData.
 * @param radius the radius that is used to compute moments
 * @param invariant an invariant in the polynomial format
 */
string vtkMomentInvariants::getArrayNameFromInvariant(double radius, const vtkMomentsPolynomial & invariant) {
  ostringstream ss;
  ss << std::fixed << std::setprecision(2) << radius;
  return "R" + ss.str() + "_" + invariant.printTensorConcise();
}

int vtkMomentInvariants::RequestInformation(vtkInformation*,
                                            vtkInformationVector** inputVector,
                                            vtkInformationVector* outInfoVector)
{
  vtkInformation* momentInfo = inputVector[MomentPort]->GetInformationObject(0);
  // similarity fields
  vtkInformation* outInfo0 = outInfoVector->GetInformationObject(0);
  // normalized moments of the field
  vtkInformation* outInfo1 = outInfoVector->GetInformationObject(1);
  // moments of the pattern
  vtkInformation* outInfo2 = outInfoVector->GetInformationObject(2);
  // normalized moments of the pattern
  vtkInformation* outInfo3 = outInfoVector->GetInformationObject(3);
  int zeroExtent[6] = {0,0,0,0,0,0};
  outInfo2->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
                zeroExtent, 6);
  outInfo3->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
                zeroExtent, 6);
  if (momentInfo)
  {
    outInfo0->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
                 momentInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
                 6);
    outInfo1->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
                  momentInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
                  6);
  }
  return 1;
}

int vtkMomentInvariants::RequestUpdateExtent(vtkInformation*,
                                             vtkInformationVector** inputVector,
                                             vtkInformationVector* outInfoVector)
{
  vtkInformation* patternInfo = inputVector[PatternPort]->GetInformationObject(0);
  vtkInformation* momentInfo = inputVector[MomentPort]->GetInformationObject(0);
  vtkInformation* outInfo0  = outInfoVector->GetInformationObject(0);
  //vtkInformation* outInfo1  = outInfoVector->GetInformationObject(1);

  if (momentInfo)
  {
    momentInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
                    outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT()),
                    6);
//    momentInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
//                    outInfo1->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT()),
//                    6);
    int piece = outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER());
    int numPieces = outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES());
    int ghostLevels = outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS());
    momentInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER(), piece);
    momentInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES(), numPieces);
    momentInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS(), ghostLevels);
  }

  // we want the whole pattern on each rank
  patternInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER(), 0);
  patternInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES(), 1);

  patternInfo->Set(vtkStreamingDemandDrivenPipeline::EXACT_EXTENT(), 1);
  patternInfo->Remove(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT());
  if (patternInfo->Has(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()))
  {
    patternInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
                     patternInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
                     6);
  }


  return 1;
}

/** main executive of the program, reads the input, calles the functions, and produces the utput.
 * @param request: ?
 * @param inputVector: the input information
 * @param outputVector: the output information
 */
int vtkMomentInvariants::RequestData(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  // get the info objects
  vtkInformation* patternInfo = inputVector[PatternPort]->GetInformationObject(0);
  vtkInformation* momentsInfo = inputVector[MomentPort]->GetInformationObject(0);
  // similarity fields
  vtkInformation* outInfo0 = outputVector->GetInformationObject(0);
  // normalized moments of the field
  vtkInformation* outInfo1 = outputVector->GetInformationObject(1);
  // moments of the pattern
  vtkInformation* outInfo2 = outputVector->GetInformationObject(2);
  // normalized moments of the pattern
  vtkInformation* outInfo3 = outputVector->GetInformationObject(3);

  // get the input and output
  vtkImageData* pattern =
    vtkImageData::SafeDownCast(patternInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* momentData = nullptr;
  if (momentsInfo)
  {
    momentData = vtkImageData::SafeDownCast(momentsInfo->Get(vtkDataObject::DATA_OBJECT()));
  }
  vtkImageData* similarityFields =
    vtkImageData::SafeDownCast(outInfo0->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* normalizedMomentsField =
    vtkImageData::SafeDownCast(outInfo1->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* originalMomentsPattern =
    vtkImageData::SafeDownCast(outInfo2->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* normalizedMomentsPattern =
    vtkImageData::SafeDownCast(outInfo3->Get(vtkDataObject::DATA_OBJECT()));

  CheckValidity(pattern);
  this->InterpretPattern(pattern);

  if (momentsInfo)
  {
    this->InterpretField(momentData);
  }

  momentInvariantsPatternDetection(momentsInfo, pattern, momentData, similarityFields, normalizedMomentsField, originalMomentsPattern, normalizedMomentsPattern);
  
  return 1;
}
