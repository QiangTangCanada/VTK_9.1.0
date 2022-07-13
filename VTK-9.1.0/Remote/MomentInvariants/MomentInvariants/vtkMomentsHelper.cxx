/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkMomentsHelper.cxx

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

#include "vtkMomentsHelper.h"
#include "vtkCell.h"
#include "vtkDoubleArray.h"
#include "vtkImageConstantPad.h"
#include "vtkImageData.h"
#include "vtkImageTranslateExtent.h"
#include "vtkMomentsTensor.h"
#include "vtkNew.h"
#include "vtkPixel.h"
#include "vtkPointData.h"
#include "vtkProbeFilter.h"
#include "vtkQuad.h"
#include "vtkTetra.h"
#include "vtkTriangle.h"

#include <algorithm>    // std::sort
#include <vector>

#ifndef M_PI
#define M_PI vtkMath::Pi()
#endif

/**
 * The monomial basis is not orthonormal. We need this function for the reconstruction of the
 * function from the moments. This function uses Gram Schmidt
 * @param dimension: 2D or 3D
 * @param moments: the moments at a point
 * @param radius: the corresponding integration radius
 * @return the orthonormal moments
 */
std::vector<vtkMomentsTensor> vtkMomentsHelper::orthonormalizeMoments(int dimension,
  std::vector<vtkMomentsTensor> moments,
  double radius)
{
  Eigen::VectorXd b = Eigen::VectorXd::Zero(moments.size() * moments.back().size());
  for (int k = 0; k < static_cast<int>(moments.size()); ++k)
  {
    for (int i = 0; i < static_cast<int>(moments.at(k).size()); ++i)
    {
      b(i + k * moments.back().size()) = moments.at(k).get(i);
    }
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(
    moments.size() * moments.back().size(), moments.size() * moments.back().size());
  for (int k1 = 0; k1 < static_cast<int>(moments.size()); ++k1)
  {
    for (int i1 = 0; i1 < static_cast<int>(moments.at(k1).size()); ++i1)
    {
      for (int k2 = 0; k2 < static_cast<int>(moments.size()); ++k2)
      {
        for (int i2 = 0; i2 < static_cast<int>(moments.at(k2).size()); ++i2)
        {
          if (moments.at(k1).getFieldIndices(i1) == moments.at(k2).getFieldIndices(i2))
          {
            if (dimension == 2)
            {
              A(i1 + k1 * moments.back().size(), i2 + k2 * moments.back().size()) =
                translationFactorAnalytic(radius,
                  2,
                  moments.at(k1).getOrders(i1).at(0) + moments.at(k2).getOrders(i2).at(0),
                  moments.at(k1).getOrders(i1).at(1) + moments.at(k2).getOrders(i2).at(1),
                  0);
            }
            else
            {
              A(i1 + k1 * moments.back().size(), i2 + k2 * moments.back().size()) =
                translationFactorAnalytic(radius,
                  3,
                  moments.at(k1).getOrders(i1).at(0) + moments.at(k2).getOrders(i2).at(0),
                  moments.at(k1).getOrders(i1).at(1) + moments.at(k2).getOrders(i2).at(1),
                  moments.at(k1).getOrders(i1).at(2) + moments.at(k2).getOrders(i2).at(2));
            }
          }
        }
      }
    }
  }

  //#ifdef MYDEBUG
  //    Eigen::MatrixXd IA = Eigen::MatrixXd::Zero( moments.size() * moments.back().size(),
  //                                               moments.size() * moments.back().size() );
  //    for( int k1 = 0; k1 < static_cast<int>(moments.size()); ++k1 )
  //    {
  //        for( int i1 = 0; i1 < static_cast<int>(moments.at( k1 ).size()); ++i1 )
  //        {
  //            for( int k2 = 0; k2 < static_cast<int>(moments.size()); ++k2 )
  //            {
  //                for( int i2 = 0; i2 < static_cast<int>(moments.at( k2 ).size()); ++i2 )
  //                {
  //                    if( moments.at( k1 ).getFieldIndices( i1 ) == moments.at( k2
  //                                                                             ).getFieldIndices(
  //                                                                             i2 ) )
  //                    {
  //                        for( int d = 0; d < dimension; ++d )
  //                        {
  //                            IA( i1 + k1 * moments.back().size(), i2 + k2 * moments.back().size()
  //                            )
  //                            += ( moments.at( k1 ).getOrders( i1 ).at( d ) + moments.at( k2
  //                                                                                       ).getOrders(
  //                                                                                       i2 ).at(
  //                                                                                       d ) ) *
  //                                                                                       pow( 10,
  //                                                                                       d );
  //                        }
  //                    }
  //                }
  //            }
  //        }
  //    }
  //    Eigen::VectorXd Ib = Eigen::VectorXd::Zero( moments.size() * moments.back().size() );
  //    for( int k = 0; k < static_cast<int>(moments.size()); ++k )
  //    {
  //        for( int i = 0; i < static_cast<int>(moments.at( k ).size()); ++i )
  //        {
  //            for( int d = 0; d < dimension; ++d )
  //            {
  //                Ib( i + k * moments.back().size() ) += moments.at( k ).getOrders( i ).at( d ) *
  //                pow( 10, d );
  //            }
  //        }
  //    }
  //    std::cout<<"b="<<b<<endl;
  //    std::cout<<"Ib="<<Ib<<endl;
  //    std::cout<<"A="<<A<<endl;
  //    std::cout<<"IA="<<IA<<endl;
  //#endif

  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
  //    std::cout<<"x="<<x<<endl;
  //    assert( b.isApprox( A * x ) );
  std::vector<vtkMomentsTensor> orthonormalMoments(moments.size());
  for (int k = 0; k < static_cast<int>(moments.size()); ++k)
  {
    orthonormalMoments.at(k) = vtkMomentsTensor(moments.at(k));
    for (int i = 0; i < static_cast<int>(moments.at(k).size()); ++i)
    {
      orthonormalMoments.at(k).set(i, x(i + k * moments.back().size()));
    }
  }

  //#ifdef MYDEBUG
  //    std::cout<<"x="<<x<<endl;
  //    std::cout<<"Ax-b="<<A*x-b<<endl;
  //    for( int k = 0; k < static_cast<int>(moments.size()); ++k )
  //    {
  //        moments.at( k ).print();
  //        orthonormalMoments.at( k ).print();
  //    }
  //    cout<<"orthonormalMoments"<<endl;
  //    for( int k = 0; k < static_cast<int>(moments.size()); ++k )
  //    {
  //        orthonormalMoments.at( k ).print();
  //    }
  //#endif

  return orthonormalMoments;
}

/**
 * This function computes the moments at a given location and radius
 * the moments are the projections of the function to the monomial basis
 * they are evaluated using a numerical integration over the original dataset if it is structured
 * data
 * @param dimension: 2D or 3D
 * @param order: the maximal order up to which the moments are computed
 * @param fieldRank: 0 for scalar, 1 for vector and 2 for matrix
 * @param radius: the integration radius at which the moments are computed
 * @param dimPtId: array of dimension-wise point ids of the location where the moments are computed.
 * On a foreign proc, at least one is negative
 * @param field: the dataset of which the moments are computed
 * @param nameOfPointData: the name of the array in the point data of which the momens are computed.
 * @return the moments
 */
std::vector<vtkMomentsTensor> vtkMomentsHelper::allMomentsOrigResImageData(int dimension, int order,
  int fieldRank, double radius, int* dimPtIdGrid, vtkImageData* field, vtkImageData* grid, std::string nameOfPointData)
{
  std::vector<vtkMomentsTensor> tensors(order + 1);
  for (int o = 0; o < order + 1; o++)
  {
    tensors.at(o) = vtkMomentsTensor(dimension, o + fieldRank, fieldRank);
  }

  double gridBounds[6];
  grid->GetBounds(gridBounds);
  double center[3] = { 0.0, 0.0, 0.0 };
  for (int d = 0; d < dimension; ++d)
  {
    center[d] = gridBounds[2 * d] + dimPtIdGrid[d] * grid->GetSpacing()[d];
//    std::cout<<"centers and bounds grid: "<<center[d]<<" "<<gridBounds[2*d]<<" "<<dimPtIdGrid[d]<<"\n";
  }
  int dimPtId[3] = {0,0,0};
  double fieldBounds[6];
  field->GetBounds(fieldBounds);

  for (int d = 0; d < dimension; ++d)
  {
    dimPtId[d] = round((center[d] - fieldBounds[2 * d]) / field->GetSpacing()[d]);
//    std::cout<<"centers and bounds field: "<<center[d]<<" "<<fieldBounds[2*d]<<" "<<dimPtId[d]<<"\n";
  }
  
  double argument[3] = { 0.0, 0.0, 0.0 };
  double relArgument[3] = { 0.0, 0.0, 0.0 };
  double* h = field->GetSpacing();
  if (dimension == 2)
  {
    h[2] = 1;
  }
  int discreteRadius[3] = {0,0,0};
  
  for (int d = 0; d < dimension; ++d)
  {
    discreteRadius[d] = round (radius / h[d]);
  }
  vtkSmartPointer<vtkDataArray> fieldArray = field->GetPointData()->GetArray(nameOfPointData.c_str());
  int fieldDims[3];
  field->GetDimensions(fieldDims);
  // The three following continues serve 2 purposes
  // 1. If the index points outside of the domain (dimPtId[2] + k < 0 or dimPtId[2] + k >= fieldDims[2]), we don't do anything
  // 2. They are needed for the data parallel computation:
  // The points on the boundary between two nodes are owned by both nodes and would contribute twice in the sum
  // To avoid this, we do not add the lower left boundary points (dimPtId[2] + k == 0)
  // There is one exception though:
  // If the lower boundary point is the last point on the ball (k == -discreteRadius[2] and dimPtId[2] + k == 0)
  // because then, the center was not sent to the neighbor, because all info is available on this node
  for (int k = -discreteRadius[2]; k <= discreteRadius[2]; k++)
  {
    if ((k > -discreteRadius[2] && dimPtId[2] + k == 0) || dimPtId[2] + k < 0 || dimPtId[2] + k >= fieldDims[2])
    {
      continue;
    }
    for (int j = -discreteRadius[1]; j <= discreteRadius[1]; j++)
    {
      if ((j > -discreteRadius[1] && dimPtId[1] + j == 0) || dimPtId[1] + j < 0 || dimPtId[1] + j >= fieldDims[1])
      {
        continue;
      }
      for (int i = -discreteRadius[0]; i <= discreteRadius[0]; i++)
      {
        if ((i > -discreteRadius[0] && dimPtId[0] + i == 0) || dimPtId[0] + i < 0 || dimPtId[0] + i >= fieldDims[0])
        {
          continue;
        }
        int index = dimPtId[0] + i + (dimPtId[1] + j) * fieldDims[0] +
          (dimPtId[2] + k) * fieldDims[0] * fieldDims[1];
        field->GetPoint(index, argument);
        for (int d = 0; d < dimension; ++d)
        {
          relArgument[d] = 1./ radius * (argument[d] - center[d]);
        }
//        std::cout<<i<<" "<<j<<" "<<k<<" "<<"vtkMath::Norm(relArgument): "<<vtkMath::Norm(relArgument)<< "vtkMath::Norm(relArgument) <= 1 + 1e-10: "<<(vtkMath::Norm(relArgument) <= 1 + 1e-5)<<"\n";

        if (vtkMath::Norm(relArgument) <= 1 + 1e-10)
        {
          for (int o = 0; o < order + 1; o++)
          {
            for (int s = 0; s < static_cast<int>(tensors.at(o).size()); s++)
            {
              if ( vtkMomentsHelper::isOrdered(tensors.at(o).getIndices(s), fieldRank) )
              {
                double faktor = 1;
                std::vector<int> midx = tensors.at(o).getMomentIndices(s);
                for (int mi = 0; mi < static_cast<int>(midx.size());
                     mi++)
                {
                  faktor *= relArgument[midx.at(mi)];
                }
                tensors.at(o).set(s, tensors.at(o).get(s) + 1./ pow(radius, dimension) *
                                  h[0] * h[1] * h[2] * faktor *
                                  fieldArray->GetTuple(index)[tensors.at(o).getFieldIndex(s)]);
                // std::cout << "outputFieldIndex="<<getFieldIndexFromTensorIndices(tensors.at( o
                // ).getIndices(s))
                //<<
                //"="<<getTensorIndicesFromFieldIndexAsString(getFieldIndexFromTensorIndices(tensors.at(
                // o ).getIndices(s)))<< " tensors.at( o ).getFieldIndex( s )="<<tensors.at( o
                //).getFieldIndex( s )<<"\n";  std::cout<<"radius="<<radius<<"
                // value="<<tensors.at(o).get(s)<<"\n";
              }
            }
          }
        }
      }
    }
  }

  return normalizeMoments(tensors);
}

/**
 * This function computes the moments at a given location and radius
 * the moments are the projections of the function to the monomial basis
 * they are evaluated using a numerical integration over uniformly sampled 2D or 3D space
 * @param dimension: 2D or 3D
 * @param order: the maximal order up to which the moments are computed
 * @param fieldRank: 0 for scalar, 1 for vector and 2 for matrix
 * @param radius: the integration radius at which the moments are computed
 * @param center: location where the moments are computed
 * @param stencil: contains the locations at which the dataset is evaluated for the integration
 * @param nameOfPointData: the name of the array in the point data of which the momens are computed.
 * @return the moments
 */
std::vector<vtkMomentsTensor> vtkMomentsHelper::allMoments(int dimension,
  int order,
  int fieldRank,
  double radius,
  double center[3],
  vtkImageData* stencil,
  std::string nameOfPointData)
{
  std::vector<vtkMomentsTensor> tensors(order + 1);
  for (int k = 0; k < order + 1; ++k)
  {
    tensors.at(k) = vtkMomentsTensor(dimension, k + fieldRank, fieldRank);
  }
  double argument[3];
  double relArgument[3];
  double* h = stencil->GetSpacing();
  if (dimension == 2)
  {
    h[2] = 1;
  }
  for (vtkIdType ptId = 0; ptId < stencil->GetNumberOfPoints(); ++ptId)
  {
    stencil->GetPoint(ptId, argument);
    for (int d = 0; d < 3; ++d)
    {
      relArgument[d] = 1. / radius * (argument[d] - center[d]);
    }
    //        if(center[0]==16&&center[1]==-3) std::cout<<relArgument[0]<<" "<<relArgument[1]<<"\n";
    if (vtkMath::Norm(relArgument) <= 1)
    {
      for (int o = 0; o < order + 1; o++)
      {
        for (int s = 0; s < static_cast<int>(tensors.at(o).size()); s++)
        {
          if ( vtkMomentsHelper::isOrdered(tensors.at(o).getIndices(s), fieldRank) )
          {
            double faktor = 1;
            for (int i = 0; i < static_cast<int>(tensors.at(o).getMomentIndices(s).size()); ++i)
            {
              faktor *= relArgument[tensors.at(o).getMomentIndices(s).at(i)];
            }
            //                    if(center[0]==16&&center[1]==-3)
            //                        std::cout<<relArgument[0]<<" "<<relArgument[1]<<" o="<<o<<"
            //                        s="<<s<<" faktor="<<faktor<<"\n";
            tensors.at(o).set(s,
              tensors.at(o).get(s) + 1. / pow(radius, dimension) *
                h[0] * h[1] * h[2] * faktor *
                  stencil->GetPointData()
                    ->GetArray(nameOfPointData.c_str())
                    ->GetTuple(ptId)[tensors.at(o).getFieldIndex(s)]);
          }
        }
      }
    }
  }
  return normalizeMoments(tensors);
}

/**
 * This function computes the factor that needs to be removed for the translational normalization
 * it corresponds to the moment of the function identical to one
 * if we do not know the analytic solution, we evaluate it numerically 
 * @param dimension: e.g. 2D or 3D
 * @param radius: the integration radius at which the moments are computed
 * @param p: exponent in the basis function x^p*y^q*z^r
 * @param q: exponent in the basis function x^p*y^q*z^r
 * @param r: exponent in the basis function x^p*y^q*z^r
 * @param stencil: contains the locations at which the dataset is evaluated for the integration
 * @return the translation factor
 */
double vtkMomentsHelper::translationFactor(int dimension, double radius, int p, int q, int r, vtkImageData* stencil)
{
  if (p % 2 == 1 || q % 2 == 1 || r % 2 == 1)
  {
    return 0;
  }
  double center[3];
  for (int d = 0; d < 3; ++d)
  {
    center[d] = 0.5 * (stencil->GetBounds()[2 * d + 1] + stencil->GetBounds()[2 * d]);
  }
  double argument[3];
  double relArgument[3];
  double* h = stencil->GetSpacing();
  double integral = 0;
  for (int ptId = 0; ptId < stencil->GetNumberOfPoints(); ++ptId)
  {
    stencil->GetPoint(ptId, argument);
    for (int d = 0; d < 3; ++d)
    {
      relArgument[d] = argument[d] - center[d];
    }
    //        std::cout<<"relArgument="<<relArgument[0]<<" "<<relArgument[1]<<"\n";
    if (vtkMath::Norm(relArgument) < radius + 1e-5)
    {
      integral += h[0] * h[1] * h[2] * pow(relArgument[0], p) * pow(relArgument[1], q) *
        pow(relArgument[2], r);
    }
  }
//  for (int d = 0; d < 3; ++d)
//  {
//    integral /= radius;
//  }

//        std::cout<<"radius="<<radius<<" p="<<p<<" q="<<q<<" r="<<r<<" translationFactor="<<integral<<"\n";
  return (integral / normalizationWeight(dimension, p, q, r));
}

/**
 * This function computes the factor that needs to be removed for the translational normalization
 * it corresponds to the moment of the function identical to one
 * for the lowest orders, we know the analytic solution
 * @param radius: the integration radius at which the moments are computed
 * @param dimension: 2D or 3D
 * @param p: exponent in the basis function x^p*y^q*z^r
 * @param q: exponent in the basis function x^p*y^q*z^r
 * @param r: exponent in the basis function x^p*y^q*z^r
 * @return the translation factor
 */
double vtkMomentsHelper::translationFactorAnalytic(
  double radius, int dimension, int p, int q, int r)
{
  if (dimension == 2)
  {
    if (p % 2 == 0 && q % 2 == 0)
    {
      if (p + q == 0)
      {
        return vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
      }
      else if (p + q == 2)
      {
        return 1. / 4 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
      }
      else if (p + q == 4)
      {
        if (p == 4 || q == 4)
        {
          return 1. / 8 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
        }
        else
        {
          return 1. / 24 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
        }
      }
      else if (p + q == 6)
      {
        if (p == 6 || q == 6)
        {
          return 5. / 64 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
        }
        else
        {
          return 1. / 64 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
        }
      }
      else if (p + q == 8)
      {
        if (p == 8 || q == 8)
        {
          return 7. / 128 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
        }
        else if (p == 6 || q == 6)
        {
          return 1. / 128 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
        }
        else
        {
          return 3. / 640 * vtkMath::Pi() * pow(radius, 2 + p + q) / normalizationWeight(dimension, p, q, r);
        }
      }
      else
      {
        double center[3] = { 0.0, 0.0, 0.0 };
        vtkNew<vtkImageData> stencil;
        stencil->SetDimensions(25, 25, 1);
        stencil->SetSpacing(2. * radius / 25, 2. * radius / 25, 1);
        stencil->SetOrigin(center);
        return (translationFactor(dimension, radius, p, q, r, stencil));
      }
    }
    else
    {
      return 0;
    }
  }
  else
  {
    if (p % 2 == 0 && q % 2 == 0 && r % 2 == 0)
    {
      if (p + q + r == 0)
      {
        return 4. / 3 * vtkMath::Pi() * pow(radius, 3 + p + q + r) / normalizationWeight(dimension, p, q, r);
      }
      else if (p + q + r == 2)
      {
        return 4. / 15 * vtkMath::Pi() * pow(radius, 3 + p + q + r) / normalizationWeight(dimension, p, q, r);
      }
      else if (p + q + r == 4)
      {
        if (p == 4 || q == 4 || r == 4)
        {
          return 4. / 35 * vtkMath::Pi() * pow(radius, 3 + p + q + r) / normalizationWeight(dimension, p, q, r);
        }
        else
        {
          return 4. / 105 * vtkMath::Pi() * pow(radius, 3 + p + q + r) / normalizationWeight(dimension, p, q, r);
        }
      }
      else if (p + q + r == 6)
      {
        if (p == 6 || q == 6 || r == 6)
        {
          return 4. / 63 * vtkMath::Pi() * pow(radius, 3 + p + q + r) / normalizationWeight(dimension, p, q, r);
        }
        else if (p == 4 || q == 4 || r == 4)
        {
          return 4. / 315 * vtkMath::Pi() * pow(radius, 3 + p + q + r) / normalizationWeight(dimension, p, q, r);
        }
        else
        {
          return 4. / 945 * vtkMath::Pi() * pow(radius, 3 + p + q + r) / normalizationWeight(dimension, p, q, r);
        }
      }
      else
      {
        double center[3] = { 0.0, 0.0, 0.0 };
        vtkNew<vtkImageData> stencil;
        stencil->SetDimensions(25, 25, 25);
        stencil->SetSpacing(2. * radius / 25, 2. * radius / 25, 2. * radius / 25);
        stencil->SetOrigin(center);
        return (translationFactor(dimension, radius, p, q, r, stencil));
      }
    }
    else
    {
      return 0;
    }
  }
}

/**
 * This function generates the stencil, which contains the locations at which the dataset is
 * evaluated for the integration
 * @param stencil: contains the locations at which the dataset is evaluated for the integration
 * @param radius: the integration radius at which the moments are computed
 * @param numberOfIntegrationSteps: how fine the discrete integration done in each dimension
 * @param dimension: 2D or 3D
 * @param source: the dataset
 * @param nameOfPointData: the name of the array in the point data of which the momens are computed.
 * @return the moments
 */
void vtkMomentsHelper::BuildStencil(vtkImageData* stencil,
  double radius,
  int numberOfIntegrationSteps,
  int dimension,
  vtkDataSet* source,
  std::string nameOfPointData)
{
  double spacing = 2. * radius / numberOfIntegrationSteps;

  if (dimension == 2)
  {
    stencil->SetDimensions(numberOfIntegrationSteps, numberOfIntegrationSteps, 1);
    stencil->SetSpacing(spacing, spacing, 1);
  }
  else
  {
    stencil->SetDimensions(
      numberOfIntegrationSteps, numberOfIntegrationSteps, numberOfIntegrationSteps);
    stencil->SetSpacing(spacing, spacing, spacing);
  }

  // set the copy attribute to tell interpolatePoint, which array to use
  stencil->GetPointData()->CopyAllOff();
  int idx;
  vtkDataSetAttributes* sPD = source->GetPointData();
  sPD->GetArray(nameOfPointData.c_str(), idx);
  int attIdx = sPD->IsArrayAnAttribute(idx);
  if (attIdx >= 0)
  {
    stencil->GetPointData()->SetCopyAttribute(attIdx, 1);
  }
  stencil->GetPointData()->CopyFieldOn(nameOfPointData.c_str());

  stencil->GetPointData()->InterpolateAllocate(
    sPD, stencil->GetNumberOfPoints(), stencil->GetNumberOfPoints());

  double bounds[6];
  stencil->GetBounds(bounds);
  stencil->SetOrigin(
    -0.5 * (bounds[1] - bounds[0]), -0.5 * (bounds[3] - bounds[2]), -0.5 * (bounds[5] - bounds[4]));

//  std::ostream stream(std::cout.rdbuf());
//  std::cout<<"Stencil\n";
//  std::cout<<"StencilSpacing="<<stencil->GetSpacing()[0]<<", "<<stencil->GetSpacing()[1]<<", "<<stencil->GetSpacing()[2]<<"\n";
//  stencil->GetPointData()->PrintSelf(stream, vtkIndent(0));
//  std::cout<<"\n";
//  double x[3];
//  for( int ptId = 0; ptId < stencil->GetNumberOfPoints(); ++ptId )
//  {
//    stencil->GetPoint(ptId, x);
//    std::cout<<ptId<<" x="<<x[0]<<" "<<x[1]<<"\n";
//  }
}

/**
 * This function moves the stencil to the current location, where the integration is supposed o be
 * performed
 * @param center: the location
 * @param source: the dataset
 * @param stencil: contains the locations at which the dataset is evaluated for the integration
 * @param numberOfIntegrationSteps: how fine the discrete integration done in each dimension
 * @return 0 if the stencil lies completely outside the field
 */
bool vtkMomentsHelper::CenterStencil(double center[3], vtkDataSet* source, vtkImageData* stencil,
  int numberOfIntegrationSteps, std::string vtkNotUsed(nameOfPointData))
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
      center[1] - 0.5 * (bounds[3] - bounds[2]),
      center[2] - 0.5 * (bounds[5] - bounds[4]));
  }
  //        if( stencil->GetBounds()[0] < source->GetBounds()[0] || stencil->GetBounds()[2] <
  //        source->GetBounds()[2] || stencil->GetBounds()[4] < source->GetBounds()[4] ||
  //        stencil->GetBounds()[1] > source->GetBounds()[1] || stencil->GetBounds()[3] >
  //        source->GetBounds()[3] || stencil->GetBounds()[5] > source->GetBounds()[5] )
  //        {
  //            return( false );
  //        }

  // interpolation of the source data at the integration points
  int subId = 0;
  for (vtkIdType ptId = 0; ptId < stencil->GetNumberOfPoints(); ++ptId)
  {
    // find point coordinates
    double x[3];
    stencil->GetPoint(ptId, x);

    // find cell
    double pcoords[3];
    double* weights = new double[source->GetMaxCellSize()];
    vtkIdType cellId = source->FindCell(x, NULL, -1, 1, subId, pcoords, weights);
    vtkCell* cell;
    if (cellId >= 0)
    {
      cell = source->GetCell(cellId);
    }
    else
    {
      cell = 0;
    }
    if (cell)
    {
      // Interpolate the point data
      stencil->GetPointData()->InterpolatePoint(
        source->GetPointData(), ptId, cell->PointIds, weights);
    }
//      // this would make the boundary zero by force, but actually that happens anyway
//    else
//    {
//      return (false);
//    }
  }

  //  if( center[0] == 0 && center[1] == 0 )
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

  //  vtkNew<vtkPResampleWithDataSet> resample;
  ////  resample->SetController(this->Controller);
  //  resample->SetInputData(stencil);
  //  resample->SetSourceData(source);
  //  resample->Update();
  //
  //  vtkNew<vtkProbeFilter> resample;
  //  resample->SetInputData(stencil);
  //  resample->SetSourceData(source);
  //  resample->Update();
  //
  //  if(
  //  vtkImageData::SafeDownCast(resample->GetOutput())->GetPointData()->GetArray("vtkValidPointMask")->GetRange()[1]
  //  == 0 )
  //  {
  //    return( false );
  //  }
  //
  //  stencil->GetPointData()->RemoveArray(nameOfPointData.c_str());
  //  stencil->GetPointData()->AddArray(vtkImageData::SafeDownCast(resample->GetOutput())->GetPointData()->GetArray(nameOfPointData.c_str()));

  return (true);
}

/**
 * checks if the moment indices are in ascending order
 * used to reduce redundancy of symmetric tensors
 * @param indices: the index of this output field pointdata array
 * @param dimension: 2D or 3D
 * @param order: the maximal order up to which the moments are computed
 * @param fieldRank: 0 for scalar, 1 for vector and 2 for matrix
 * @return the moments
 */
bool vtkMomentsHelper::isOrdered(std::vector<int> indices, int fieldRank)
{
  for (int i = 1; i < static_cast<int>(indices.size()) - fieldRank; ++i)
  {
    if (indices.at(i-1) > indices.at(i))
    {
      return false;
    }
  }
  return true;
}

/**
 * Inverse function to getFieldIndexFromTensorIndices
 * The output contains a vector with the tensor indices that describe the basis function that
 * belongs to the given output array. they are sorted by increasing order and then by the index as
 * returned by vtkMomentsTensor.getIndices(i)
 * @param index: the index of this output field pointdata array
 * @param dimension: 2D or 3D
 * @param order: the maximal order up to which the moments are computed
 * @param fieldRank: 0 for scalar, 1 for vector and 2 for matrix
 * @return the moments
 */
std::vector<int> vtkMomentsHelper::getTensorIndicesFromFieldIndex(
  int index, int dimension, int order, int fieldRank)
{
  int i = index;
  int r = fieldRank;
  while (i >= 0 && r <= order + fieldRank)
  {
    i -= pow(dimension, r);
    ++r;
  }
  --r;
  i += pow(dimension, r);
  // std::cout<<" rawindexToI="<<index-i;
  // std::cout<<" remainingindexToI="<<i<<": ";

  vtkMomentsTensor dumvtkMomentsTensor = vtkMomentsTensor(dimension, r, fieldRank);
  //    for( int j = 0; j < dumvtkMomentsTensor.getIndices(i).size(); ++j )
  //    {
  //        cout<<dumvtkMomentsTensor.getIndices(i).at(j);
  //    }
  std::vector<int> indices = dumvtkMomentsTensor.getIndices(i);
  // std::cout<<"getTensorIndicesFromFieldIndex i="<<i<<" "<<"r="<<r<<" index="<<index<<" "<<
  // getFieldIndexFromTensorIndices(dumvtkMomentsTensor.getIndices( i ))<<"\n";

  return dumvtkMomentsTensor.getIndices(i);
}

/**
 * Inverse function to getFieldNameFromTensorIndices
 * @param name: the name of this output field pointdata array
 * @return vector with the tensor indices that describe the basis function that
 * belongs to the given output array. they are sorted by increasing order and then by the index as
 * returned by vtkMomentsTensor.getIndices(i)
 */
std::vector<int> getTensorIndicesFromFieldName(std::string name)
{
//  std::cout<<"name="<<name <<"end \n";
  std::vector<int> indices;
  for (int i = name.find("x") + 1; i < name.length(); ++i)
  {
//    std::cout<<i<<" :"<<name[i]<<"\n";
    indices.push_back(static_cast<int>(name[i]));
  }
  return indices;
}

/**
 * Inverse function to getTensorIndicesFromFieldName
 * given a vector with tensor indices and a radius, this function returns the name of in the output
 * of this algorithm that corresponds tothis basis function
 * @param radius: radius in the radii vector
 * @param indices: the given tensor indices
 * @return the name of the array
 */
std::string vtkMomentsHelper::getFieldNameFromTensorIndices(double radius, std::vector<int> indices, int fieldRank)
{
  std::sort(indices.begin(), indices.begin() + indices.size() - fieldRank);
  std::string fieldName = "radius" + std::to_string(radius) + "index";
  for (int i = 0; i < static_cast<int>(indices.size()); ++i)
  {
    fieldName += std::to_string(indices.at(i));
  }
  return fieldName;
}

/**
 * Inverse function to getTensorIndicesFromFieldIndex
 * given a vectro with tensor indices and a radius, this function returns the index in the output of
 * this algorithm that corresponds to this basis function
 * @param radiusIndex: index of this radius in the radii vector
 * @param indices: the given tensor indices
 * @param dimension: 2D or 3D
 * @param fieldRank: 0 for scalar, 1 for vector and 2 for matrix
 * @param numberOfBasisFunctions: number of basis functions in the source
 * equals \sum_{i=0}^order dimension^o
 */
int vtkMomentsHelper::getFieldIndexFromTensorIndices(int radiusIndex, std::vector<int> indices,
  int dimension, int fieldRank, int numberOfBasisFunctions)
{
  // cout<<"\n getFieldIndexFromTensorIndices indices=";
  //    for( int j = 0; j < static_cast<int>(indices.size()); ++j )
  //    {
  //        cout<<indices.at(j);
  //    }
  vtkMomentsTensor dumvtkMomentsTensor =
    vtkMomentsTensor(dimension, static_cast<int>(indices.size()), fieldRank);
  int index = 0;
  for (int i = fieldRank; i < static_cast<int>(indices.size()); ++i)
  {
    index += pow(dimension, i);
  }
  // std::cout<<" rawindexFromI="<<index;
  // std::cout<<" remainingindexFromI="<<dumvtkMomentsTensor.getIndex( indices )<<": ";
  //    for( int j = 0; j < static_cast<int>(indices.size()); ++j )
  //    {
  //        cout<<indices.at(j);
  //    }
  return index + dumvtkMomentsTensor.getIndex(indices) + radiusIndex * numberOfBasisFunctions;
}

/**
 * The output contains the tensor indices that describe the basis function that belongs to the given
 * output array as string. Convenience function. they are sorted by increasing order and then by the
 * index as returned by vtkMomentsTensor.getIndices(i)
 * @param index: the index of this output field pointdata array
 * @param dimension: 2D or 3D
 * @param order: the maximal order up to which the moments are computed
 * @param fieldRank: 0 for scalar, 1 for vector and 2 for matrix
 * @return the moments
 */
std::string vtkMomentsHelper::getTensorIndicesFromFieldIndexAsString(
  int index, int dimension, int order, int fieldRank)
{
  std::string indexString = "";
  std::vector<int> indices =
    vtkMomentsHelper::getTensorIndicesFromFieldIndex(index, dimension, order, fieldRank);
  for (int i = 0; i < static_cast<int>(indices.size()); ++i)
  {
    indexString += std::to_string(indices.at(i));
  }
  return indexString;
}

/**
 * This Method checks if the ball with radius around the current point exceeds the global boundary.
 * @param center: the current center of the convolution
 * @param radius: the current radius
 * @param boundary: the (global) boundary of the field
 * @param dimension: 2 or 3
 * @return true if it lies ouside the boundary
 */
bool vtkMomentsHelper::IsCloseToBoundary(double center[3], double radius, double boundary[6], int dimension)
{
  for (int d = 0; d < dimension; ++d)
  {
    if (center[d] - boundary[2*d] < radius * (1 - 1e-10)
        || boundary[2*d+1] - center[d] < radius * (1 - 1e-10))
    {
      return 1;
    }
  }
  return 0;
}

///**
// * the function returns true if the point lies within radius of the boundary of the dataset
// * @param ptId: ID of the point in question
// * @param field: the field that contains the point
// */
//bool vtkMomentsHelper::isCloseToEdge(int dimension, int ptId, double radius, vtkImageData* field)
//{
//  if (dimension == 2)
//  {
//    return !(ptId % field->GetDimensions()[0] >= radius / field->GetSpacing()[0] &&
//      ptId % field->GetDimensions()[0] <=
//        field->GetDimensions()[0] - 1 - radius / field->GetSpacing()[0] &&
//      ptId / field->GetDimensions()[0] >= radius / field->GetSpacing()[1] &&
//      ptId / field->GetDimensions()[0] <=
//        field->GetDimensions()[1] - 1 - radius / field->GetSpacing()[1]);
//  }
//  else
//  {
//    return !(ptId % field->GetDimensions()[0] >= radius / field->GetSpacing()[0] &&
//      ptId % field->GetDimensions()[0] <=
//        field->GetDimensions()[0] - 1 - radius / field->GetSpacing()[0] &&
//      (ptId / field->GetDimensions()[0]) % field->GetDimensions()[1] >=
//        radius / field->GetSpacing()[1] &&
//      (ptId / field->GetDimensions()[0]) % field->GetDimensions()[1] <=
//        field->GetDimensions()[1] - 1 - radius / field->GetSpacing()[1] &&
//      ptId / field->GetDimensions()[0] / field->GetDimensions()[1] >=
//        radius / field->GetSpacing()[2] &&
//      ptId / field->GetDimensions()[0] / field->GetDimensions()[1] <=
//        field->GetDimensions()[2] - 1 - radius / field->GetSpacing()[2]);
//  }
//}
//
bool vtkMomentsHelper::isEdge(int dimension, int ptId, vtkImageData* field)
{
  if (dimension == 2)
  {
    //        cout << ptId << " " << field->GetPoint(ptId)[0] << " " << field->GetPoint(ptId)[1] <<
    //        " " << !(ptId % field->GetDimensions()[0] > 0
    //        && ptId % field->GetDimensions()[0] < field->GetDimensions()[0]-1
    //        && ptId / field->GetDimensions()[0] > 0
    //        && ptId / field->GetDimensions()[0] < field->GetDimensions()[1]-1) << endl;
    return !(ptId % field->GetDimensions()[0] > 0 &&
      ptId % field->GetDimensions()[0] < field->GetDimensions()[0] - 1 &&
      ptId / field->GetDimensions()[0] > 0 &&
      ptId / field->GetDimensions()[0] < field->GetDimensions()[1] - 1);
  }
  else
  {
    //        cout << ptId << " " << field->GetPoint(ptId)[0] << " " << field->GetPoint(ptId)[1] <<
    //        " " << field->GetPoint(ptId)[2] << " " <<
    //        !(ptId % field->GetDimensions()[0] > 0
    //          && ptId % field->GetDimensions()[0] < field->GetDimensions()[0]-1
    //          && (ptId / field->GetDimensions()[0]) % field->GetDimensions()[1] > 0
    //          && (ptId / field->GetDimensions()[0]) % field->GetDimensions()[1]  <
    //          field->GetDimensions()[1]-1
    //          && ptId / field->GetDimensions()[0] / field->GetDimensions()[1] > 0
    //          && ptId / field->GetDimensions()[0] / field->GetDimensions()[1]  <
    //          field->GetDimensions()[2]-1)
    //        << endl;
    return !(ptId % field->GetDimensions()[0] > 0 &&
      ptId % field->GetDimensions()[0] < field->GetDimensions()[0] - 1 &&
      (ptId / field->GetDimensions()[0]) % field->GetDimensions()[1] > 0 &&
      (ptId / field->GetDimensions()[0]) % field->GetDimensions()[1] <
        field->GetDimensions()[1] - 1 &&
      ptId / field->GetDimensions()[0] / field->GetDimensions()[1] > 0 &&
      ptId / field->GetDimensions()[0] / field->GetDimensions()[1] < field->GetDimensions()[2] - 1);
  }
}

//----------------------------------------------------------------------------------
vtkIdType vtkMomentsHelper::getArrayIndex(std::vector<int> coord, std::vector<int> dimensions)
{
  return coord[0] + coord[1] * dimensions[0] + coord[2] * dimensions[0] * dimensions[1];
}

//----------------------------------------------------------------------------------
std::vector<int> vtkMomentsHelper::getCoord(vtkIdType index, std::vector<int> dimensions)
{
  int z = 0;
  if (dimensions[2] > 1)
  {
    z = index / dimensions[0] / dimensions[1];
  }

  int y = (index - z * dimensions[0] * dimensions[1]) / dimensions[0];

  std::vector<int> arr(3);
  arr[0] = index - z * dimensions[0] * dimensions[1] - y * dimensions[0];
  arr[1] = y;
  arr[2] = z;

  return arr;
}

//--------------------------------------------------------------------
vtkSmartPointer<vtkImageData> vtkMomentsHelper::translateToOrigin(vtkImageData* data)
{
  // Translate to the origin
  vtkNew<vtkImageTranslateExtent> trans;
  trans->SetTranslation(-data->GetExtent()[0], -data->GetExtent()[2], -data->GetExtent()[4]);
  trans->SetInputData(data);
  trans->Update();
  return trans->GetOutput();
}

//------------------------------------------------------------------------------------------
vtkSmartPointer<vtkImageData> vtkMomentsHelper::padField(vtkImageData* field,
  vtkImageData* kernel,
  int dimension,
  std::string nameOfPointData)
{
  // Translate to the origin
  vtkSmartPointer<vtkImageData> transR = translateToOrigin(field);
  vtkSmartPointer<vtkImageData> transK = translateToOrigin(kernel);

  int dataMinExtent =
    std::min(std::min(transR->GetExtent()[0], transR->GetExtent()[2]), transR->GetExtent()[4]);
  int dataMaxExtent =
    std::max(std::max(transR->GetExtent()[1], transR->GetExtent()[3]), transR->GetExtent()[5]);
  int patternMinExtent =
    std::min(std::min(transK->GetExtent()[0], transK->GetExtent()[2]), transK->GetExtent()[4]);
  int patternMaxExtent =
    std::max(std::max(transK->GetExtent()[1], transK->GetExtent()[3]), transK->GetExtent()[5]);
  int minExtent = dataMinExtent - patternMinExtent;
  int maxExtent = dataMaxExtent + patternMaxExtent;

  int dataExtentPad[6] = { 0, 0, 0, 0, 0, 0 };
  for (int i = 0; i < dimension; i++)
  {
    dataExtentPad[2 * i] = minExtent;
    dataExtentPad[2 * i + 1] = maxExtent;
  }

  vtkNew<vtkImageData> output;
  output->SetOrigin(0, 0, 0);
  output->SetSpacing(transR->GetSpacing());
  output->SetExtent(dataExtentPad);

  vtkDataArray* origArray = transR->GetPointData()->GetArray(nameOfPointData.c_str());

  vtkNew<vtkDoubleArray> paddedArray;
  paddedArray->SetName(nameOfPointData.c_str());
  paddedArray->SetNumberOfComponents(origArray->GetNumberOfComponents());
  paddedArray->SetNumberOfTuples(output->GetNumberOfPoints());
  paddedArray->Fill(0.0);

  const int* tmp = transR->GetDimensions();
  std::vector<int> origSize = std::vector<int>(tmp, tmp + 3);

  tmp = output->GetDimensions();
  std::vector<int> paddedSize = std::vector<int>(tmp, tmp + 3);

  /* KissFFT Implementation for padding field */
  for (vtkIdType i = 0; i < transR->GetNumberOfPoints(); i++)
  {
    std::vector<int> coord = getCoord(i, origSize);
    for (int j = 0; j < static_cast<int>(coord.size()); j++)
    {
      coord[j] += transK->GetDimensions()[j] / 2;
    }

    vtkIdType index = getArrayIndex(coord, paddedSize);
    paddedArray->SetTuple(index, origArray->GetTuple(i));
  }

  output->GetPointData()->AddArray(paddedArray);

  /* Using Constant Pad Filter for padding 0's */
  // vtkNew<vtkImageConstantPad> paddedField;
  // paddedField->SetOutputWholeExtent(dataExtentPad);
  // paddedField->SetInputData(transR);
  // paddedField->SetConstant(0.0);
  // paddedField->Update();
  // vtkImageData* paddedOutput = paddedField->GetOutput();

  /* Writing to Image .vti file */
  // vtkNew<vtkXMLImageDataWriter> writer;
  // writer->SetInputData(paddedOutput);
  // writer->SetFileName("/Users/ktsai/Documents/VTK_MomentInvariants/momentPatternDetetctionTest/output/paddedField.vti");
  // writer->Write();

  return output.GetPointer();
}

//--------------------------------------------------------------------------------------------
vtkSmartPointer<vtkImageData> vtkMomentsHelper::padKernel(vtkImageData* kernel,
  vtkImageData* paddedField)
{
  // Translate to the origin
  vtkSmartPointer<vtkImageData> trans = translateToOrigin(kernel);

  vtkNew<vtkImageData> output;
  output->SetOrigin(0, 0, 0);
  output->SetSpacing(trans->GetSpacing());
  output->SetExtent(paddedField->GetExtent());

  vtkDataArray* scalars = trans->GetPointData()->GetScalars();

  vtkNew<vtkDoubleArray> scalarsPad;
  scalarsPad->SetName("kernel");
  scalarsPad->SetNumberOfComponents(1);
  scalarsPad->SetNumberOfTuples(output->GetNumberOfPoints());
  scalarsPad->Fill(0.0);

  const int* tmp = trans->GetDimensions();
  std::vector<int> origSize = std::vector<int>(tmp, tmp + 3);

  tmp = output->GetDimensions();
  std::vector<int> paddedSize = std::vector<int>(tmp, tmp + 3);

  /* KissFFT Implementation for padding kernel */
  for (vtkIdType i = 0; i < scalars->GetNumberOfTuples(); i++)
  {
    std::vector<int> coord = getCoord(i, origSize);
    vtkIdType index = getArrayIndex(coord, paddedSize);
    scalarsPad->SetTuple1(index, scalars->GetTuple1(i));
  }

  output->GetPointData()->SetScalars(scalarsPad);
  return output.GetPointer();
}

/**
 * get weighs that is used for nomalizing moments
 * @param dimension e.g. 2D or 3D
 * @param i0 index0
 * @param i1 index1
 * @param i2 index2
 * @return weight 
 */
double vtkMomentsHelper::normalizationWeight(unsigned dimension, unsigned i0, unsigned i1, unsigned i2) {
  if (dimension == 2) {
    static const double weights[5][5] = {1, 1 / 4.0, 1 / 8.0, 5 / 64.0, 7 / 128.0,
					 1 / 4.0, 1 / 24.0, 1 / 64.0, 1 / 128.0, 7 / 1536.0,
					 1 / 8.0, 1 / 64.0, 3 / 640.0, 1 / 512.0, 1 / 1024.0,
					 5 / 64.0, 1 / 128.0, 1 / 512.0, 5 / 7168.0, 5 / 16384.0,
					 7 / 128.0, 7 / 1536.0, 1 / 1024.0, 5 / 16384.0, 35 / 294912.0};
    
    return weights[i1][i0] * M_PI; 
  }
  else if (dimension == 3) {
    static const double weights[5][5][5] = {4 / 3.0, 4 / 15.0, 4 / 35.0, 4 / 63.0, 4 / 99.0,
					    4 / 15.0, 4 / 105.0, 4 / 315.0, 4 / 693.0, 4 / 1287.0,
					    4 / 35.0, 4 / 315.0, 4 / 1155.0, 4 / 3003.0, 4 / 6435.0,
					    4/63.0, 4/693.0, 4/3003.0, 4/9009.0, 4/21879.0, 
					    4/99.0, 4/1287.0, 4/6435.0, 4/21879.0, 28/415701.0, 
					    4/15.0, 4/105.0, 4/315.0, 4/693.0, 4/1287.0, 
					    4/105.0, 4/945.0, 4/3465.0, 4/9009.0, 4/19305.0, 
					    4/315.0, 4/3465.0, 4/15015.0, 4/45045.0, 4/109395.0, 
					    4/693.0, 4/9009.0, 4/45045.0, 4/153153.0, 4/415701.0, 
					    4/1287.0, 4/19305.0, 4/109395.0, 4/415701.0, 4/1247103.0, 
					    4/35.0, 4/315.0, 4/1155.0, 4/3003.0, 4/6435.0, 
					    4/315.0, 4/3465.0, 4/15015.0, 4/45045.0, 4/109395.0, 
					    4/1155.0, 4/15015.0, 4/75075.0, 4/255255.0, 4/692835.0, 
					    4/3003.0, 4/45045.0, 4/255255.0, 4/969969.0, 4/2909907.0, 
					    4/6435.0, 4/109395.0, 4/692835.0, 4/2909907.0, 4/9561123.0, 
					    4/63.0, 4/693.0, 4/3003.0, 4/9009.0, 4/21879.0, 
					    4/693.0, 4/9009.0, 4/45045.0, 4/153153.0, 4/415701.0, 
					    4/3003.0, 4/45045.0, 4/255255.0, 4/969969.0, 4/2909907.0, 
					    4/9009.0, 4/153153.0, 4/969969.0, 20/20369349.0, 20/66927861.0, 
					    4/21879.0, 4/415701.0, 4/2909907.0, 20/66927861.0, 4/47805615.0, 
					    4/99.0, 4/1287.0, 4/6435.0, 4/21879.0, 28/415701.0, 
					    4/1287.0, 4/19305.0, 4/109395.0, 4/415701.0, 4/1247103.0, 
					    4/6435.0, 4/109395.0, 4/692835.0, 4/2909907.0, 4/9561123.0, 
					    4/21879.0, 4/415701.0, 4/2909907.0, 20/66927861.0, 4/47805615.0, 
					    28/415701.0, 4/1247103.0, 4/9561123.0, 4/47805615.0, 28/1290751605.0};

    return weights[i2][i1][i0] * M_PI;
  }
  else {
    cerr << "vtkMomentsHelper::normalizationWeight: dimension > 3.\n";
    exit(0);
  }
}

/**
 * This function normalizes a set of moments in the vtkMomentsTensor class
 * @param moments a set of moments
 * @return normalized moments
 */
std::vector<vtkMomentsTensor> & vtkMomentsHelper::normalizeMoments(std::vector<vtkMomentsTensor> & moments) {
  unsigned order = static_cast<int>(moments.size());
  unsigned dimension = moments.front().getDimension();
  

  for (unsigned o = 0; o < order; o++) 
    for (unsigned i = 0; i < moments[o].size(); i++) {
      std::vector<int> midx = moments[o].getMomentIndices(i);
      std::vector<unsigned> idx(dimension + 1,0);
      for (unsigned j = 0; j < midx.size();) 
	idx[midx[j++]]++;

      unsigned i2;
      if (dimension == 3)
	i2 = idx[2];

      moments[o].set(i, moments[o].get(i) / normalizationWeight(dimension, idx[0], idx[1], i2));

    }
  
  return moments;
}
