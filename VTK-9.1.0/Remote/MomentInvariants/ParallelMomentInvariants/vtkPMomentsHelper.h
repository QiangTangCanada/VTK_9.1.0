/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkPMomentsHelper.h

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
/**
 * @class   vtkPMomentsHelper
 * @brief   rotation invariant pattern detetction
 *
 * vtkPMomentsHelper is a helper class that contains functions that will
 * be used by more than one algorithm in the parallel moments module the theory and
 * the algorithm is described in Roxana Bujack and Hans Hagen: "Moment
 * Invariants for Multi-Dimensional Data"
 * http://www.informatik.uni-leipzig.de/~bujack/2017TensorDagstuhl.pdf
 * @par Thanks:
 * Developed by Karen Tsai at Los Alamos National Laboratory.
 */

#ifndef vtkPMomentsHelper_h
#define vtkPMomentsHelper_h
#ifndef __VTK_WRAP__

#include "ParallelMomentInvariantsModule.h" // For export macro

#include "vtkMultiProcessController.h"
#include "vtkSmartPointer.h" // for vtkSmartPointer.
#include "vtkType.h"         // for vtkIdType
#include <string>            // for std::string
#include <vector>            // for std::vector

class vtkImageData;
class vtkMomentsTensor;

struct PARALLELMOMENTINVARIANTS_EXPORT vtkPMomentsHelper
{
  /**
   * Pad the field in parallel to the specification of the fft library
   * DFFTLIB: The two outer edges that touch origin needs to be padded with 0's
   *          where the size is half the size of the kernel
   *          The other two edges is padded with >= 1 layers of 0's
   *          to achieve a power of 2 size for the final data array
   */
  static vtkSmartPointer<vtkImageData> padPField(
    vtkImageData* field,
    int dimension,
    std::vector<int> fieldGlobalExtent,
    std::vector<int> kernelExtent,
    std::vector<int>& paddedFieldGlobalExtent,
    std::string nameOfPointData,
    vtkMultiProcessController* controller);

  /**
   * Get all local bounds in an array of size procId * 6 + boundIndex.
   * The last row contains the global bounds
   * @param allBounds the return array
   * @param field: the dataset of which the bounds are computed
   * @param controller: the mpi controler
   */
  static void GetAllBounds(double allBounds[], vtkImageData* field, vtkMultiProcessController* controller);

  /**
   * Get all local bounds in an array of size procId * 6 + boundIndex.
   * The last row contains the global bounds
   * but if the field has ghost cells, we only return the bounds reduced by spacing
   * @param allBounds the return array
   * @param field: the dataset of which the bounds are computed
   * @param controller: the mpi controler
   */
  static void GetAllBoundsWithoutGhostCells(double allBounds[], vtkImageData* field, vtkMultiProcessController* controller);

  /**
   * Get all local bounds in an array of size procId * 6 + boundIndex.
   * The last row contains the global bounds
   * @param allBounds the return array
   * @param field: the dataset of which the bounds are computed
   * @param controller: the mpi controler
   * @return this vector contains a vector for each proc. with the centers that are close to that proc's boundary
   */
  static std::vector<std::vector<std::vector<double> > > GetBoundaryCenters(double allBounds[], vtkImageData* field, int dimension, double radius, vtkMultiProcessController* controller);

  static std::vector<std::vector<std::vector<double> > > GetBoundaryInformation(double allBounds[], vtkImageData* field, int dimension, double radius, vtkMultiProcessController* controller);

  static std::vector<std::vector<std::vector<double> > > ExchangeBoundaryInformation(int numberOfInformation, std::vector<std::vector<std::vector<double> > > myBoundaryCenters, int dimension, vtkMultiProcessController* controller);
};

#endif // __VTK_WRAP__
#endif
// VTK-HeaderTest-Exclude: vtkPMomentsHelper.h
