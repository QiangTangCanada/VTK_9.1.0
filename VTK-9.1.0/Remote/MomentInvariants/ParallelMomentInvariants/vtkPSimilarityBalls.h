/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkPSimilarityBalls.h

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
 * @class   vtkPSimilarityBalls
 * @brief   rotation invariant pattern detetction
 *
 * vtkPSimilarityBalls is a the distributed memory version of vtkSimilarityBalls.
 * Look there for more information.
 * @par Thanks:
 * Developed by Roxana Bujack and Karen Tsai at Los Alamos National Laboratory.
 */

#ifndef vtkPSimilarityBalls_h
#define vtkPSimilarityBalls_h

#include "vtkSimilarityBalls.h"
#include "ParallelMomentInvariantsModule.h" // For export macro

#include "vtkMPI.h" // For MPI support

class vtkMultiProcessController;

class PARALLELMOMENTINVARIANTS_EXPORT vtkPSimilarityBalls : public vtkSimilarityBalls
{
public:
  static vtkPSimilarityBalls* New();
  vtkTypeMacro(vtkPSimilarityBalls, vtkSimilarityBalls);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Set/Get the multiprocess controller object.
   */
  void SetController(vtkMultiProcessController* c);
  vtkGetObjectMacro(Controller, vtkMultiProcessController);

  /**
   * Set/Get the MPI communicator.
   */
  vtkSetMacro(MPI_Communicator, MPI_Comm*);
  vtkGetMacro(MPI_Communicator, MPI_Comm*);

protected:
  vtkPSimilarityBalls();
  ~vtkPSimilarityBalls() override;

  int RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  /**
   * main executive of the program, reads the input, calles the
   * functions, and produces the utput.
   * @param inputVector: the input information
   * @param outputVector: the output information
   */
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
  /**
   * Multiprocess controller object.
   */
  vtkMultiProcessController* Controller;

  /**
   * MPI multiprocess communicator object.
   */
  MPI_Comm *MPI_Communicator;

  void CheckValidity(vtkImageData* similarityData, vtkImageData* gridData);

  /** extraction of the local maxima of the similaity field. this avoids clutter in the
   * visualization.
   * @param similarityData: the output of this algorithm. it has the topology of moments and will
   * have a number of scalar fields euqal to NumberOfRadii. each point contains the similarity of
   * its surrounding (of size radius) to the pattern
   * @param localMaxData: contains the similariy value and the corresponding radius if the
   * similarity field had a local maximum in space plus scale at the given point. It also stored the
   * radius that caused the maximum
   */
  virtual void LocalMaxSimilarity(vtkImageData* similarityData, vtkImageData* localMaxData) override;

  /**
   * This method draws a sphere (full and hollow) around the local maxima.
   * @param localMaxData: contains the similarity value and the corresponding radius if the similarity
   * field at a local maximum in space plus scale at the given point.
   * @param gridData: the grid for the balls
   * @param ballsData: a solid ball drawn around the local maxima
   * @param spheresData: an empty sphere drawn around the local maxima
   */
  virtual void Balls(vtkImageData* localMaxData,
             vtkImageData* gridData,
             vtkImageData* ballsData,
             vtkImageData* spheresData) override;

  vtkPSimilarityBalls(const vtkPSimilarityBalls&) = delete;
  void operator=(const vtkPSimilarityBalls&) = delete;

};

#endif
