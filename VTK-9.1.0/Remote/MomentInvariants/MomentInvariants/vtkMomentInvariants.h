/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkMomentInvariants.h

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
 * @class   vtkMomentInvariants
 * @brief   rotation invariant pattern detetction
 *
 * vtkMomentInvariants is a filter that performs pattern detection
 * it is able to determine the similarity independent from the orientation
 * of the template it takes the moments (momentData) as produced by
 * computeMoments and a pattern as inputs
 * 0. it produces a scalar field as output. Each point in this field contains
 * the similarity between the pattern and that point. it also puts out the
 * following things
 * 1. the normalized moments of the field
 * 2. the moments of the pattern
 * 1. the normalized moments of the pattern
 * the theory and the algorithm is described in Roxana Bujack and Hans Hagen:
 * "Moment Invariants for Multi-Dimensional Data"
 * http://www.informatik.uni-leipzig.de/~bujack/2017TensorDagstuhl.pdf
 * @par Thanks:
 * Developed by Roxana Bujack at Los Alamos National Laboratory.
 */

#ifndef vtkMomentInvariants_h
#define vtkMomentInvariants_h

#include "vtkDataSetAlgorithm.h"
#include "vtkDataSetAttributes.h"             // needed for vtkDataSetAttributes::FieldList
#include "MomentInvariantsModule.h" // For export macro
#include "vtkTuple.h"                         // For internal API
#include "vtkMomentsPolynomial.h"
#include <vector>                             // For internal API
#include <list>

class vtkImageData;
class vtkMomentsTensor;

class MOMENTINVARIANTS_EXPORT vtkMomentInvariants : public vtkDataSetAlgorithm
{
public:
  static vtkMomentInvariants* New();

  vtkTypeMacro(vtkMomentInvariants, vtkDataSetAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;
  /**
   * Input ports,
   * port 0 is the output of compute moments (which can be temporal),
   * port 1 is the patern. Temporal data needs to be on port 0.
   */
  const int PatternPort = 1;
  const int MomentPort = 0;

  /**
   * standard pipeline input for port 1
   * This is the pattern, a vtkDataSet of scalar, vector, or matrix type.
   */
  void SetPatternData(vtkDataObject* input)
  {
    this->SetInputData(PatternPort, input);
  }

  /**
   * standard pipeline input for port 1
   * This is the pattern, a vtkDataSet of scalar, vector, or matrix type.
   */
  void SetPatternConnection(vtkAlgorithmOutput* algOutput)
  {
    this->SetInputConnection(PatternPort, algOutput);
  };

  /**
   * standard pipeline input for port 1
   * This is the vtkImageData field of which the moments are calculated. The
   * locations at which the moments are calculated are the points of the grid
   * in input.
   */
  void SetMomentData(vtkDataObject* input)
  {
    this->SetInputData(MomentPort, input);
  }

  /**
   * standard pipeline input for port 1
   * This is the vtkImageData field of which the moments are calculated. The
   * locations at which the moments are calculated are the points of the grid
   * in input.
   */
  void SetMomentConnection(vtkAlgorithmOutput* algOutput)
  {
    this->SetInputConnection(MomentPort, algOutput);
  };

  //@{
  /**
   * Set/Get the resolution of the integration.
   */
  vtkSetMacro(NumberOfIntegrationSteps, int);
  vtkGetMacro(NumberOfIntegrationSteps, int);
  //@}

  //@{
  /**
   * Set/Get the flag of using the original resolution of the field for the integration.
   */
  vtkSetMacro(UseOriginalResolution, bool);
  vtkGetMacro(UseOriginalResolution, bool);
  //@}

  //@{
  /**
   * Set/Get the resolution of the integration.
   */
  vtkSetMacro(AngleResolution, int);
  vtkGetMacro(AngleResolution, int);
  //@}

  //@{
  /**
   * Set/Get the resolution of the integration.
   */
  vtkSetMacro(Eps, double);
  vtkGetMacro(Eps, double);
  //@}

  //@{
  /**
   * Set/Get the index of the array in the point data of which the momens are computed.
   */
  vtkSetMacro(NameOfPointData, std::string);
  vtkGetMacro(NameOfPointData, std::string);
  //@}

  //@{
  /**
   * Set/Get if the user wants invariance w.r.t. outer translation
   * that is addition of a constant
   */
  vtkSetMacro(IsTranslation, bool);
  vtkGetMacro(IsTranslation, bool);
  //@}

  //@{
  /**
   * Set/Get if the user wants invariance w.r.t. outer translation
   * that is addition of a constant
   */
  vtkSetMacro(IsScaling, bool);
  vtkGetMacro(IsScaling, bool);
  //@}

  //@{
  /**
   * Set/Get if the user wants invariance w.r.t. total rotation
   */
  vtkSetMacro(IsRotation, bool);
  vtkGetMacro(IsRotation, bool);
  //@}

  //@{
  /**
   * Set/Get if the user wants invariance w.r.t. total reflection
   */
  vtkSetMacro(IsReflection, bool);
  vtkGetMacro(IsReflection, bool);
  //@}

  /**
   * Get the number of basis functions in the momentData
   * equals \sum_{i=0}^order dimension^o
   */
  vtkGetMacro(NumberOfBasisFunctions, int);

  /**
   * Get the different integration radii from the momentData
   */
  std::vector<double> GetRadii() { return this->Radii; }

  /**
   * Get the different integration radii from the momentData as constant length array for python
   * wrapping
   */
  void GetRadiiArray(double radiiArray[10])
  {
    for (int i = 0; i < 10; ++i)
    {
      radiiArray[i] = 0;
    }
    for (int i = 0; i < static_cast<int>(this->Radii.size()); ++i)
    {
      radiiArray[i] = this->Radii.at(i);
    }
  };

  /**
   * Get the number of the different integration radii from the momentData
   */
  int GetNumberOfRadii() { return static_cast<int>(this->Radii.size()); };

  /**
   * Get the different integration radii from the momentData as string. Convenience function.
   */
  std::string GetStringRadii(int i) { return std::to_string(this->Radii.at(i)).c_str(); };

  /**
   * Get the translation factor
   */
  double GetTranslationFactor(int radius, int p, int q, int r)
  {
    return this->TranslationFactor[radius + p * this->Radii.size() +
      q * this->Radii.size() * (this->Order + 1) +
      r * this->Radii.size() * (this->Order + 1) * (this->Order + 1)];
  };

  /**
   * Get the translation factor
   */
  void SetTranslationFactor(int radius, int p, int q, int r, double value)
  {
    this->TranslationFactor[radius + p * this->Radii.size() +
      q * this->Radii.size() * (this->Order + 1) +
      r * this->Radii.size() * (this->Order + 1) * (this->Order + 1)] = value;
  };

  inline void SetInvariantMethod(std::string name)
  {
    invariantMethod = name;
  }

protected:
  /**
   * constructior setting defaults
   */
  vtkMomentInvariants();

  /**
   * destructor
   */
  ~vtkMomentInvariants() override;

  int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  int RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  /**
   * main executive of the program, reads the input, calles the
   * functions, and produces the utput.
   * @param inputVector: the input information
   * @param outputVector: the output information
   */
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

private:
  vtkMomentInvariants(const vtkMomentInvariants&) = delete;
  void operator=(const vtkMomentInvariants&) = delete;

  /**
   * Integration radius of the pattern
   */
  double RadiusPattern;

  /**
   * center of the pattern
   */
  double CenterPattern[3];

  /**
   * Dimension 
can be 2 or 3
   */
  int Dimension;

  /**
   * Rank of the momentData is 0 for scalars, 1 for vectors, 3 for matrices
   */
  int FieldRank;

  /**
   * Maximal order up to which the moments are calculated
   */
  unsigned int Order;

  /**
   * Integration radius
   */
  std::vector<double> Radii;

  /**
   * How fine is the discrete integration done?
   */
  int NumberOfIntegrationSteps;

  /**
   * Flag for using the original resolution of the input dataset
   */
  bool UseOriginalResolution;

  /**
   * If the pattern has multiple fields in its point data, you can choose the one that the moments
   * shall be calculated of
   */
  std::string NameOfPointData;

  /**
   * the number of fields in the momentData
   * equals NumberOfBasisFunctions * numberOfRadii
   */
  int NumberOfFields;

  /**
   * the number of different fields is the actual number
   * equals NumberOfDifferentBasisFunctions * numberOfRadii
   */
  int NumberOfDifferentFields;

  /**
   * the number of basis functions in the momentData
   * equals \sum_{i=0}^order dimension^o
   */
  int NumberOfBasisFunctions;

  /**
   * the number of basis functions in the field
   * for 2D equals \sum_{o=0}^order (o+1)
   * for 3D equals \sum_{o=0}^order 1./2*(o+1)*(o+2)
   */
  int NumberOfDifferentBasisFunctions;

  /**
   * if the user wants invariance w.r.t. outer translation
   * that is addition of a constant
   */
  bool IsTranslation;

  /**
   * if the user wants invariance w.r.t. outer scaling
   * that is multiplication of a constant
   */
  bool IsScaling;

  /**
   * if the user wants invariance w.r.t. total rotation
   */
  bool IsRotation;

  /**
   * if the user wants invariance w.r.t. total reflection
   */
  bool IsReflection;

  /**
   * if the algorithm is not able to find dominant contractions for the normalization w.r.t.
   * rotation, it has to default back to looking "everywhere". This parameter determines how fine
   * this is performed in 2D, we divide phi=[0,...,2Pi] into that many equidistant steps in 3D, we
   * divide phi=[0,...,2Pi] into that many equidistant steps and theta=[0,...,Pi] in half that many
   * steps to determine the rotation axis. Then, we use anther AngleResolution different rotation
   * angles in [0,...,2Pi] to cover all positions the number of comparisons is AngleResolution in 2D
   * and 0.5 * AngleResolution^3 in 3D
   */
  int AngleResolution;

  /**
   * this parameter determines if the algorithm is not able to find dominant contractions for the
   * normalization w.r.t. rotation. Non-zero means > eps
   */
  double Eps;

  /**
   * this contains the moments of the pattern
   */
  std::vector<vtkMomentsTensor> MomentsPattern;

  /**
   * this contains the moments of the pattern after normalization w.r.t. translation. c00 should be
   * 0 now
   */
  std::vector<vtkMomentsTensor> MomentsPatternTNormal;

  /**
   * this contains the moments of the pattern after normalization w.r.t. scaling. the norm should be
   * 1 now
   */
  std::vector<vtkMomentsTensor> MomentsPatternTSNormal;

  /**
   * this contains the moments of the pattern after normalization w.r.t. rotation. several standard
   * positions can occur. it contains all orientations of the moments of the pattern. during the
   * detection later, we will compare the moments of the field to all these version of the pattern
   */
  std::vector<std::vector<vtkMomentsTensor> > MomentsPatternNormal;

  /**
   * this contains the translational factors necessary for normalization w.r.t. translation
   * we have radius and then p,q,r
   */
  double* TranslationFactor;

  /**
   * The invariant mothod
   */
  std::string invariantMethod;

  /**
   * the threshold for the LU decomposition, which is used to compute the rank of a maxtrix.
   */
  double luThreshold;

  /**
   * the minimum nonzero order for computing the mixed invariants in the generator algorithm 
   */
  int minimumNonZeroOrder;

  /**
   * the threshold for choosing the minimum nonzero order
   */
  double minimumNonZeroOrderThreshold;

  /**
   * generator invariants
   */
  std::list<std::list<vtkMomentsPolynomial>> generatorInvariants;

  /**
   * Langbein invariants
   */
  std::list<vtkMomentsPolynomial> langbeinInvariants;

  /**
   * values of moments of pattern
   */
  indexValueMap patternMoments;

  /**
   * the agorithm has two input ports
   * port 0 is the pattern, which is a vtkDataSet of scalar, vector, or matrix type
   * port 1 is the output of computeMoments, which is vtkImageData
   */
  int FillInputPortInformation(int port, vtkInformation* info) override;

  /**
   * the agorithm generates 4 outputs, all are vtkImageData
   * the first two have the topology of the momentData
   * a field storing the similarity to the pattern for all radii in a scalar field each
   * the normalized moments of the field
   * the latter two have extent 0, they only have 1 point in each field
   * the moments of the pattern
   * the first standard position of the normalized moments of the pattern
   */
  int FillOutputPortInformation(int port, vtkInformation* info) override;

  /**
   * Make sure that the user has not entered weird values.
   * @param pattern: the pattern that we will look for
   */
  void CheckValidity(vtkImageData* pattern);

  /**
   * this functions reads out the parameters from the pattern and checks if they assume reasonable
   * values
   * @param pattern: the pattern that we will look for
   */
  void InterpretPattern(vtkImageData* pattern);

  /**
   * this functions reads out the parameters from the momentData and checks if they assume
   * reasonable values and if they match the ones from the pattern
   * @param moments: the moment data
   */
  void InterpretField(vtkImageData* moments);

  /**
   * calculation of the moments of the pattern and its invariants.
   * we choose, which contractions (dominantContractions) can be used for the normalization of this
   * particular pattern, i.e. the ones that are nt zero or linearly dependent. They will later be
   * used for the normalization of the field moments, too.
   * @param dominantContractions: the vectors that can be used for the normalization of this
   * particular pattern, i.e. the ones that are nt zero or linearly dependent
   * @param pattern: the pattern
   * @param originalMomentsPattern: the moments of the pattern
   * @param normalizedMomentsPattern: the normalized moments of the pattern. It
   * visualizes how the standard position of this particular pattern looks like
   */
  void HandlePatternNormalization(std::vector<std::vector<vtkMomentsTensor> >& dominantContractions,
    vtkImageData* pattern,
    vtkImageData* originalMomentsPattern,
    vtkImageData* normalizedMomentsPattern);

  /**
   * main part of the pattern detetction
   * the moments of the field at each point are normalized and compared to the moments of the
   * pattern
   * @param dominantContractions: the dominant contractions, i.e. vectors for the normalization
   * w.r.t. rotation
   * @param moments: the moments of the field
   * @param normalizedMoments: the moment invariants of the field
   * @param pattern: the pattern
   * @param similarityFields: the output of this algorithm. it has the topology of moments and will
   * have a number of scalar fields euqal to NumberOfRadii. each point contains the similarity of
   * its surrounding (of size radius) to the pattern
   */
  void HandleFieldNormalization(std::vector<std::vector<vtkMomentsTensor> >& dominantContractions,
    vtkImageData* moments,
    vtkImageData* normalizedMoments,
    vtkImageData* pattern,
    vtkImageData* similarityFields);

  /**
   * this computes the translational factors necessary for normalization w.r.t. translation
   * we have radius and then p,q,r
   * @param pattern: the pattern
   */
  void BuildTranslationalFactorArray(vtkImageData* pattern);

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
  std::vector<vtkMomentsTensor> NormalizeT(std::vector<vtkMomentsTensor>& moments,
    double radius,
    bool isTranslation,
    vtkImageData* stencil);

  /**
   * normalization with respect to outer transaltion, i.e. the result will be invariant to adding a
   * constant
   * the translational factor is evaluated with the same stencil as the moments
   * @param moments: the moments at one point stored in a vector of tensors
   * @param radiusIndex: the index pointing to the integration radius over which the moments were
   * computed
   * @param isTranslation: if normalization w.tr.t. translation is desired by the user
   * @return the translationally normalized moments
   */
  std::vector<vtkMomentsTensor> NormalizeT(std::vector<vtkMomentsTensor>& moments,
    int radiusIndex,
    bool isTranslation);

  /**
   * normalization with respect to outer transaltion, i.e. the result will be invariant to adding a
   * constant
   * the translational factor is evaluated from the analytic formula
   * @param moments: the moments at one point stored in a vector of tensors
   * @param radius: the integration radius over which the moments were computed
   * @param isTranslation: if normalization w.tr.t. translation is desired by the user
   * @return the translationally normalized moments
   */
  std::vector<vtkMomentsTensor> NormalizeTAnalytic(std::vector<vtkMomentsTensor>& moments,
    double radius,
    bool isTranslation);

  /**
   * normalization with respect to outer scaling, i.e. the result will be invariant to multiplying a
   * constant
   * @param moments: the moments at one point stored in a vector of tensors
   * @param isScaling: if normalization w.tr.t. scalin is desired by the user
   * @return the scale normalized moments
   */
  std::vector<vtkMomentsTensor> NormalizeS(std::vector<vtkMomentsTensor>& moments,
    bool isScaling,
    double radius);

  /** normalization of the pattern with respect to rotation and reflection
   * @param dominantContractions: the vectors used for the normalization
   * @param isRotation: if the user wants normalization w.r.t rotation
   * @param isReflection: if the user wants normalization w.r.t reflection
   * @param moments: the moments at a given point
   */
  std::vector<vtkMomentsTensor> NormalizeR(std::vector<vtkMomentsTensor>& dominantContractions,
    bool isRotation,
    bool isReflection,
    std::vector<vtkMomentsTensor>& moments);

  /** calculation of the dominant contraction
   * there can be multiple dominant contractions due to EV or 3D
   * dominantContractions.at( i ) contains 1 vector in 2D and 2 in 3D
   * dominantContractions.size() = 1 if no EV, 2 if 1 EV, 4 if 2EV are chosen
   * if no contraction was found dominantContractions.size() = 0
   * if only one contraction was found in 3D dominantContractions.at(i).size() = 1 instead of 2
   * @param momentsPattern: the moments of the pattern
   * @return the dominant contractions, i.e. the biggest vectors that can be used for the
   * normalizaion w.r.t. rotation
   */
  std::vector<std::vector<vtkMomentsTensor> > CalculateDominantContractions(
    std::vector<vtkMomentsTensor>& momentsPattern);

  /** the dominant contractions are stored as a vector of integers that encode which tensors were
   * multiplied and contracted to form them. This function applies these excat instructions to the
   * moments in the field. That way, these can be normalized in the same way as the pattern was,
   * which is crucial for the comparison.
   * @param dominantContractions: the vectors that can be used for the normalization of this
   * particular pattern, i.e. the ones that are nt zero or linearly dependent
   * @param moments: the moments at one point
   */
  std::vector<vtkMomentsTensor> ReproduceContractions(
    std::vector<vtkMomentsTensor>& dominantContractions,
    std::vector<vtkMomentsTensor>& moments);

  /** if no dominant contractions could be found to be non-zero, the algorithm defaults back to
   * looking for all possible orientations of the given template the parameter AngleResolution
   * determines what "everywhere" means in 2D, we divide phi=[0,...,2Pi] into that many equidistant
   * steps in 3D, we divide phi=[0,...,2Pi] into that many equidistant steps and theta=[0,...,Pi] in
   * half that many steps to determine the rotation axis. Then, we use anther AngleResolution
   * different rotation angles in [0,...,2Pi] to cover all positions
   * @param momentsPatternNormal: this contains all orientations of the moments of the pattern.
   * during the detection later, we will compare the moments of the field to all these version of
   * the pattern
   * @param momentsPatternTranslationalNormal: this contains the moments that are not invariant to
   * orientation yet
   */
  void LookEverywhere(std::vector<std::vector<vtkMomentsTensor> >& momentsPatternNormal,
    std::vector<vtkMomentsTensor>& momentsPatternTranslationalNormal);

  /** if only one dominant contraction could be found to be non-zero, but no second one to be
   * linearly independent from the first one, the algorithm, will rotate the first contraction to
   * the x-axis and the look for all possible orientations of the given template around this axis.
   * In principal, it reduces the 3D problem to a 2D problem. the parameter AngleResolution determines
   * what "everywhere" means we divide phi=[0,...,2Pi] into that many equidistant steps
   * @param dominantContractions: the vectors used for the normalization
   * @param momentsPatternNormal: this contains all orientations of the moments of the pattern.
   * during the detection later, we will compare the moments of the field to all these version of
   * the pattern
   */
  void LookEverywhere(std::vector<std::vector<vtkMomentsTensor> >& dominantContractions,
    std::vector<std::vector<vtkMomentsTensor> >& momentsPatternNormal);

  /**
   * this functions uses the moments, weighs them with their corresponding basis function and adds
   * them up to approximate the value of the original function. The more moments are given, the
   * better the approximation, like in a taylor series
   * @param p: the location (3D point) at which the reconstructed field is evaluated
   * @param moments: the moments at a given location, which is used for the reconstruction
   * @param center: location, where the moments are given
   * @return: the value of the reconstructed function can be scalar, vector, or matrix valued
   */
  template<size_t S>
  vtkTuple<double, S> Reconstruct(double* p,
    std::vector<vtkMomentsTensor>& moments,
    double* center);

  /**
   * This function computes homogeneous invariants for a specific order
   * @param generator random generator
   * @param homogeneousInvariant a list of independent homogeneous invariants
   * @param order the maximum order of homogeneous invariants
   */
  std::list<vtkMomentsPolynomial> computeHomogeneousInvariant(default_random_engine & generator, std::list<vtkMomentsPolynomial> & homogeneousInvariant, unsigned order) const;

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
  std::vector<double> handlePatternGenerator(list<vtkMomentsIndex> variables,
					     indexValueMap & values,
					     list<vtkMomentsPolynomial> & invariants,
					     const list<list<vtkMomentsPolynomial>> & homogeneousInvariants,
					     const list<list<list<vtkMomentsPolynomial>>> & mixedInvariants,
					     vtkImageData* normalizedMomentsPattern);

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
  void handleFieldGenerator(std::list<vtkMomentsPolynomial> & invariants,
			    indexValueMap & values,
			    const vector<double> & patternDesc,
			    vtkImageData* moments,
			    vtkImageData* pattern,
			    vtkImageData* similarityFields,
			    vtkImageData* normalizedMomentsFields);

  /**
   * This function computes the invariants using the algorithm proposed by Langbein. 
   * For the sake of reducing computation time, there are a factor cap(4) and a rank cap (14).
   * This, of course, prevents the algorithm from finding 
   * all possible invariants. This is also a drawback of the algorithm.
   * The function stops when the number of invariants found reaches its theoretical maximum.
   * @return a list of Langbein invariants
   */
  std::list<vtkMomentsPolynomial> computeLangbeinInvariants();

  /**
   * This function computes the Langbein invariants based on the moments
   * of the pattern. 
   * @param[in] invariants a list of Langbein invariants
   * @param values values of moments
   * @return a vector of moment invariants computed on the pattern moments. 
   */  
  std::vector<double> handlePatternLangbeinInvariants(const std::list<vtkMomentsPolynomial> & invariants, indexValueMap & values, vtkImageData* normalizedMomentsPattern);

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
  inline bool computeMixedInvariantOrderPairHelper(list<vtkMomentsPolynomial> & invariants, const list<vtkMomentsIndex> & variables, forward_list<vtkMomentsPolynomial> & polys, list<polyDerivativePair> & pairs, default_random_engine & generator, unsigned o1, unsigned exp1, unsigned o2, unsigned exp2, unsigned maxNum) const;

  /** 
   * This funcion searches for independent mixed invariants for a pair of order.
   * @param homoInv a list of homogeneous invariants
   * @param variables a list of moments
   * @param o1 order of the first moment
   * @param o2 order of the second moment
   * @return a list of independent mixed invariants
   */
  list<vtkMomentsPolynomial> computeMixedInvariantsOrderPair(const list<vtkMomentsPolynomial> & homoInv, const list<vtkMomentsIndex> & variables, unsigned o1, unsigned o2) const;

  /**
   * This function computes the mixed invariants for every order pairs
   * The outtermost list sorts the mixed invariants by Order
   * The middle list sorts the mixed invariants by the maximum order of a mixed invariant
   * The innermost list sorts the mixed invariants by the minium order of a mixed invariant
   * @param HomoInvariants a list of homogeneous invariants
   * @param variables a list of moments
   * @return mixed invariants
   */ 
  list<list<list<vtkMomentsPolynomial>>> computeMixedInvariant(list<vtkMomentsPolynomial> & HomoInvariants, const list<vtkMomentsIndex> & variables) const;

  /**
   * The maximum number of invariants can be computed theoretically.
   * This function provides the expected number of invariants.
   * @param d dimension
   * @param fr field rank
   * @param o order
   * @return a vector of length 5. They are moments of order o, existing homo. inv. of order o, moments up to order o_m, theor. existing inv. up to order o_m and mixed inv. needed
   */
  static vector<int> getInvariantNumberVec(int d, int fr, int o);

  /**
   * This function creates moments up to the Order
   * The moments are sorted, e.g. M0,M00,M01,M11...
   * @return a list of moments
   */
  inline list<vtkMomentsIndex> createVariables() const;

  /**
   * This function assigns random values to moments.
   * @param variables a list of moments
   * @param generator a random number generator
   * @return a map whose key is moment and value is the value of a moment.
   */  
  static indexValueMap generateValues(const list<vtkMomentsIndex> variables, default_random_engine & generator);

  /**
   * This function computes derivatives of a given polynomial with repsect to given moments
   * @param variables a list of moments
   * @param poly a the polynomial 
   * @return the derivatives with respect to moments. 
   */
  static polyDerivativePair computeDerivative(const list<vtkMomentsIndex> variables, const vtkMomentsPolynomial & poly);

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
  bool addIndependentInvariants(default_random_engine & generator,
				list<polyDerivativePair> & indInv,
				const list<vtkMomentsIndex> variables,
				const vtkMomentsPolynomial & poly) const;

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
   bool computeInvariant(vtkMomentsTensorSimple & tensor,
			 list<vtkMomentsPolynomial> & invariants,
			 forward_list<vtkMomentsPolynomial> & polys,
			 const list<vtkMomentsIndex> & variables,
			 list<polyDerivativePair> & pairs,
			 default_random_engine & generator,
			 int maxNumber = -1) const;

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
  void momentInvariantsPatternDetection(vtkInformation* momentsInfo, vtkImageData* pattern, vtkImageData* momentData, vtkImageData* similarityFields, vtkImageData* normalizedMomentsField, vtkImageData* originalMomentsPattern, vtkImageData* normalizedMomentsPattern);

  /**
   * Print invariants and values for generator and Langbein algorithm
   */
  void printInvariants();

  /**
   * This function creates a name for an invariant. The name is used to for storing to vtkImageData.
   * @param radius the radius that is used to compute moments
   * @param invariant an invariant in the polynomial format
   */
  static string getArrayNameFromInvariant(double radius, const vtkMomentsPolynomial & invariant);
};

#endif
