/**
 * @class vtkMomentsTensorSimple
 * @brief a simplified verision of vtkMomentsTensor
 *
 * class vtkMomentsTensorSimple is a simplified version of the class vtkMomentsTensor.
 * It does not do any tensor data computation. The tensor operations only change
 * productInfo, contractionInfo and other parameters. 
 */

#ifndef VTKMOMENTSTENSORSIMPLE_H
#define VTKMOMENTSTENSORSIMPLE_H
#ifndef __VTK_WRAP__

#include <iostream>
#include <list>
#include <math.h>
#include <vector>

#include "MomentInvariantsModule.h"


using namespace std;

class MOMENTINVARIANTS_EXPORT vtkMomentsTensorSimple {
 private:
 /**
  * Dimension of the tensor can for example be 2 or 3
  */
 unsigned m_dimension;

 /**
  * rank of this tensor, i.e. number of indices to reference its entries.
  * for uncontracted tensor it is: m_momentRank + m_FieldRank
  */
 unsigned m_rank;

 /**
  * Fieldrank of the tensor is 0 for scalars, 1 for vectors, 3 for matrices
  */
 unsigned m_fieldRank;

 /**
  * Order of the moment tensor
  */
 unsigned m_momentRank;

 /**
  * optional outer tensor product information that produced it
  */
 list<unsigned> m_productInfo;
 
 /**
  * optional contraction information that produced it
  */
 list<unsigned> m_contractionInfo;

 public:
 /**
  * defalut constructor
  */
  vtkMomentsTensorSimple();

  /**
   * default contructor for a paticular dimension
   * @param dimension i.e. 2 or 3
   */
  vtkMomentsTensorSimple(unsigned dimension);

 /**
  * constructor 
  * @param dimension i.e. 2 or 3
  * @param rank rank of the tensor, i.e. number of indices to reference its entries
  * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
  */
  vtkMomentsTensorSimple(unsigned dimension, unsigned rank, unsigned fieldRank);

  /**
   * copy constructor
   */
  vtkMomentsTensorSimple(const vtkMomentsTensorSimple & tensor);

  /**
   * copy constructor. It computes a contraction after the input is copied.
   * @param i the first constraction index
   * @param j the second constraction index
   */
  vtkMomentsTensorSimple(const vtkMomentsTensorSimple & tensor, unsigned i, unsigned j);

 /**
  * constructor: rank = fieldRank + momentRank
  * @param dimension i.e. 2 or 3
  * @param rank rank of the tensor, i.e. number of indices to reference its entries
  * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
  * @param momentRank Order of the moment tensor
  */
  vtkMomentsTensorSimple(unsigned dimension, unsigned rank, unsigned fieldRank, unsigned momentRank);

  /**
   * get rank of the tensor, i.e. number of indices to reference its entries
   */
  unsigned getRank() const ;

  /**
   * get field rank, 0 for scalars, 1 for vectors, 2 for matrices
   */
  unsigned getFieldRank() const ;

  /**
   * get moment rank, order of the moment tensor
   */
  unsigned getMomentRank() const;

  /**
   * get dimension, e.g. 2D or 3D
   */
  unsigned getDimension() const;

  /**
   * get the number of entries of this tensor
   */
  unsigned size() const;

  /**
   * get the vector with the indices that indicate which tensor this one was produced from through
   * contraction
   */
  const list<unsigned> & getContractionInfo() const;

  /**
   * get the vector with the indices that indicate which tensor this one was produced from through
   * multiplication
   */
  const list<unsigned> & getProductInfo() const;

  /**
   * inverse function to getIndex
   * @param index the place in the flat c++ std vector
   * @return vector of tensor indices that identify an entry
   */
  vector<int> getIndices(int index) const;

 /**
  * the moment tensors have two types of indices fieldIndices and momentIndices
  * fieldIndices of length fieldRank refer to the components of the original data (3 for vector, 9
  * for matrix) momentIndices of length momentRank refer to the basis function
  * @param index the place in the flat c++ std vector
  * @return the indices that correspond to the basis function
  */
 vector<int> getMomentIndices(int index) const;
 
  /**
   * append the contraction indices to the m_contractionInfo
   * @param i index for contraction
   * @param j index for contraction
   */
  void setContractionInfo(unsigned i, unsigned j);
  
  /** 
   * This function concatenates productInfo of itself and the other tensor.
   * @param productInfo the productInfo of the other tensor
   */
  void concatProductInfo(const list<unsigned>& productInfo);

  /**
   * after tensors are multiplied, we store which tensors produced them
   * @param parentInfo contraction information  of the tensor that produces the new tensor through
   */
  void setProductInfo(const list<unsigned>& parentInfo);

  /**
   * multiply a tensor with another tensor.
   * the rank of the result is the sum of the ranks of its parents
   * @param tensor the other thensor
   */
  void tensorProduct(const vtkMomentsTensorSimple & tensor);

  /**
   * compute the power of the tensor.
   * @param exp the exponent 
   */
  void tensorPow(unsigned exp);

  /**
   * This function produces a tensor contraction of the indices i and j
   * @param i index for contraction
   * @param j index for contraction
   */
  void contract(unsigned i, unsigned j);

  /** 
   * This function creates a list of contractions, i.e. (0,1), (0,2)...
   * @param tensorList the list that the contractions will append to
   */ 
  void contractAll(list<vtkMomentsTensorSimple> & tensorList);

  /** 
   * This function creates a list of contractions, all of which are rank 0.
   * @param tensorList the list that the contractions will append to
   */ 
  static void contractAllRank0(list<vtkMomentsTensorSimple> & tensorList);

  /**
   * This function computes the order of two tensors.
   */
  static bool compare(const vtkMomentsTensorSimple & tensor1,const vtkMomentsTensorSimple & tensor2);

  friend bool operator == (const vtkMomentsTensorSimple & tensor1, const vtkMomentsTensorSimple & tensor2);
};


inline unsigned vtkMomentsTensorSimple::getRank() const { return m_rank; }

inline unsigned vtkMomentsTensorSimple::getFieldRank() const { return m_fieldRank; }

inline unsigned vtkMomentsTensorSimple::getMomentRank() const { return m_momentRank; }

inline unsigned vtkMomentsTensorSimple::getDimension() const { return m_dimension; }

inline unsigned vtkMomentsTensorSimple::size() const { return pow(m_dimension, m_rank); }

inline const list<unsigned> & vtkMomentsTensorSimple::getContractionInfo() const { return m_contractionInfo; }

inline const list<unsigned> & vtkMomentsTensorSimple::getProductInfo() const { return m_productInfo; }

inline void vtkMomentsTensorSimple::setContractionInfo(unsigned i, unsigned j) {
  m_contractionInfo.push_back(i);
  m_contractionInfo.push_back(j);
}

inline void vtkMomentsTensorSimple::concatProductInfo(const list<unsigned>& productInfo) {
  m_productInfo.insert(m_productInfo.end(), productInfo.begin(), productInfo.end());
  m_productInfo.sort();
}

inline void vtkMomentsTensorSimple::setProductInfo(const list<unsigned>& parentInfo) { m_productInfo = parentInfo;}

inline void vtkMomentsTensorSimple::contractAll(list<vtkMomentsTensorSimple> & tensorList) {
  for (unsigned i = 1; i < m_rank; ++i)  
    tensorList.emplace_back(*this, 0, i);  
}

#endif // __VTK_WRAP__
#endif //VTKMOMENTSTENSORSIMPLE_H
