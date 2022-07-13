/**
 * @class vtkMomentsIndex
 * @brief a representation of moment
 *
 * vtkMomentsIndex represents a moment using its indices, i.e. the occurrence of each dimension.
 * For example, a moment \int x^2 y f(x,y) dxdy is represented as M_{001}.
 */
#ifndef VTKMOMENTSINDEX_H
#define VTKMOMENTSINDEX_H
#ifndef __VTK_WRAP__

#include <algorithm>
#include <forward_list>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "vtkMomentsTensorSimple.h"
#include "vtkMomentsGeneratorHelper.h"

class MOMENTINVARIANTS_EXPORT vtkMomentsIndex {
  private:
  /**
   * Order of the moment tensor
   */
  unsigned m_momentRank;

  /**
   * Fieldrank of the tensor is 0 for scalars, 1 for vectors, 3 for matrices
   */
  unsigned m_fieldRank;

  /**
   * MomentIndex describes the occurrence of each dimension.
   * For example, the momentIndex of M_{001} is 21, becuase there are two 0s and one 1.
   */
  vector<unsigned> m_momentIndex;

  /**
   * MomentIndices is "001" in M_{001}. 
   * It is always sorted. It can't be "010" or "100".
   * Its meaning for 2D moment is \int x^2 y f(x,y) dxdy
   */
  vector<int> m_momentIndices;
  
  /**
   * It the index of a field, e.g. NULL for scalar field
   * 0 and 1 for vector field
   * 00,01,10 and 11 for matrix field.
   */
  vector<int> m_fieldIndices;

  /**
   * The signature is used for ordering.
   * It is faster to compare a signature than comparing a vector.
   */
  unsigned m_momentSignature;
  unsigned m_fieldSignature;

  /**
   * compute m_momentSignature and m_fieldSignature
   */
  void computeSignature();

  public:
  /**
   * default constructor
   */
  vtkMomentsIndex();

  /**
   * The baseIndex for moment M_{001} is "001". It is not sorted. 
   * It could be "010", which is a duplicate of "001".
   * @param dimension e.g. 2D or 3D
   * @param momentRank Order of the moment tensor
   * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
   * @param baseIndex e.g. "001"
   */
  vtkMomentsIndex(unsigned dimension, unsigned momentRank, unsigned fieldRank, const vector<unsigned> & baseIndex);

  /**
   * copy contructor
   */
  vtkMomentsIndex(const vtkMomentsIndex & index);

  /**
   * get moment rank, order of the moment tensor
   */
  unsigned getMomentRank() const;

  /**
   * get field rank, 0 for scalars, 1 for vectors, 2 for matrices
   */
  unsigned getFieldRank() const;

  /**
   * get rank of the tensor, i.e. number of indices to reference its entries
   */
  unsigned getRank() const;

  /**
   * get m_momentIndex, e.g. "21" for  "001"
   */
  const vector<unsigned> & getMomentIndex() const;

  /**
   * get dimension, e.g. 2D or 3D
   */
  unsigned getDimension() const;

  /**
   * get a concatenated indices of m_momentIndices and m_fieldIndices
   */
  vector<int> getIndices() const;

  /**
   * compute momentIndices from momentIndex. It thus is sorted.
   */
  void computeMomentIndices();

  /**
   * get m_momentSignature
   */
  unsigned getMomentSignature() const;

  /**
   * get m_fieldSignature
   */
  unsigned getFieldSignature() const;

  friend bool operator == (const vtkMomentsIndex & index1, const vtkMomentsIndex & index2);
  friend bool operator < (const vtkMomentsIndex & index1, const vtkMomentsIndex & index2);

  /**
   * output stream for printing
   */
  friend ostream & operator << (ostream & os, const vtkMomentsIndex & index);

  /**
   * encode to a string
   */
  friend stringstream & operator << (stringstream & ofs, const vtkMomentsIndex & index);

  /**
   * decode from a stream
   */
  friend istream & operator >> (istream & ifs, vtkMomentsIndex & index);

  /**
   * generate variables (moments)
   * The duplicates are removed.
   * For example, for an order 2 vector field, the moments are
   * 000,010,110,001,011,111. 100 and 101 are duplicates. Thus they are removed.
   * @param dimension: e.g. 2D or 3D
   * @param momentRank: Order of the moment tensor
   * @param fieldRank: 0 for scalars, 1 for vectors, 3 for matrices
   * @return a list of moments
   */
  static list<vtkMomentsIndex> generateVariables(unsigned dimension, unsigned momentRank, unsigned fieldRank);
  
};

inline void  vtkMomentsIndex::computeSignature() {
  for (unsigned i = 0; i < m_momentIndex.size(); i++)
    m_momentSignature += m_momentIndex[i] * (unsigned)pow((double)(m_momentRank + 1), (double)i);

  for (unsigned i = 0; i < m_fieldIndices.size(); i++)
    m_fieldSignature += m_fieldIndices[i] * (unsigned)pow((double)getDimension(), (double)i);
}
inline unsigned vtkMomentsIndex::getMomentRank() const { return m_momentRank; }
inline unsigned vtkMomentsIndex::getFieldRank() const { return m_fieldRank; }
inline unsigned vtkMomentsIndex::getRank() const { return m_momentRank + m_fieldRank; }
inline const vector<unsigned> & vtkMomentsIndex::getMomentIndex() const { return m_momentIndex; }
inline unsigned vtkMomentsIndex::getDimension() const { return m_momentIndex.size(); }
inline unsigned vtkMomentsIndex::getMomentSignature() const { return m_momentSignature; }
inline unsigned vtkMomentsIndex::getFieldSignature() const { return m_fieldSignature; }



/**
 * The hash functions are necessary to the unordered_map.
 */
inline void hash_combine(std::size_t& seed) {}

template <typename T, typename... Rest>
inline void hash_combine(size_t& seed, const T& v, Rest... rest) {
  hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  hash_combine(seed, rest...);
}

namespace std{
  template<>
  struct hash<vtkMomentsIndex> {
    size_t operator () (const vtkMomentsIndex & index) const noexcept {
      size_t seed = 0;
      hash_combine<unsigned>(seed, index.getMomentRank(), index.getFieldRank(), index.getMomentSignature(), index.getFieldSignature());
      return seed;
    }
  };
}

/**
 * This is for getting the value of a moment. It is the basic operation when computing the value
 * of a polynomial. For example given M_{0} = 0, M_{1} = 1, M_{0} + M_{1} = 0 + 1 = 1.
 */
typedef unordered_map<vtkMomentsIndex, double> indexValueMap;

#endif // __VTK_WRAP__
#endif //VTKMOMENTSINDEX_H
