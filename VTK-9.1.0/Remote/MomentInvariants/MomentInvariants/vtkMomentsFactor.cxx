#include "vtkMomentsFactor.h"

/**
 * default constructor
 */
vtkMomentsFactor::vtkMomentsFactor()
  : m_factorOrder(0) {}

/**
 * constructor
 * the m_factorOrder is default 1
 * @param dimension e.g. 2D or 3D
 * @param momentRank Order of the moment tensor
 * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
 * @param baseIndex e.g. "001"
 */
vtkMomentsFactor::vtkMomentsFactor(unsigned dimension, unsigned momentRank, unsigned fieldRank, const vector<unsigned> & baseIndex)
  : m_factorOrder(1)
  , m_index(dimension, momentRank, fieldRank, baseIndex) {}

/**
 * constructor
 * @param factorOrder e.g. M_{001}^factororder
 * @param dimension e.g. 2D or 3D
 * @param momentRank Order of the moment tensor
 * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
 * @param baseIndex e.g. "001"
 */
vtkMomentsFactor::vtkMomentsFactor(unsigned factorOrder, unsigned dimension, unsigned momentRank, unsigned fieldRank, const vector<unsigned> & baseIndex)
  : m_factorOrder(factorOrder)
  , m_index(dimension, momentRank, fieldRank, baseIndex) {}

/**
 * copy constructor
 */
vtkMomentsFactor::vtkMomentsFactor(const vtkMomentsFactor & factor)
  : m_factorOrder(factor.m_factorOrder)
  , m_index(factor.m_index) {}

bool operator == (const vtkMomentsFactor & factor1, const vtkMomentsFactor & factor2) {
  return (factor1.m_factorOrder == factor2.m_factorOrder) &&
    (factor1.m_index == factor2.m_index);
}

bool operator < (const vtkMomentsFactor & factor1, const vtkMomentsFactor & factor2) {  
  return (factor1.m_index < factor2.m_index) ||
    ((factor1.m_index == factor2.m_index) &&
     (factor1.m_factorOrder < factor2.m_factorOrder));
}

/**
 * output stream for printing
 */
ostream & operator <<(ostream & os, const vtkMomentsFactor & factor) {
  if (factor.m_factorOrder == 0) {
    cerr << "vtkMomentsFactor operator << error: m_factorOrder == 0\n";
    exit(0);
  }

  os << factor.m_index;
  
  if (factor.m_factorOrder > 1)
    os << "^{" << factor.m_factorOrder << "}";

  return os;
}

/**
 * encode to a string
 */
stringstream & operator << (stringstream & ofs, const vtkMomentsFactor & factor) {
  ofs << factor.m_factorOrder << " ";
  ofs << factor.m_index;
  return ofs;
}

/**
 * decode from a stream
 */
istream & operator >> (istream & ifs, vtkMomentsFactor & factor) {
  ifs >> factor.m_factorOrder;
  ifs >> factor.m_index;
  return ifs;
}
