/**
 * @class vtkMomentsFactor
 * @brief a moment factor
 *
 * a vtkMomentsFactor is vtkMomentsIndex^m_factorOrder, e.g M_{001}^3
 */
#ifndef VTKMOMENTSFACTOR_H
#define VTKMOMENTSFACTOR_H
#ifndef __VTK_WRAP__

#include "vtkMomentsIndex.h"

class MOMENTINVARIANTS_EXPORT vtkMomentsFactor {
private:
  /**
   * a factor is M^m_factorOrder
   */
  unsigned m_factorOrder;

  /**
   * the moment represented in the format M_{001}
   */
  vtkMomentsIndex m_index;
  
public:
  /**
   * default constructor
   */
  vtkMomentsFactor();

  /**
   * constructor 
   * the m_factorOrder is default 1
   * @param dimension e.g. 2D or 3D
   * @param momentRank Order of the moment tensor
   * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
   * @param baseIndex e.g. "001"
   */
  vtkMomentsFactor(unsigned dimension, unsigned momentRank, unsigned fieldRank, const vector<unsigned> & baseIndex);

  /**
   * constructor
   * @param factorOrder e.g. M_{001}^factororder
   * @param dimension e.g. 2D or 3D
   * @param momentRank Order of the moment tensor
   * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
   * @param baseIndex e.g. "001"
   */
  vtkMomentsFactor(unsigned factorOrder, unsigned dimension, unsigned momentRank, unsigned fieldRank, const vector<unsigned> & baseIndex);

  /**
   * copy constructor
   */
  vtkMomentsFactor(const vtkMomentsFactor & factor);

  /**
   * get the order of the factor
   */
  unsigned getFactorOrder() const;

  /**
   * get the moment, e.g. M_{001}
   */
  const vtkMomentsIndex & getIndex() const;

  friend bool operator == (const vtkMomentsFactor & factor1, const vtkMomentsFactor & factor2);
  
  friend bool operator < (const vtkMomentsFactor & factor1, const vtkMomentsFactor & factor2);

  /**
   * output stream for printing
   */
  friend ostream & operator << (ostream & os, const vtkMomentsFactor & factor);

  /**
   * encode to a string
   */
  friend stringstream & operator << (stringstream & ofs, const vtkMomentsFactor & factor);

  /**
   * decode from a stream
   */
  friend istream & operator >> (istream & ifs, vtkMomentsFactor & factor);

  /**
   * m_factorOrder += 1
   */
  void increaseFactorOrder(unsigned order = 1);
  
  /**
   * m_factorOrder -= 1
   */
  bool decreaseFactorOrder(unsigned order = 1);
  
  /**
   * compute the value of the factor
   */
  double assignValue(const indexValueMap & map) const;

};

inline unsigned vtkMomentsFactor::getFactorOrder() const { return m_factorOrder; }
inline const vtkMomentsIndex & vtkMomentsFactor::getIndex() const { return m_index; }

inline void vtkMomentsFactor::increaseFactorOrder(unsigned order) {
  m_factorOrder += order;
}

inline bool vtkMomentsFactor::decreaseFactorOrder(unsigned order) {
  m_factorOrder -= order;

  // remove factor if the order is 0
  if (m_factorOrder == 0)
    return true;
  else
    return false;
}

inline double vtkMomentsFactor::assignValue(const indexValueMap & map) const {
  auto it = map.find(m_index);
  if (it == map.end())
    return 0;
  else
    return pow(it->second, m_factorOrder);
}

#endif // __VTK_WRAP__
#endif //VTKMOMENTSFACTOR_H
