/**
 * @class vtkMomentsProduct
 * @brief a product of moment factors
 *
 * a vtkMomentsProduct is a product of moment factors, e.g. 2 * 1M^2 * 2M
 */
#ifndef VTKMOMENTSPRODUCT_H
#define VTKMOMENTSPRODUCT_H
#ifndef __VTK_WRAP__

#include "vtkMomentsFactor.h"

class MOMENTINVARIANTS_EXPORT vtkMomentsProduct {
private:
  /**
   * a product is m_scalar * M_{00} * M_{01}
   */
  unsigned m_scalar;

  /**
   * a list of factors
   */
  list<vtkMomentsFactor> m_product;

  /**
   * signature for ordering. Compareing it is faster than 
   * comparing a vector.
   */
  unsigned m_signature;

public:
  /**
   * default costructor
   */
  vtkMomentsProduct();

  /**
   * constructor
   * @param scalar a product is scalar * M_{0} * M_{00}^2 ...
   * @param factor a moment factor, e.g. M_{00}^2
   */
  vtkMomentsProduct(unsigned scalar, const vtkMomentsFactor & factor);

  /**
   * copy constructor
   */
  vtkMomentsProduct(const vtkMomentsProduct & prod);

  /**
   * get the m_scalar
   */
  unsigned getScalar() const;

  /**
   * get m_product
   */
  const list<vtkMomentsFactor> & getProduct() const;

  /**
   * get m_signature
   */
  unsigned getSignature() const;

  /**
   * get the sum of the orders of all factors
   */
  unsigned getTotalFactorOrder() const;

  friend bool operator == (const vtkMomentsProduct & product1, const vtkMomentsProduct & product2);
  
  friend bool operator < (const vtkMomentsProduct & product1, const vtkMomentsProduct & product2);

  /**
   * output stream for printing
   */
  friend ostream & operator << (ostream & os, const vtkMomentsProduct& prod);
  
  /**
   * encode to a string
   */
  friend stringstream & operator << (stringstream & ofs, const vtkMomentsProduct & product);

  /**
   * decode from a stream
   */
  friend istream & operator >> (istream & ifs, vtkMomentsProduct & product);

  /**
   * compute the signature
   */
  void computeSignature();

  /**
   * add a product to itslef and the current product becomes the result    
   * only products with the same moment can be added
   * @param prod a product
   * @return whether or not the two products have the same moment
   */
  bool add(const vtkMomentsProduct & prod);

  /**
   * multiply the product with a factor
   * @param inputFactor a factor
   */
  void multiply(const vtkMomentsFactor & inputFactor);

  /**
   * take derivative with respect to a moment
   * e.g. d 2*M^3 / dM = (2*3) * M^(3-1)
   * @param index the variable M
   * @return whether or not the product contains the varible M
   */
  bool takeDerivative(const vtkMomentsIndex & index);

  /**
   * compute the value of the product
   * @param map a map whose key is vtkMomentsIndex and value is the value of the moment
   * @return the value of the product
   */
  double assignValue(const indexValueMap & map) const;
};

inline unsigned vtkMomentsProduct::getScalar() const { return m_scalar;}
inline const list<vtkMomentsFactor> & vtkMomentsProduct::getProduct() const { return m_product;}
inline unsigned vtkMomentsProduct::getSignature() const { return m_signature; }

inline bool vtkMomentsProduct::add(const vtkMomentsProduct & prod) {
  if (*this == prod) {
    m_scalar += prod.getScalar();
    return true;
  } else
    return false;
}

#endif // __VTK_WRAP__
#endif //VTKMOMENTSPRODUCT_H
