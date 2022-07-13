/**
 * @class vtkMomentsPolynomial
 * @brief a polynomial of moments
 *
 * a vtkMomentsPolynomial is a summation of moment products, e.g. 2 * 1M^2 * 2M + 3 * 2M^2
 */
#ifndef VTKMOMENTSPOLYNOMIAL_H
#define VTKMOMENTSPOLYNOMIAL_H
#ifndef __VTK_WRAP__

#include "vtkMomentsProduct.h"


class MOMENTINVARIANTS_EXPORT vtkMomentsPolynomial {
private:
  /**
   * a list of product
   */
  list<vtkMomentsProduct> m_polynomial;

  /**
   * optional outer tensor product information that produced it
   */
  list<unsigned> m_productInfo;

  /**
   * optional contraction information that produced it
   */
  list<unsigned> m_contractionInfo;

  /**
   * contraction information in the absolute indices format
   */
  vector<pair<unsigned,unsigned>> m_contractionInfoAbs;

protected:
  /**
   * add a product to the current polynomial
   * @param prod add a product to the polynomial
   */
  void addProduct(const vtkMomentsProduct & prod);

  /**
   * compute the contractionInfo. Note that the indices in m_contractionInfo are relative indices
   * For example, m_contractionInfo = "0201" means (0,2), (1,3)
   * @return the absolute indices
   */
  vector<pair<unsigned,unsigned>> reconstructContractions() const;

  /**
   * This function constructs a polynomail from both m_productInfo and m_contractionInfo.
   * @param dimension e.g. 2 or 3
   * @param fieldRank Fieldrank of the tensor is 0 for scalars, 1 for vectors, 3 for matrices
   * @param productInfo optional outer tensor product information that produced it
   * @param contractionInfo optional contraction information that produced it
   */
  void vtkMomentsPolynomialHelper(unsigned dimension, unsigned fieldRank, const list<unsigned> & productInfo, const list<unsigned> & contractionInfo);

public:
  /**
   * default constructor
   */
  vtkMomentsPolynomial();

  /**
   * constructor
   * @param dimension e.g. 2 or 3
   * @param fieldRank Fieldrank of the tensor is 0 for scalars, 1 for vectors, 3 for matrices
   * @param productInfo optional outer tensor product information that produced it
   * @param contractionInfo optional contraction information that produced it
   */
  vtkMomentsPolynomial(unsigned dimension, unsigned fieldRank, const list<unsigned> & productInfo, const list<unsigned> & contractionInfo);

  /**
   * constructor that uses a tensor 
   * @param tensor a tensor that will be converted to a polynomial
   */
  vtkMomentsPolynomial(const vtkMomentsTensorSimple & tensor);

  /**
   * copy constructor
   */
  vtkMomentsPolynomial(const vtkMomentsPolynomial & poly);

  /**
   * get the const reference of polynomial, i.e. a list of product
   * @return m_polynomial
   */
  const list<vtkMomentsProduct>& getPolynomialConst() const; 
  
  /**
   * get the reference of polynomial, i.e. a list of product
   * @return m_polynomial
   */
  list<vtkMomentsProduct>& getPolynomial();

  /**
   * get the m_productInfo
   * @return m_productInfo
   */
  const list<unsigned> & getProductInfo() const;

  /**
   * get the m_contractionInfo
   * @return m_contractionInfo
   */
  const list<unsigned> & getContractionInfo() const;


  /**
   * get the sum of the orders of all factors
   */  
  unsigned getTotalFactorOrder() const;

  /**
   * get the max order in the tensor. For example, the max order of tensor 2M^3 3M^2 is 3.
   */
  unsigned getMaximumTensorOrder() const;

  /**
   * get the min order in the tensor. For example, the min order of tensor 2M^3 3M^2 is 2.
   */
  unsigned getMinimumTensorOrder() const;

  /**
   * overload operators
   */
  friend bool operator == (const vtkMomentsPolynomial & poly1, const vtkMomentsPolynomial & poly2);
  friend bool operator < (const vtkMomentsPolynomial & poly1, const vtkMomentsPolynomial & poly2);

  /**
   * output stream for printing
   */
  friend ostream & operator << (ostream & os, const vtkMomentsPolynomial & poly);

  /**
   * encode to a string
   */
  friend stringstream & operator << (stringstream & ofs, const vtkMomentsPolynomial & polynomial);

  /**
   * decode from a stream
   */
  friend istream & operator >> (istream & ifs, vtkMomentsPolynomial & polynomial);

  /**
   * assign operator
   */
  vtkMomentsPolynomial & operator = (const vtkMomentsPolynomial & poly);

  /**
   * take derivative with respect to a moment
   * @param index a moment
   */
  void takeDerivative(const vtkMomentsIndex & index);

  /**
   * compute the value of a tensor
   * @param map values of moments
   * @return the value of the polynomial
   */
  double assignValue(const indexValueMap & map) const;

  /**
   * compute the value of the tensor, then normalize the value base on the number of factor
   * e.g. (1M2M3M)^(1/3) because there are 3 moments
   * @param map values of moments
   * @return the normalized value of the polynomial
   */
  double assignValueAndNormalize(const indexValueMap & map) const;

  /**
   * print the 1M2M3M style representation in Latex format of the tensor
   */
  string printTensor() const;

  /**
   * print the 1M2M3M style representation of the tensor
   */
  string printTensorConcise() const;  
};

typedef pair<vtkMomentsPolynomial,vector<vtkMomentsPolynomial>> polyDerivativePair;

inline const list<vtkMomentsProduct>& vtkMomentsPolynomial::getPolynomialConst() const { return m_polynomial;}

inline list<vtkMomentsProduct>& vtkMomentsPolynomial::getPolynomial() { return m_polynomial;}

inline unsigned vtkMomentsPolynomial::getTotalFactorOrder() const { return m_polynomial.front().getTotalFactorOrder(); }

inline const list<unsigned> & vtkMomentsPolynomial::getProductInfo() const { return m_productInfo; }

inline const list<unsigned> & vtkMomentsPolynomial::getContractionInfo() const { return m_contractionInfo; }

inline unsigned vtkMomentsPolynomial::getMaximumTensorOrder() const { return *max_element(m_productInfo.cbegin(), m_productInfo.cend()) - getPolynomialConst().front().getProduct().front().getIndex().getFieldRank(); }

inline unsigned vtkMomentsPolynomial::getMinimumTensorOrder() const { return *min_element(m_productInfo.cbegin(), m_productInfo.cend()) - getPolynomialConst().front().getProduct().front().getIndex().getFieldRank(); }

#endif // __VTK_WRAP__
#endif // VTKMOMENTSPOLYNOMIAL_H
