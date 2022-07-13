#include "vtkMomentsProduct.h"

/**
 * default costructor
 */
vtkMomentsProduct::vtkMomentsProduct()
  : m_scalar(1)
  , m_signature(0)
{}

/**
 * constructor
 * @param scalar a product is scalar * M_{0} * M_{00}^2 ...
 * @param factor a moment factor, e.g. M_{00}^2
 */
vtkMomentsProduct::vtkMomentsProduct(unsigned scalar, const vtkMomentsFactor & factor)
  : m_scalar(scalar)
  , m_product(1,factor) {
  computeSignature();
}

/**
 * copy constructor
 */
vtkMomentsProduct::vtkMomentsProduct(const vtkMomentsProduct & prod)
  : m_scalar(prod.m_scalar)
  , m_product(prod.m_product)
  , m_signature(prod.m_signature) {}

/**
 * get the sum of the orders of all factors
 */
unsigned vtkMomentsProduct::getTotalFactorOrder() const {
  unsigned s = 0;
  for (auto & factor:m_product)
    s += factor.getFactorOrder();

  return s;
}

/**
 * compute the signature
 */
void vtkMomentsProduct::computeSignature() {
  // compute total rank
  unsigned rank = 0;
  
  for (auto & factor:m_product)
    rank += factor.getFactorOrder() * (factor.getIndex().getRank());
  
  forward_list<int> m_index;
  
  auto it = m_index.before_begin();
  for (auto & factor : m_product) {
    vector<int> indices = factor.getIndex().getIndices();
    
    for (unsigned j = 0; j < factor.getFactorOrder(); j++) {
      m_index.insert_after(it,indices.begin(),indices.end());
      advance(it,indices.size());
    }      
  }

  m_signature = 0;
  auto cit = m_index.cbegin();

  for (unsigned i = 0; i < rank; i++, ++cit) 
    m_signature += (*cit) * (unsigned)pow((double)m_product.front().getIndex().getDimension(), (double)i);
        
}

/**
 * multiply the product with a factor
 * @param inputFactor a factor
 */
void vtkMomentsProduct::multiply(const vtkMomentsFactor & inputFactor) {
  for (auto & factor:m_product)
    if (factor.getIndex() == inputFactor.getIndex()) {
      factor.increaseFactorOrder();
      return;
    }

  // the product consists of factors with ranks in decending order
  for (auto it = m_product.cbegin(); it != m_product.cend(); ++it)
    if (inputFactor < (*it)) {
      m_product.insert(it, inputFactor);
      return;
    }
  
  m_product.push_back(inputFactor);

}


/**
 * take derivative of a monomial with respect to a moment
 * e.g. d 2*M^3 / dM = (2*3) * M^(3-1)
 * @param index the variable M
 * @return whether or not the product contains the varible M
 */
bool vtkMomentsProduct::takeDerivative(const vtkMomentsIndex & index) {
  for (auto it = m_product.begin(); it != m_product.end(); ++it)
    if ((*it).getIndex() == index) {
      m_scalar *= (*it).getFactorOrder();
      
      if ((*it).decreaseFactorOrder()) 
	m_product.erase(it);
      
      return false;
    }
  
  // if no identical factor is found, the derivative of this product is 0
  return true;
}

/**
 * compute the value of the product
 * @param map a map whose key is vtkMomentsIndex and value is the value of the moment
 * @return the value of the product
 */
double vtkMomentsProduct::assignValue(const indexValueMap & map) const {
  double prod = (double)m_scalar;
  for (auto & factor:m_product)
    prod *= factor.assignValue(map);
  return prod;
}

bool operator == (const vtkMomentsProduct & product1, const vtkMomentsProduct & product2) {
  if (product1.m_product.size() != product2.m_product.size())
    return false;

  return equal(product1.m_product.cbegin(),product1.m_product.cend(),product2.m_product.cbegin());
}

bool operator < (const vtkMomentsProduct & product1, const vtkMomentsProduct & product2) {
  if (product1.m_product.size() < product2.m_product.size())
    return true;
  else if (product1.m_product.size() > product2.m_product.size())
    return false;
  else {
    auto it1 = product1.m_product.cbegin(), it2 = product2.m_product.cbegin();
    for(;it1 != product1.m_product.cend(); ++it1, ++it2) 
      if (*it1 < *it2)
  	return true;
      else if (*it1 == *it2)
  	continue;
      else
  	return false;
    return false;
  }
}

/**
 * output stream for printing
 */
ostream & operator <<(ostream & os, const vtkMomentsProduct & prod) {
  if (prod.m_product.size() == 0) {
    os << prod.m_scalar;
    return os;
  }
  
  if (prod.m_scalar > 1)
    os << prod.m_scalar;
  
  for (auto & factor:prod.m_product)
    os << factor;

  return os;
}

/**
 * encode to a string
 */
stringstream & operator << (stringstream & ofs, const vtkMomentsProduct & product) {
  ofs << product.m_scalar << " ";
  ofs << product.m_product;
  return ofs;
}

/**
 * compute the signature
 */
istream & operator >> (istream & ifs, vtkMomentsProduct & product) {
  ifs >> product.m_scalar;
  ifs >> product.m_product;
  return ifs;
}
