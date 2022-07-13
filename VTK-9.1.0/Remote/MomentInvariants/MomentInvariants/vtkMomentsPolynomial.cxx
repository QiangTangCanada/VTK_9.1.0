#include "vtkMomentsPolynomial.h"

/**
 * default constructor
 */
vtkMomentsPolynomial::vtkMomentsPolynomial() {}

/**
 * constructor
 * @param dimension e.g. 2 or 3
 * @param fieldRank Fieldrank of the tensor is 0 for scalars, 1 for vectors, 3 for matrices
 * @param productInfo optional outer tensor product information that produced it
 * @param contractionInfo optional contraction information that produced it
 */
vtkMomentsPolynomial::vtkMomentsPolynomial(unsigned dimension, unsigned fieldRank, const list<unsigned> & productInfo, const list<unsigned> & contractionInfo)
  : m_productInfo(productInfo)
  , m_contractionInfo(contractionInfo) {  
  vtkMomentsPolynomialHelper(dimension, fieldRank, productInfo, contractionInfo);
}

/**
 * constructor that uses a tensor 
 * @param tensor a tensor that will be converted to a polynomial
 */
vtkMomentsPolynomial::vtkMomentsPolynomial(const vtkMomentsTensorSimple & tensor)
  : m_productInfo(tensor.getProductInfo())
  , m_contractionInfo(tensor.getContractionInfo()){
  vtkMomentsPolynomialHelper(tensor.getDimension(), tensor.getFieldRank() / tensor.getProductInfo().size() , tensor.getProductInfo(), tensor.getContractionInfo());
}

/**
 * copy constructor
 */
vtkMomentsPolynomial::vtkMomentsPolynomial(const vtkMomentsPolynomial & poly)
  : m_polynomial(poly.m_polynomial)
  , m_productInfo(poly.getProductInfo())
  , m_contractionInfo(poly.getContractionInfo())
  , m_contractionInfoAbs(poly.m_contractionInfoAbs) {}

/**
 * This function constructs a polynomail from both m_productInfo and m_contractionInfo.
 * @param dimension e.g. 2 or 3
 * @param fieldRank Fieldrank of the tensor is 0 for scalars, 1 for vectors, 3 for matrices
 * @param productInfo optional outer tensor product information that produced it
 * @param contractionInfo optional contraction information that produced it
 */
void vtkMomentsPolynomial::vtkMomentsPolynomialHelper(unsigned dimension, unsigned fieldRank, const list<unsigned> & productInfo, const list<unsigned> & contractionInfo) {
  
  unsigned rank = 0;
  for (auto x:productInfo)
    rank += x;

  unsigned maximumIndexValue = (unsigned)pow((double)dimension, (double)rank) - 1;

  if (rank % 2 != 0) {
    cerr << "vtkMomentsTensorPolynomial: rank " << rank << " is not even.\n";
    
    exit(0);
  }

  // initialize indexList {0,1,2 ...}
  forward_list<unsigned> indexList;
  for (int i = rank - 1; i >= 0; i--) 
    indexList.push_front((unsigned)i);

  //store the power of dimension for future use
  vector<unsigned> powDimN(rank / 2);
  for (unsigned i = 0; i < powDimN.size(); i++) 
    powDimN[i] = (unsigned)pow((double)dimension, (double)i);

  unsigned indicesNum = (unsigned)pow((double)dimension, (double)powDimN.size());;
  
  //initialize valueIndex: {00,10,01,11}
  forward_list<vector<unsigned>> valueIndex;
  
  forward_list<vector<unsigned>> tensorIndices;
  
  for (unsigned i = 0; i < indicesNum; i++) {    
    vector<unsigned> index(powDimN.size());
    unsigned rem = i;  
    vector<unsigned>::const_reverse_iterator ritPow = powDimN.crbegin();
  
    for (auto ritIndex = index.rbegin(); ritIndex != index.rend(); ++ritIndex,++ritPow) {    
      *ritIndex = rem / *ritPow;
      rem = rem % *ritPow;
    }
    
    valueIndex.push_front(index);
    tensorIndices.push_front(vector<unsigned>(rank));
  }

  //reconstruct tensorIndices from constractionInfo
  unsigned j, k;
  list<unsigned>::const_iterator citCI = contractionInfo.cbegin();
  forward_list<unsigned>::iterator itIL;
  for (unsigned i = 0; i < powDimN.size(); i++) {
    itIL = indexList.before_begin();
    advance(itIL,(unsigned)*(citCI++));
    j = *next(itIL);
    indexList.erase_after(itIL);

    itIL = indexList.before_begin();
    advance(itIL,(unsigned)*(citCI++) - 1);
    k = *next(itIL);
    indexList.erase_after(itIL);
    
    auto itVI = valueIndex.cbegin();
    auto itTI = tensorIndices.begin();

    for (; itVI != valueIndex.cend(); ++itVI, ++itTI) {
      (*itTI).at(j) = (*itVI).at(i);
      (*itTI).at(k) = (*itVI).at(i);
    }
  }

  //compute poly from tensorIndices
  for (auto & idx:tensorIndices) {
    auto it = idx.cbegin();
    vtkMomentsProduct product;

    for (auto n:productInfo) {
      auto vec = vtkMomentsFactor(dimension, n - fieldRank, fieldRank, vector<unsigned>(it,it + n));
      product.multiply(vec);
      advance(it,n);
    }

    product.computeSignature();
    addProduct(product);
  }

  m_contractionInfoAbs = reconstructContractions();
}

/**
 * add a product to the current polynomial
 * @param prod add a product to the polynomial
 */
void vtkMomentsPolynomial::addProduct(const vtkMomentsProduct & prod) {
  for (auto it = m_polynomial.begin(); it != m_polynomial.end(); ++it)
    if (prod < *(it)) {
      m_polynomial.insert(it,prod);
      return;
    }
    else if ((*(it)).add(prod)) 
      return;

  m_polynomial.push_back(prod);
}

/**
 * take derivative of a polynomial with respect to a moment
 * @param index a moment
 */
void vtkMomentsPolynomial::takeDerivative(const vtkMomentsIndex & index) {
  for (auto it = m_polynomial.begin(); it != m_polynomial.end();)  {
    if ((*it).takeDerivative(index))
      it = m_polynomial.erase(it);
    else
      ++it;
  }
}

/**
 * compute the value of a tensor
 * @param map values of moments
 * @return the value of the polynomial
 */
double vtkMomentsPolynomial::assignValue(const indexValueMap & map) const {
  if (map.size() == 0)
    cerr << "vtkMomentsPolynomial::assignValue: index and value map is empty.\n";
  double s = 0;
  for (auto & prod:m_polynomial)
    s += prod.assignValue(map);
  return s; 
}

/**
 * compute the value of the tensor, then normalize the value base on the number of factor
 * e.g. (1M2M3M)^(1/3) because there are 3 moments
 * @param map values of moments
 * @return the normalized value of the polynomial
 */
double vtkMomentsPolynomial::assignValueAndNormalize(const indexValueMap & map) const {
  if (map.size() == 0)
    cerr << "vtkMomentsPolynomial::assignValueAndNormalize: index and value map is empty.\n";
  double s = 0;
  for (auto & prod:m_polynomial)
    s += prod.assignValue(map);
  if (s >= 0)
    return pow(s, 1 / (double)getTotalFactorOrder());
  else
    return -pow(-s, 1 / (double)getTotalFactorOrder());
}

/**
 * compute the contractionInfo. Note that the indices in m_contractionInfo are relative indices
 * For example, m_contractionInfo = "0201" means (0,2), (1,3)
 * @return the absolute indices
 */
vector<pair<unsigned,unsigned>> vtkMomentsPolynomial::reconstructContractions() const {
  vector<pair<unsigned,unsigned>> pairs(m_contractionInfo.size() / 2);  
  forward_list<unsigned> indexList;
  unsigned rank = accumulate(m_productInfo.cbegin(),m_productInfo.cend(),0);
  for (int i = rank - 1; i >= 0; i--) 
    indexList.push_front((unsigned)i);
  
  auto citCI = m_contractionInfo.cbegin();
  forward_list<unsigned>::const_iterator itIL;
  for (auto it = pairs.begin(); it != pairs.end(); ++it) {
    itIL = indexList.cbefore_begin();
    advance(itIL,*(citCI++));
    (*it).first = *next(itIL);
    indexList.erase_after(itIL);

    itIL = indexList.cbefore_begin();
    advance(itIL,*(citCI++) - 1);
    (*it).second = *next(itIL);
    indexList.erase_after(itIL);    
  }

  return pairs;
}

/**
 * Print the 1M2M3M style representation in Latex format of the tensor.
 * The numbers represent order instead of rank. Thus there is no explicit field rank information.
 */
string vtkMomentsPolynomial::printTensor() const {
  if (!m_productInfo.empty()) {
    stringstream ss;
    unsigned n = 1;
    unsigned fr = m_polynomial.front().getProduct().front().getIndex().getFieldRank();

    for (auto itProdInfo = m_productInfo.cbegin(); itProdInfo != m_productInfo.cend(); ++itProdInfo) 
      if (itProdInfo != prev(m_productInfo.cend()) && *itProdInfo == *next(itProdInfo)) 
	n++;
      else {
	ss << "\\tensor[^" << (*itProdInfo - fr) << "]{M}{^" << n << "}";
	n = 1;
      }
  
    if (!m_contractionInfoAbs.empty()){
      ss << "_{";
      ss << "(" << m_contractionInfoAbs.front().first << "," << m_contractionInfoAbs.front().second << ")";
      for (auto it = next(m_contractionInfoAbs.cbegin()); it != m_contractionInfoAbs.cend(); ++it) 
	ss << ",(" << (*it).first << "," << (*it).second << ")";
      ss << "}";    
    }

    return ss.str();
  }

  return "";
}

/**
 * print the 1M2M3M style representation of the tensor
 * The numbers represent order instead of rank. Thus there is no explicit field rank information.
 */
string vtkMomentsPolynomial::printTensorConcise() const {
  if (!m_productInfo.empty()) {
    stringstream ss;
    unsigned n = 1;
    unsigned fr = m_polynomial.front().getProduct().front().getIndex().getFieldRank();

    for (auto itProdInfo = m_productInfo.cbegin(); itProdInfo != m_productInfo.cend(); ++itProdInfo) 
      if (itProdInfo != prev(m_productInfo.cend()) && *itProdInfo == *next(itProdInfo)) 
	n++;
      else {
	ss << (*itProdInfo - fr) << "M" << n;
	n = 1;
      }
  
    for (auto pair : m_contractionInfoAbs)
      ss << "_" << pair.first << pair.second;

    return ss.str();
  }

  return "";
}

/**
 * assign operator
 */
vtkMomentsPolynomial & vtkMomentsPolynomial::operator = (const vtkMomentsPolynomial & poly) {
  if (this == &poly)
    return *this;
  m_polynomial = poly.m_polynomial;
  m_productInfo = poly.m_productInfo;
  m_contractionInfo = poly.m_contractionInfo;
  m_contractionInfoAbs = poly.m_contractionInfoAbs;
  return *this;
}

bool operator == (const vtkMomentsPolynomial & poly1, const vtkMomentsPolynomial & poly2) {
  if (poly1.m_polynomial.size() != poly2.m_polynomial.size())
    return false;

  return equal(poly1.m_polynomial.cbegin(), poly1.m_polynomial.cend(), poly2.m_polynomial.cbegin());
}

bool operator < (const vtkMomentsPolynomial & poly1, const vtkMomentsPolynomial & poly2) {
  if (poly1.m_polynomial.size() < poly2.m_polynomial.size())
    return true;
  else if (poly1.m_polynomial.size() > poly2.m_polynomial.size())
    return false;
  else {
    auto it1 = poly1.m_polynomial.cbegin(), it2 = poly2.m_polynomial.cbegin();
    for(;it1 != poly1.m_polynomial.cend(); ++it1, ++it2) 
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
ostream & operator <<(ostream & os, const vtkMomentsPolynomial & poly) {  
  const list<vtkMomentsProduct> & prodList = poly.m_polynomial;

  if (prodList.empty()) {
    os << 0;
    return os;
  }

  os << prodList.front();

  unsigned i = 0;
  for (auto it = next(prodList.cbegin()); it != prodList.cend(); ++it, i++) 
    if (i == 3) {
      i = 0;
      os << "\\\\\n &+ " << *it;
    }
    else
      os << " + " << *it;

  return os;
}

/**
 * encode to a string
 */
stringstream & operator << (stringstream & ofs, const vtkMomentsPolynomial & polynomial) {
  ofs << polynomial.getPolynomialConst().front().getProduct().front().getIndex().getDimension() << " ";
  ofs << polynomial.getPolynomialConst().front().getProduct().front().getIndex().getFieldRank() << " ";
  ofs << polynomial.getProductInfo() << " ";
  ofs << polynomial.getContractionInfo() << " ";
  return ofs;
}

/**
 * decode from a stream
 */
istream & operator >> (istream & ifs, vtkMomentsPolynomial & polynomial) {
  unsigned dimension, fieldRank;
  list<unsigned> productInfo,contractionInfo;
  ifs >> dimension;
  ifs >> fieldRank;
  ifs >> productInfo;
  ifs >> contractionInfo;
  polynomial = vtkMomentsPolynomial(dimension, fieldRank, productInfo, contractionInfo);
  
  return ifs;
}


