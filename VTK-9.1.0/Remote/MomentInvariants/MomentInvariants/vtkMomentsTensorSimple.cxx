#include "vtkMomentsTensorSimple.h"


/**
 * defalut constructor
 */
vtkMomentsTensorSimple::vtkMomentsTensorSimple()
  : m_dimension(0)
  , m_rank(0)
  , m_fieldRank(0)
  , m_momentRank(0) {}

/**
 * default contructor for a paticular dimension
 * @param dimension i.e. 2 or 3
 */
vtkMomentsTensorSimple::vtkMomentsTensorSimple(unsigned dimension)
  : m_dimension(dimension)
  , m_rank(0)
  , m_fieldRank(0)
  , m_momentRank(0) {}

/**
 * constructor 
 * @param dimension i.e. 2 or 3
 * @param rank rank of the tensor, i.e. number of indices to reference its entries
 * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
 */
vtkMomentsTensorSimple::vtkMomentsTensorSimple(unsigned dimension, unsigned rank, unsigned fieldRank)
  : m_dimension(dimension)
  , m_rank(rank)
  , m_fieldRank(fieldRank)
  , m_momentRank(rank - fieldRank)
  , m_productInfo(1, rank) {
  if (rank < fieldRank) {
    cerr << "The rank must be at least as big as the fieldrank. rank = " << rank << " fieldRank = " << fieldRank << endl;
    exit(0);
  }
}

/**
 * copy constructor
 */
vtkMomentsTensorSimple::vtkMomentsTensorSimple(const vtkMomentsTensorSimple & tensor)
  : m_dimension(tensor.getDimension())
  , m_rank(tensor.getRank())
  , m_fieldRank(tensor.getFieldRank())
  , m_momentRank(tensor.getMomentRank())
  , m_contractionInfo(tensor.getContractionInfo())
  , m_productInfo(tensor.getProductInfo()) {}

/**
 * copy constructor. It computes a contraction after the input is copied.
 * @param i index for contraction
 * @param j index for contraction 
 */
vtkMomentsTensorSimple::vtkMomentsTensorSimple(const vtkMomentsTensorSimple & tensor, unsigned i, unsigned j)
  : m_dimension(tensor.getDimension())
  , m_rank(tensor.getRank())
  , m_fieldRank(tensor.getFieldRank())
  , m_momentRank(tensor.getMomentRank())
  , m_contractionInfo(tensor.getContractionInfo())
  , m_productInfo(tensor.getProductInfo()) {
  contract(i,j);
}

/**
 * constructor: rank = fieldRank + momentRank
 * @param dimension i.e. 2 or 3
 * @param rank rank of the tensor, i.e. number of indices to reference its entries
 * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
 * @param momentRank Order of the moment tensor
 */
vtkMomentsTensorSimple::vtkMomentsTensorSimple(unsigned dimension, unsigned rank, unsigned fieldRank, unsigned momentRank)
  : m_dimension(dimension)
  , m_rank(rank)
  , m_fieldRank(fieldRank)
  , m_momentRank(momentRank)
  , m_productInfo(1,rank) {}

/**
 * multiply a tensor with another tensor.
 * the rank of the result is the sum of the ranks of its parents
 * @param tensor the other thensor
 */
void vtkMomentsTensorSimple::tensorProduct(const vtkMomentsTensorSimple & tensor) {
  if (m_dimension != tensor.getDimension()) {
    cerr << "only tensor with the same dimension can be multiplied." << endl;
    exit(0);
  }

  m_rank += tensor.getRank();
  m_fieldRank += tensor.getFieldRank();
  concatProductInfo(tensor.getProductInfo());

}

/**
 * compute the power of the tensor.
 * @param exp the exponent 
 */
void vtkMomentsTensorSimple::tensorPow(unsigned exp) {
  if (exp == 0) {
    cerr << "vtkMomentsTensorSimple::pow: exp == 0\n";
    exit(0);
  }

  vector<vtkMomentsTensorSimple> vec = vector<vtkMomentsTensorSimple>(exp - 1, *this);

  for (unsigned i = 0; i < exp - 1; i++)
    tensorProduct(vec[i]);

}

/**
 * This function produces a tensor contraction of the indices i and j
 * @param i index for contraction
 * @param j index for contraction
 * @return the contracted tensor. its rank is two lower than the rank of its parent
 */
void vtkMomentsTensorSimple::contract(unsigned i, unsigned j) {
  if (m_rank < 2) {
      cerr << "rank too small for contraction." << endl;
      exit(0);
  }
  if (i >= j) {
    cerr << "indices for contracion are not asending." << endl;
    exit(0);
  }

  setContractionInfo(i, j);
  m_rank -= 2;
}

/** 
 * This function creates a list of contractions, all of which are rank 0.
 * @param tensorList the list that the contractions will append to
 */ 
void vtkMomentsTensorSimple::contractAllRank0(list<vtkMomentsTensorSimple> & tensorList) {
  for (auto it = tensorList.begin(); it != tensorList.end();) {
    if ((*it).getRank() == 0)
      ++it;
    else {
      (*it).contractAll(tensorList);
      it = tensorList.erase(it);
    }
  }
}

/**
 * inverse function to getIndex
 * @param index the place in the flat c++ std vector
 * @return vector of tensor indices that identify an entry
 */
vector<int> vtkMomentsTensorSimple::getIndices(int index) const {
  vector<int> indices(m_rank, 0);
  for (int i = 0; i < m_rank; ++i)
    indices.at(i) = (int(int(index) / int(pow((double)m_dimension, (double)i)))) % int(m_dimension);

  return indices;
}

/**
 * the moment tensors have two types of indices fieldIndices and momentIndices
 * fieldIndices of length fieldRank refer to the components of the original data (3 for vector, 9
 * for matrix) momentIndices of length momentRank refer to the basis function
 * @param index the place in the flat c++ std vector
 * @return the indices that correspond to the basis function
 */
vector<int> vtkMomentsTensorSimple::getMomentIndices(int index) const {
  vector<int> indices(m_momentRank, 0);
  for (int i = 0; i < m_momentRank; ++i)
    indices.at(i) = (int(int(index) / int(pow((double)m_dimension, (double)i)))) % int(m_dimension);

  return indices;
}

/**
 * This function computes the order of two tensors.
 */
bool vtkMomentsTensorSimple::compare(const vtkMomentsTensorSimple & tensor1,const vtkMomentsTensorSimple & tensor2) {
  if (tensor1.m_productInfo.size() == tensor2.m_productInfo.size()) {
    list<unsigned> productInfo1 = tensor1.m_productInfo;
    list<unsigned> productInfo2 = tensor2.m_productInfo;
    productInfo1.sort();
    productInfo2.sort();
    // auto it1 = tensor1.m_productInfo.cbegin(), it2 = tensor2.m_productInfo.cbegin();
    // while (it1 != tensor1.m_productInfo.cend()) {
    auto it1 = productInfo1.cbegin(), it2 = productInfo2.cbegin();
    while(it1 != productInfo1.cend()) {
      if (*it1 == *it2) {
	++it1;
	++it2;
      }
      else
	return *it1 < *it2;
    }
  }
  else
    return tensor1.m_productInfo.size() < tensor2.m_productInfo.size();

  return false;
}

bool operator == (const vtkMomentsTensorSimple & tensor1, const vtkMomentsTensorSimple & tensor2) {
  if (tensor1.getRank() != tensor2.getRank() || tensor1.getProductInfo().size() != tensor2.getProductInfo().size() || tensor1.getContractionInfo().size() != tensor2.getContractionInfo().size())
      return false;

    return equal(tensor1.getProductInfo().cbegin(),tensor1.getProductInfo().cend(),tensor2.getProductInfo().cbegin()) && equal(tensor1.getContractionInfo().cbegin(),tensor1.getContractionInfo().cend(),tensor2.getContractionInfo().cbegin());
}
