#include "vtkMomentsIndex.h"

/**
 * default constructor
 */
vtkMomentsIndex::vtkMomentsIndex()
  : m_momentRank(0)
  , m_fieldRank(0)
  , m_momentIndex(0)
  , m_momentIndices(0)
  , m_fieldIndices(0)
  , m_momentSignature(0)
  , m_fieldSignature(0) {}

/**
 * The baseIndex for moment M_{001} is "001". It is not sorted. 
 * It could be "010", which is a duplicate of "001".
 * @param dimension e.g. 2D or 3D
 * @param momentRank Order of the moment tensor
 * @param fieldRank 0 for scalars, 1 for vectors, 3 for matrices
 * @param baseIndex e.g. "001"
 */
vtkMomentsIndex::vtkMomentsIndex(unsigned dimension, unsigned momentRank, unsigned fieldRank, const vector<unsigned> & baseIndex)
  : m_momentRank(momentRank)
  , m_fieldRank(fieldRank)
  , m_momentIndex(dimension, 0)
  , m_momentSignature(0)
  , m_fieldSignature(0)
{
  for (unsigned i = 0; i < momentRank; i++)
    m_momentIndex[baseIndex[i]]++;

  computeMomentIndices();
  m_fieldIndices = vector<int>(fieldRank);

  for (unsigned i = 0; i < fieldRank; i++)
    m_fieldIndices[i] = (int)baseIndex[i + momentRank];
  
  computeSignature();
}

/**
 * copy contructor
 */
vtkMomentsIndex::vtkMomentsIndex(const vtkMomentsIndex & index)
  : m_momentRank(index.m_momentRank)
  , m_fieldRank(index.m_fieldRank)
  , m_momentIndex(index.m_momentIndex)
  , m_momentIndices(index.m_momentIndices)
  , m_fieldIndices(index.m_fieldIndices)
  , m_momentSignature(index.m_momentSignature)  
  , m_fieldSignature(index.m_fieldSignature)  
{}

/**
 * get a concatenated indices of m_momentIndices and m_fieldIndices
 */
vector<int> vtkMomentsIndex::getIndices() const {
  vector<int> indices = m_momentIndices;
  indices.insert(indices.end(), m_fieldIndices.begin(), m_fieldIndices.end());
  return indices;
}

/**
 * compute momentIndices from momentIndex. It thus is sorted.
 */
void vtkMomentsIndex::computeMomentIndices() {
  if (m_momentIndices.empty()) {
    m_momentIndices = vector<int>(getMomentRank());
    unsigned k = 0;
  
    for (int i = 0; i < m_momentIndex.size(); i++) 
      for (int j = 0; j < m_momentIndex[i]; j++)
	m_momentIndices[k++] = i;  
  }
}

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
list<vtkMomentsIndex> vtkMomentsIndex::generateVariables(unsigned dimension, unsigned momentRank, unsigned fieldRank) {
  list<vtkMomentsIndex> variables;
  unsigned rank = momentRank + fieldRank;
  unsigned maxValue = (unsigned)pow((double)dimension,(double)rank);
  
  for (unsigned i = 0; i < maxValue; i++) {
    vector<unsigned> baseIndex = vector<unsigned>(rank);
    unsigned rem = i,x;
    
    for (int j = rank - 1; j >= 0; j--) {
      x = (unsigned)pow((double)dimension, (double)j);

      baseIndex[j] = rem / x;
      rem = rem % x;
    }

    vtkMomentsIndex index(dimension, momentRank, fieldRank, baseIndex);
    
    auto it = variables.cbegin();
    while (true)
      if (it == variables.cend() ||
	  index < *it) {
	variables.insert(it, index);
	break;
      }
      else if (index == *it)
	break;
      else
	++it;  
  }

  return variables;
}

bool operator == (const vtkMomentsIndex & index1, const vtkMomentsIndex & index2) {
  return (index1.m_momentRank == index2.m_momentRank) && 
    (index1.m_fieldRank == index2.m_fieldRank) &&
    (index1.m_momentSignature == index2.m_momentSignature) &&
    (index1.m_fieldSignature == index2.m_fieldSignature);
}

bool operator < (const vtkMomentsIndex & index1, const vtkMomentsIndex & index2) {
  unsigned rank1 = index1.m_momentRank + index1.m_fieldRank,
    rank2 = index2.m_momentRank + index2.m_fieldRank;

  return (rank1 < rank2) ||
    ((rank1 == rank2) &&
     (index1.m_momentSignature < index2.m_momentSignature)) ||
    ((rank1 == rank2) &&
     (index1.m_momentSignature == index2.m_momentSignature) && 
     (index1.m_fieldSignature < index2.m_fieldSignature)) ;
}

/**
 * output stream for printing
 */
ostream & operator << (ostream & os, const vtkMomentsIndex & index) {
  if (index.getRank() == 0) 
    os << "M";
  else {
    os << "M_{";
  
    for (unsigned i = 0; i < index.getDimension(); i++) 
      for (unsigned j = 0; j < index.m_momentIndex[i]; j++)
	os << i;

    for (unsigned i = 0; i < index.m_fieldIndices.size(); i++)
      os << index.m_fieldIndices[i];
 
    os << "}";
  }
  return os;
}

/**
 * encode to a string
 */
stringstream & operator << (stringstream & ofs, const vtkMomentsIndex & index) {
  ofs << index.m_momentRank << " " << index.m_fieldRank << " ";  
  ofs << index.m_momentIndex << " ";
  ofs << index.m_fieldIndices;
  return ofs;
}

/**
 * decode from a stream
 */
istream & operator >> (istream & ifs, vtkMomentsIndex & index) {
  ifs >> index.m_momentRank >> index.m_fieldRank;
  ifs >> index.m_momentIndex;
  ifs >> index.m_fieldIndices;
  index.computeMomentIndices();
  index.computeSignature();
  return ifs;
}


