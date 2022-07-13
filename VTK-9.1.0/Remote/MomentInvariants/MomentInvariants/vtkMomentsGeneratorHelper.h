/**
 * vtkmomentsGeneratorHelper provides input and output templates for 
 * encoding and decoding moment invariants.
 */
#ifndef VTKMOMENTSGENERATORHELPER_H
#define VTKMOMENTSGENERATORHELPER_H
#ifndef __VTK_WRAP__

#include <list>
#include <vector>
#include <sstream>

using namespace std;

/**
 * output a list to stringstream.
 */
template <class T>
stringstream & operator << (stringstream & ofs, const list<T> & input) {    
  ofs << input.size() << " ";
  for (auto & x:input) {    
    ofs << x;
    ofs << " ";
  }

  return ofs;
}

/**
 * output a vector to stringstream.
 */
template <class T>
stringstream & operator << (stringstream & ofs, const vector<T> & input) {
  ofs << input.size() << " ";
  for (auto & x:input) {    
    ofs << x;
    ofs << " ";
  }

  return ofs;
}

/**
 * input stream to a list.
 */
template <class T>
istream & operator >> (istream & ifs, list<T> & output) {
  unsigned n;
  ifs >> n;
  output.clear();
  
  for (unsigned i = 0; i < n; i++) {
    T x;
    ifs >> x;
    output.push_back(x);
  }

  return ifs;
}

/**
 * input stream to a vector
 */
template <class T>
istream & operator >> (istream & ifs, vector<T> & output) {
  unsigned n;
  ifs >> n;
  output.resize(n);
  
  for (unsigned i = 0; i < n; i++) 
    ifs >> output[i];

  return ifs;
}

#endif // __VTK_WRAP__
#endif // VTKMOMENTSGENERATORHELPER_H
