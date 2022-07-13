#include "vtkComputeMoments.h"
#include "vtkMomentsHelper.h"
//#include "vtkMomentsTensor.h"

#ifdef VTKM_AVAILABLE
#include "vtkmlib/ImageDataConverter.h"
#include <vtkm/filter/ComputeMoments.h>
#endif

#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>

#include <algorithm>
#include <vector>

namespace
{
#ifdef VTKM_AVAILABLE
struct ExtractComponentImpl
{
  template <typename T, typename S>
  void operator()(const vtkm::cont::ArrayHandle<T, S>& field,
                  const int *idx,
                  vtkDoubleArray* out) const
  {
    auto portal = field.ReadPortal();
    auto numComps = vtkm::VecTraits<T>::GetNumberOfComponents(portal.Get(0));

    vtkm::IdComponent compIdx = 0;
    switch (numComps)
    {
      case 1:
        compIdx = 0;
        break;
      case 2: case 3:
        compIdx = static_cast<vtkm::IdComponent>(*idx);
        break;
      case 4:
        compIdx = static_cast<vtkm::IdComponent>(idx[1] * 2 + idx[0]);
        break;
      case 6: case 9:
        compIdx = static_cast<vtkm::IdComponent>(idx[0] * 3 + idx[1]);
        break;
      default:
        std::cout << "execution shouldn't reach here\n";
        abort();
    }

    for (vtkm::Id i = 0; i < portal.GetNumberOfValues(); ++i)
    {
      out->SetValue(i, static_cast<double>(vtkm::VecTraits<T>::GetComponent(portal.Get(i), compIdx)));
    }
  }
};

void ExtractComponent(const vtkm::cont::Field& field, const int* idx, vtkDataArray* out)
{
  vtkm::cont::CastAndCall(
    field.GetData().ResetTypes<typename vtkm::filter::ComputeMoments::SupportedTypes>(),
    ExtractComponentImpl{},
    idx,
    vtkDoubleArray::SafeDownCast(out));
}
#endif
} // anonymous namespace

void vtkComputeMoments::ComputeVtkm(
  int radiusIndex, vtkImageData* grid, vtkImageData* field, vtkImageData* output)
{
  std::cout << "vtkComputeMoments::ComputeVtkm \n";

#ifdef VTKM_AVAILABLE
  int gridDims[3];
  grid->GetDimensions(gridDims);
  const bool is2D = gridDims[2] == 1;

  double fieldSpacing[3];
  field->GetSpacing(fieldSpacing);
//  vtkm::Vec<double, 3> spacing{fieldSpacing};
  vtkm::Vec<double, 3> spacing{fieldSpacing[0], fieldSpacing[1], fieldSpacing[2]};

  if (grid != field)
  {
    int fieldDims[3];
    double gridSpacing[3];
    grid->GetSpacing(gridSpacing);
    field->GetDimensions(fieldDims);

    for (int i = 0; i < 3; ++i)
    {
      if (gridDims[i] != fieldDims[i] || gridSpacing[i] != fieldSpacing[i])
      {
        vtkErrorMacro(<< "The structure of grid and field must be the same for VTK-m");
        return;
      }
    }
  }

  auto fieldArray = field->GetPointData()->GetArray(this->NameOfPointData.c_str());

  try
  {
    // convert the input dataset to a vtkm::cont::DataSet
    vtkm::cont::DataSet in = tovtkm::Convert(field);
    vtkm::cont::Field vtkmfield =
      tovtkm::Convert(fieldArray, vtkDataObject::FIELD_ASSOCIATION_POINTS);
    in.AddField(vtkmfield);
    //in.PrintSummary(std::cout);

    vtkm::filter::ComputeMoments computeMoments;
    computeMoments.SetOrder(this->Order);
    computeMoments.SetSpacing(spacing);
    computeMoments.SetRadius(this->Radii.at(static_cast<std::size_t>(radiusIndex)));
    computeMoments.SetActiveField(this->NameOfPointData);

    vtkm::cont::DataSet out = computeMoments.Execute(in);
    //out.PrintSummary(std::cout);

    std::vector<int> indices;
    for (int order = 0; order <= this->Order; ++order)
    {
      const int maxR = is2D ? 0 : order; // 2D grids don't use r
      for (int r = 0; r <= maxR; ++r)
      {
        const int qMax = order - r;
        for (int q = 0; q <= qMax; ++q)
        {
          const int p = order - r - q;

          indices.resize(static_cast<std::size_t>(order));

          // Fill indices according to pqr values:
          if (!indices.empty())
          {
            auto iter = indices.begin();
            iter = std::fill_n(iter, p, 0);
            iter = std::fill_n(iter, q, 1);
            iter = std::fill_n(iter, r, 2);
            assert(iter == indices.end());
          }

          auto vtkmFieldName = std::string("index");
          for (int i : indices)
          {
            vtkmFieldName += std::to_string(i);
          }

//          std::cerr << "Order: " << order << " "
//                    << "pqr: " << p << "x" << q << "x" << r << " "
//                    << "Field name: " << vtkmFieldName << "\n";

          vtkmfield = out.GetField(vtkmFieldName);

          auto numComps = static_cast<int>(std::pow(this->Dimension, this->FieldRank));
          for (int c = 0; c < numComps; ++c)
          {
            indices.resize(static_cast<std::size_t>(order));
            for (int i = 0; i < this->FieldRank; ++i)
            {
              indices.push_back((c / static_cast<int>(std::pow(this->Dimension, i))) % this->Dimension);
            }

            auto vtkFieldName =
                vtkMomentsHelper::getFieldNameFromTensorIndices(
                  this->Radii.at(static_cast<std::size_t>(radiusIndex)), indices, this->FieldRank);
            ExtractComponent(vtkmfield,
                             indices.data() + order,
                             output->GetPointData()->GetArray(vtkFieldName.c_str()));
          }
        }
      }
    }
  }
  catch (const vtkm::cont::Error& e)
  {
    vtkErrorMacro(<< "VTK-m error: " << e.GetMessage());
    return;
  }
#else
  vtkErrorMacro(<< "Please enable AcceleratorVTKm to use GPU");
#endif
}
