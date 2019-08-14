#define BOOST_TEST_MODULE eigen_cuda
#include "eigencuda.hpp"
#include <boost/test/included/unit_test.hpp>
#include <vector>
using eigencuda::Mat;

BOOST_AUTO_TEST_CASE(triple_tensor_product) {
  eigencuda::EigenCuda<double> EC;
  Mat<double> A = Mat<double>::Zero(2, 2);
  Mat<double> B = Mat<double>::Zero(2, 2);
  Mat<double> C = Mat<double>::Zero(2, 2);
  Mat<double> D = Mat<double>::Zero(2, 2);

  // Matrices declaration
  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8.;
  C << 9., 10., 11., 12.;
  D << 13., 14., 15., 16.;

  std::vector<Mat<double>> tensor{C, D};
  Mat<double> rs = EC.triple_tensor_product(A, B, tensor);

  BOOST_CHECK_EQUAL(rs.col(0).sum(), 2854.);
  BOOST_CHECK_EQUAL(rs.col(1).sum(), 3894.);
}
