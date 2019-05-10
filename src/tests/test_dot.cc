#define BOOST_TEST_MODULE eigen_cuda
#include <boost/test/included/unit_test.hpp>
#include "eigencuda.hpp"


using eigencuda::Mat;

BOOST_AUTO_TEST_CASE(dot_product) {
eigencuda::EigenCuda<float> EC;
Mat<float> A = Mat<float>::Zero(2, 2);
Mat<float> B = Mat<float>::Zero(2, 2);

 A << 1., 2.,
      3., 4.; 
 B << 5., 6.,
      7., 8.;

 Mat<float> C = EC.dot(A, B);
 BOOST_CHECK_EQUAL(C.sum(), 134.);
}
