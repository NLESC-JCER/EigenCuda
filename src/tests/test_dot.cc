#define BOOST_TEST_MODULE eigen_cuda
#include "eigencuda.hpp"
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(dot_product) {
  eigencuda::EigenCuda EC;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2, 2);

  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8.;

  Eigen::MatrixXd C = EC.matrix_mult(A, B);
  BOOST_CHECK_EQUAL(C.sum(), 134.);
}

BOOST_AUTO_TEST_CASE(right_matrix_multiplication) {
  // Call the class to handle GPU resources
  eigencuda::EigenCuda EC;

  // Call matrix multiplication GPU
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(3, 2);

  // Define matrices
  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8., 9., 10.;
  C << 9., 10., 11., 12., 13., 14.;
  D << 13., 14., 15., 16., 17., 18.;
  X << 23., 34., 31., 46., 39., 58.;
  Y << 39., 58., 47., 70., 55., 82.;
  Z << 55., 82., 63., 94., 71., 106.;

  std::vector<Eigen::MatrixXd> tensor{B, C, D};
  EC.right_matrix_tensor_mult(std::move(tensor), A);

  // Expected results
  BOOST_TEST(X.isApprox(tensor[0]));
  BOOST_TEST(Y.isApprox(tensor[1]));
  BOOST_TEST(Z.isApprox(tensor[2]));
}

BOOST_AUTO_TEST_CASE(wrong_shape_cublas) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(5, 5);

  eigencuda::EigenCuda EC;
  std::vector<Eigen::MatrixXd> tensor{B};
  try {
    EC.right_matrix_tensor_mult(std::move(tensor), A);
  } catch (const std::runtime_error& error) {
    std::string error_msg = error.what();
    std::string reason = "an illegal value";
    if (error_msg.find(reason) != std::string::npos) {
      auto eptr = std::current_exception();
      std::rethrow_exception(eptr);
    }
  }
}
