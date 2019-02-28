#include "tools.h"

using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * DONE: Calculate the RMSE here.
   */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // DONE: YOUR CODE HERE
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if(0 == estimations.size())
      return rmse;
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size())
      return rmse;

  // DONE: accumulate squared residuals
  for (int i=0; i < estimations.size(); ++i) {
    // ... your code here
    VectorXd tmpvec = estimations[i] - ground_truth[i];
    VectorXd tmpvec2 = tmpvec.array() * tmpvec.array();
    rmse = rmse + tmpvec2;
  }

  // DONE: calculate the mean
  rmse = rmse / estimations.size();
  // DONE: calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;

}