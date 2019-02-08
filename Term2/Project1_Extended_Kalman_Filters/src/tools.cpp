#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
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

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * DONE:
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // DONE: YOUR CODE HERE

  // check division by zero
  static float eps = 0.0000001;
  if( fabs(px)<eps && fabs(py)<eps )
  {
    Hj.setZero();
    return Hj;
  }
  
  // compute the Jacobian matrix
  float px2 = px*px;
  float py2 = py*py;

  Hj << px/(sqrt(px2+py2)), py/(sqrt(px2+py2)), 0, 0,
        -py/(px2+py2), px/(px2+py2), 0, 0,
        py*(vx*py-vy*px)/(pow(px2+py2, 1.5)), px*(vy*px-vx*py)/(pow(px2+py2, 1.5)),
        px/(sqrt(px2+py2)), py/(sqrt(px2+py2));

  return Hj;
}
