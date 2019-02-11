#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {
  F_ = MatrixXd(4,4);
  // set the process covariance matrix Q
  Q_ = MatrixXd(4, 4);
  // set the initial state covariance matrix P
  P_ = MatrixXd(4, 4);
  
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * DONE: predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * DONE: update the state by using Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

double NormalizeAngle(double phi)
{
  //  const double Max =  M_PI;
  //  const double Min = -M_PI;
  //  return phi < Min  ? Max + std::fmod(phi - Min, Max - Min)
  //                    : std::fmod(phi - Min, Max - Min) + Min;
  // Normalize the angle
  while (phi > M_PI)  { phi -= M_PI; }
  while (phi < -M_PI) { phi += M_PI; }
  return phi;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * DONE: update the state by using Extended Kalman Filter equations
   */
  double rho = sqrt(x_[0]*x_[0] + x_[1]*x_[1]);
  double phi = atan2(x_[1], x_[0]);
  double rho_dot;
  if(fabs(rho)<0.00001)
    rho_dot = 0.0;
  else
    rho_dot = (x_[0]*x_[2]+x_[1]+x_[3])/rho;
  
  //VectorXd z_pred = H_ * x_; // H was set to jacobian Hj
  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;
  // now normalize angle of difference vector y to [-pi,pi] (because atan assumes this range)
  y[1] = NormalizeAngle(y[1]);
  
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}
