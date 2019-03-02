#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0     , 0.0225;

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0;


  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0; // DONE hint from the video: change this

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4.0; // DONE hint from the video: change this
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  // set state dimension
  n_x_ = 5;

  // set augmented dimension
  n_aug_ = 7;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * DONE: Initialize the state x_ with the first measurement.
     * DONE: Create the covariance matrix.
     * Convert radar from polar to cartesian coordinates.
     */
    // first measurement
    cout << "UKF: " << endl;
    x_ << 1, 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // DONE: Convert radar from polar to cartesian coordinates
      //         and initialize state.
      previous_t = measurement_pack.timestamp_; // get current timestamp

      double rho     = measurement_pack.raw_measurements_[0];
      double phi     = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];
      x_ << 0.0,
            0.0,
            0.0,
            0.0,
            0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      x_ << measurement_pack.raw_measurements_[0],
            measurement_pack.raw_measurements_[1],
            0.0,
            0.0,
            0.0;
    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_t = measurement_pack.timestamp_; // get current timestamp
    return; // don't do anything else in case of initialization
  }

  /**
   * control structure similar to EKF project
   */
  double delta_t = (measurement_pack.timestamp_ - previous_t) / 1000000.0;
  Prediction(delta_t);

  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(measurement_pack);
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(measurement_pack);
  }
  previous_t = measurement_pack.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented sigma points
  // Lesson 7, section 18: Augmentation Assignment 2

  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  // Predict Sigma Points
  // Lesson 7, section 21: Sigma Point Prediction Assignment 2

  // init matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // predict sigma points
  for (int i = 0; i< 2 * n_aug_ + 1; i++) {
    // extract for better visibility
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw) );
      py_p = p_y + v/yawd * ( cos (yaw) - cos(yaw + yawd*delta_t) );
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p    = v;
    double yaw_p  = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p   += 0.5 * nu_a * delta_t*delta_t * cos(yaw);
    py_p   += 0.5 * nu_a * delta_t*delta_t * sin(yaw);
    v_p    +=       nu_a * delta_t;

    yaw_p  += 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p += nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  // Predict mean and covariance
  // Lesson 7, section 24: Predicted Mean and Covariance Assignment 2

  // create vector for weights
  weights_ = VectorXd(2*n_aug_+1);

  // set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for(int i = 1; i<2*n_aug_+1; ++i)
  {
    weights_(i) = 1.0 / (2.0 * (lambda_+n_aug_));
  }

  // predict state mean
  x_.fill(0.0);
  for(int i=0; i<n_x_; ++i)
  {
    double rowsum = 0;
    for(int j=0; j<2*n_aug_+1; ++j)
    {
      rowsum += weights_(j)*Xsig_pred_(i,j);
    }
    x_(i) = rowsum;
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for(int i=0; i<2*n_aug_+1; ++i)
  {
    VectorXd diffvec      = Xsig_pred_.col(i) - x_;
    RowVectorXd diffvectrans = diffvec.transpose();
    //Matrix<double, Dynamic, 1> diffvectrans = diffvec.transpose();
    P_ = P_ + weights_(i) * diffvec * diffvectrans;
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  /**
   * DONE: update the state by using Kalman Filter equations, reused from EKF
   */
  int n_z = 2;
  VectorXd z(n_z);
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1];
  VectorXd z_pred_ = H_laser_ * x_;
  VectorXd y = z - z_pred_;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_laser_ * PHt + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_laser_) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // Predict Radar Sigma Points
  // Lesson 7, section 27:

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for(int i=0; i<2*n_aug_+1; ++i)  //2n+1 sigma points
  {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double v1  = cos(yaw)*v;
    double v2  = sin(yaw)*v;

    //measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);         // rho
    Zsig(1, i) = atan2(p_y, p_x);                 // phi
    Zsig(2, i) = (p_x*v1 + p_y*v2) / Zsig(0, i);  // rho_dot
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for(int i=0; i< 2*n_aug_ + 1; ++i)
  {
    z_pred += weights_(i)*Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    RowVectorXd z_diff_trans = z_diff.transpose();
    // angle normalization
    //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff_trans;
  }

  // adding of measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<  std_radr_*std_radr_, 0, 0,
  0, std_radphi_*std_radphi_, 0,
  0, 0,std_radrd_*std_radrd_;
  S += R;

  // update radar
  // Lesson 7, section 30

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  // fill with zeros
  Tc.fill(0.0);
  for(int i=0; i < 2 * n_aug_ + 1; i++)
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // normalize angle
    normalizeAngle(x_diff(3));

    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // also normalize angle here
    normalizeAngle(z_diff(1)); // phi

    // accumulate
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  // is product of cross corr matrix Tc times the inverse of the predicted measurement covariance S
  MatrixXd K;
  K = Tc * S.inverse();
  // residual between real measurement z_{k+1} (==z) and mean predicted z_pred
  VectorXd z_diff = x_ - z_pred;
  // normalize phi angle of z_diff
  normalizeAngle(z_diff(1));
  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
}


void UKF::normalizeAngle(double& f_angle)
{
  while(f_angle >  M_PI) f_angle -= 2.0*M_PI;
  while(f_angle < -M_PI) f_angle += 2.0*M_PI;
}
