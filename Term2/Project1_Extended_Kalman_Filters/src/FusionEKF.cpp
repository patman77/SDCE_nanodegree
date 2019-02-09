#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0     ,
              0     , 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0     , 0   ,
              0   , 0.0009, 0   ,
              0   , 0     , 0.09;

  // state covariance matrix P from lesson 5, chapter 14
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;
  /**
   * DONE: Finish initializing the FusionEKF.
   */
  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;
    // the initial transition matrix F_ from lesson 5, chapter 14
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ = MatrixXd::Identity(4,4);
//  ekf_.F_ << 1, 0, 1, 0,
//             0, 1, 0, 1,
//             0, 0, 1, 0,
//             0, 0, 0, 1;
  /**
   * DONE: Set the process and measurement noises
   */
   noise_ax = noise_ay = 9.0;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates 
      //         and initialize state.
        float rho     = measurement_pack.raw_measurements_[0];
        float phi     = measurement_pack.raw_measurements_[1];
        float rho_dot = measurement_pack.raw_measurements_[2];
        ekf_.x_ << rho * cos(phi),
                   rho * sin(phi),
                   rho_dot * sin(phi),
                   rho_dot * cos(phi);
                   //0.0,
                   //0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // DONE: Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0.0,
                 0.0;

    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * DONE: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * DONE: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  // 1. Modify the F matrix so that the time is integrated
  ekf_.F_ << 1, 0, dt,  0,
             0, 1,  0, dt,
             0, 0,  1,  0,
             0, 0,  0,  1;
  // 2. Set the process covariance matrix Q
  float dt2 = dt*dt;
  float dt3 = dt*dt2;
  float dt4 = dt2*dt2;
  ekf_.Q_ << dt4/4.0 * noise_ax,              0.0,  dt3/2.0 * noise_ax,                0.0,
                            0.0, dt4/4.0*noise_ay,                 0.0, dt3/2.0 * noise_ay,
             dt3/2.0 * noise_ax,              0.0,      dt2 * noise_ax,                0.0,
                              0, dt3/2.0*noise_ay,                   0,     dt2 * noise_ay;
  // 3. Call the Kalman Filter predict() function
  ekf_.Predict();

  /**
   * Update
   */

  /**
   * DONE:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // DONE: Radar updates
    ekf_.R_ = R_radar_;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_); // use extended for radar

  } else {
    // DONE: Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_); // use standard for lidar
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
