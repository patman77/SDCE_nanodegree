#include "PID.h"
#include <math.h>
/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_, bool firstcall_) {
  /**
   * DONE: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;

  // init PID errors
  p_error = i_error = d_error = 0.0;

  // init firstcall
  firstcall = firstcall_;
}

void PID::UpdateError(double cte, double dt) {
  /**
   * DONE: Update PID errors based on cte.
   */
  d_error = (cte - p_error)/dt;
  p_error = cte;
  i_error += cte*dt;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return (-Kp*p_error - Kd*d_error - Ki*i_error);  // DONE: Add your total error calc here!
}

void PID::twiddle(double f_tolerance)
{
  
}

double PID::getSteerAngle()
{
  return (-Kp*p_error - Kd*d_error - Ki*i_error); // full PID controller
  //return (-Kp*p_error); // just proportional term
  //return (-Kp*p_error - Kd*d_error); // just proportional term + derivative term
  //return -Ki*i_error; // just integral term
}

bool PID::getFirstCall()
{
  return firstcall;
}

void PID::setFirstCall(bool rhs)
{
  firstcall = rhs;
}
