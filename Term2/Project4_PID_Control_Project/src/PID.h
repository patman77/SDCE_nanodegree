#ifndef PID_H
#define PID_H

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_, bool firstcall);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   * @param dt delta t, time between two successive frames
   */
  void UpdateError(double cte, double dt);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  /**
    * twiddle optimizes the parameters Kp, Ki and Kd with a simple gradient descent
    *
   */
  void twiddle(double f_tolerance);

 /**
   * returns steering angle
   */
  double getSteerAngle();

  double GetKp() { return Kp; }
  void SetKp(double newkp) { Kp = newkp; }
  double GetKi() { return Ki; }
  void SetKi(double newki) { Ki = newki; }
  double GetKd() { return Kd; }
  void SetKd(double newkd) { Kd = newkd; }
  void SetPError(double newperror) { p_error = newperror; }
  bool getFirstCall();
  void setFirstCall(bool rhs);
  double GetPError() { return p_error; }

private:
  /**
   * PID Errors
   */
  double p_error;
  double i_error;
  double d_error;

  /**
   * PID Coefficients
   */ 
  double Kp;
  double Ki;
  double Kd;

  /**
   * firstcall
   */
  bool firstcall;
};

#endif  // PID_H
