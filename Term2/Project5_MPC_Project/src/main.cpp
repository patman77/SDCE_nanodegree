#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "MPC.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    std::cout << sdata << std::endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];

          /**
           * DONE: Calculate steering angle and throttle using MPC.
           * Both are in between [-1, 1].
           */
          for(int i=0; i<ptsx.size(); ++i)
          {
            // shift car reference angle to 90 degrees
            double shift_x = ptsx[i]-px;
            double shift_y = ptsy[i]-py;

            ptsx[i] = (shift_x * cos(0-psi)-shift_y*sin(0-psi));
            ptsy[i] = (shift_x * sin(0-psi)+shift_y*cos(0-psi));
          }

          double* ptrx = &ptsx[0];
          Eigen::Map<Eigen::VectorXd> ptsx_transform(ptrx, 6);

          double* ptry = &ptsy[0];
          Eigen::Map<Eigen::VectorXd> ptsy_transform(ptry, 6);

          auto coeffs = polyfit(ptsx_transform, ptsy_transform, 3); // fit to polynomial of degree 3

          // calculate cte and epsi
          double cte = polyeval(coeffs, 0);
          double epsi = psi - atan(coeffs[1] + 2*px*coeffs[2] + 3*coeffs[3]*pow(px,2)); // derivative of 3rd order polynomial
          // double epsi = -atan(coeffs[1]); // derivate of 1st order polynomial (== affine function)

          double steer_value = j[1]["steering_angle"]; // grab from json, inspired by video walkthrough
          double throttle_value = j[1]["throttle"];    // grab from json, inspired by video walkthrough

          Eigen::VectorXd state(6);
          state << 0, 0, 0, v, cte, epsi; // fill state vector

          auto vars = mpc.Solve(state, coeffs);


          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the 
          //   steering value back. Otherwise the values will be in between 
          //   [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          // Display the MPC predicted trajectory 
          const double poly_inc = 2.5; // 2.5 distance between points predicted ahead
          const int num_points  = 25; // 25 points into the potential future

          vector<double> mpc_x_vals(num_points-2);
          vector<double> mpc_y_vals(num_points-2);
          /**
           * DONE: add (x,y) points to list here, points are in reference to
           *   the vehicle's coordinate system the points in the simulator are 
           *   connected by a Green line
           */

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          for(int i=2; i<vars.size(); ++i)
          {
            if(i%2 == 0)
            {
              //mpc_x_vals.push_back(vars[i]); // push_back can slow down
              mpc_x_vals[i] = vars[i];
            }
            else
            {
              //mpc_y_vals.push_back(vars[i]); // push_back can slow down
              mpc_y_vals[i] = vars[i];
            }
          }

          // Display the waypoints/reference line
          vector<double> next_x_vals(num_points-1);
          vector<double> next_y_vals(num_points-1);

          for(int i=1; i<num_points; ++i)
          {
            //next_x_vals.push_back(poly_inc*i);                   // push_back can get really slow
            //next_y_vals.push_back(polyeval(coeffs, poly_inc*i)); // push_back can get really slow
            next_x_vals[i-1] = poly_inc*i;
            next_y_vals[i-1] = polyeval(coeffs, poly_inc*i);
          }

          /**
           * DONE: add (x,y) points to list here, points are in reference to
           *   the vehicle's coordinate system the points in the simulator are 
           *   connected by a Yellow line
           */
          double Lf = 2.67;

          msgJson["steering_angle"] = vars[0]/(deg2rad(25)*Lf);
          msgJson["throttle"]       = vars[1];

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["next_y"] = mpc_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          //   the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          //   around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE SUBMITTING.
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
