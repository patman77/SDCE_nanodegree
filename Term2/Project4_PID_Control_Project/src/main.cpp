#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <string>
#include "json.hpp"
#include "PID.h"

// for convenience
using nlohmann::json;
using std::string;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != string::npos) {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main() {
  uWS::Hub h;

  PID pid;
  double curr_time = 0.0; // current timestamp
  double prev_time = 0.0; // timestamp of previous frame
  double t = 0.0;         // total time
  /**
   * DONE: Initialize the pid variable.
   */
  //pid.Init(0.2, 0.004, 3.0, true); // parameters from the course
  pid.Init(0.35, 0.01, 0.004, true);
  //pid.Init(0.0, 0.0, 0.0, true); // cross check: this should drive straight

  h.onMessage([&pid, &curr_time, &prev_time, &t](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(string(data).substr(0, length));

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<string>());
          double speed = std::stod(j[1]["speed"].get<string>());
          double angle = std::stod(j[1]["steering_angle"].get<string>());
          double steer_value;
          /**
           * TODO: Calculate steering value here, remember the steering value is
           *   [-1, 1].
           * NOTE: Feel free to play around with the throttle and speed.
           *   Maybe use another PID controller to control the speed!
           */
          curr_time = clock();
          double dt = (curr_time - prev_time) / CLOCKS_PER_SEC;
          std::cout<<"JSON DATA: cte="<<cte<<", speed="<<speed<<", angle="<<angle<<std::endl;
          if(pid.getFirstCall())
          {
            //std::cout<<"firstcall"<<std::endl;
            pid.SetPError(cte); // for the first frame, there is no previous value, so use the very first cte
            pid.setFirstCall(false);
          }
          else
          {
            //std::cout<<"___________CALL INBETWEEN__________"<<std::endl;
          }
          pid.UpdateError(cte, dt);
          steer_value = pid.getSteerAngle();
          // correct steering angle to interval [-1,1]
          if(steer_value > 1.0)
          {
            steer_value = 1.0;
          }
          else if(steer_value < -1.0)
          {
            steer_value = -1.0;
          }
          
          double throttle = 0.8; // inspired by discussions on study-hall
#define SIMPLE_THROTTLE_LOGIC
#ifdef SIMPLE_THROTTLE_LOGIC
          throttle = 1.0 - 0.5 * fabs(steer_value);
#else
          if(fabs(cte)>0.5)
          {
            throttle = 0.8;
          }
          // following inspired by discussions on study-hall.udacity.com
          if(fabs(pid.GetPError()-cte) > 0.1 && fabs(pid.GetPError()<= 0.2))
          {
            throttle = 0.0;
          }
          else if(fabs(pid.GetPError()-cte) > 0.2 && speed > 30.0)
          {
            throttle = -0.2; // break
          }
#endif

          // DEBUG
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value 
                    << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          //msgJson["throttle"] = 0.3; // original line
          msgJson["throttle"] = throttle;
          //msgJson["throttle"] = (1.0 - std::abs(steer_value)) * 0.5 + 0.2; // inspired by discussion on study-hall.udacity.com
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          t += dt;
          std::cout<<"t = "<<t<<std::endl;
          prev_time = curr_time;
        }  // end "telemetry" if
      } else {
        // Manual driving
        string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket message if
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
