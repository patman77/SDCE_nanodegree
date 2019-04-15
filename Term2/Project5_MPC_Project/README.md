# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

# Description of the model

State:
The used state is (x, y, psi, v), i.e. contains the vehicle position (x, y), the vehicle orientation psi, measured against a straight line to the front, the so called cross track error cte, giving the perpendicual distance from the vehicle to the reference line, and the orientation error epsi.

Actuators:
We simplify such that there is no shifting of gears. Therefore, the remaining actuators are steering and acceleration, (delta, a). The braking is modelled as negative acceleration, so no additional actuator is necessary.

Update equations:
The equations for the model are:
```
// Recall the equations for the model:
// x_[t] = x[t-1] + v[t-1] * cos(psi[t-1]) * dt
// y_[t] = y[t-1] + v[t-1] * sin(psi[t-1]) * dt
// psi_[t] = psi[t-1] + v[t-1] / Lf * delta[t-1] * dt
// v_[t] = v[t-1] + a[t-1] * dt
```

Essentially, the new position at time step t is given from the old position at time step t-1 plus the previous velocity times the discrete elapsed time dt per frame. For this, the increment for component x and y receives its contribution from the projection to each axis (cosine for x and sine for y).
Orientation psi is additively increased by a multiplicative factor of the steering angle, delta.
Lf measures the distance between the center of mass of the vehicle and it's front axle. The larger the vehicle, the slower the turn rate.
At higher speeds you turn quicker than at lower speeds. This is why v is the included in the update.

Speed v is increased by acceleration a times dt, if a is 0, speed remains constant.
a can take values between and including -1 and 1.

# Description of Timestep Length and Elapsed Duration (N & dt)

The timestep length N and elapsed duration dt were experimentally tried out to be 10 and 0.1, respectively.

The total prediction time (let's call it tpt) into the future is the product of both, so N*dt = 10 * 0.1 sec = 1 sec.

If N is too high the solver would run longer as matrix size depends on N. Being too slow is not good in a real time system. Also, tpt would be higher by tendency, and there is no benefit of looking too far into the future. Realistically, the reference line is not globally known (unless some kind of a high accuracy map plus localization is used), but rather comes from e.g. a lane detection. When going into the curve, the detection is naturally limited because sensors such as cameras usually "look" with straight "detection lines" into the world, and after some point in prediction time, there is no valid information any more.
On the other hand, if N is too low, there is not enough information, and the vehicle is unlikely to go back to reference line.
dt was to be chosen such that tpt is around 1 sec, which turned out to be a good value for prediction. As the number of variables is 6*N+2(N-1), N=10 already leads to 78 variables. This leads to dt=0.1 so that tpt becomes 1 sec.

# Description of Polynomial Fitting and MPC Preprocessing

A polynomial of degree 3 is fitted to the waypoints, see line 71 in main.cpp:

```
 auto coeffs = polyfit(ptsx_transform, ptsy_transform, 3); // fit to polynomial of degree 3
 ```

MPC procedure follows without any preprocessing, see line 84 in main.cpp:

```
auto vars = mpc.Solve(state, coeffs);
````

# Description of the Model Predictive Control with Latency

After the steering angle and the throttle has been determined by the solver, it is sent back to the simulator and thereby to the actuators steering wheel and gas pedal / break. As actuators have a latency given by their mechanical nature, the effect will be delayed. This is modeled in the entire processing chain by waiting for 100 ms after sending values to the actuators, see line 161 main.cpp:

```
std::this_thread::sleep_for(std::chrono::milliseconds(100));
```

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.

* **Ipopt and CppAD:** Please refer to [this document](https://github.com/udacity/CarND-MPC-Project/blob/master/install_Ipopt_CppAD.md) for installation instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Build with Docker-Compose
The docker-compose can run the project into a container
and exposes the port required by the simulator to run.

1. Clone this repo.
2. Build image: `docker-compose build`
3. Run Container: `docker-compose up`
4. On code changes repeat steps 2 and 3.

## Tips

1. The MPC is recommended to be tested on examples to see if implementation behaves as desired. One possible example
is the vehicle offset of a straight line (reference). If the MPC implementation is correct, it tracks the reference line after some timesteps(not too many).
2. The `lake_track_waypoints.csv` file has waypoints of the lake track. This could fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.)
4.  Tips for setting up your environment are available [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)
5. **VM Latency:** Some students have reported differences in behavior using VM's ostensibly a result of latency.  Please let us know if issues arise as a result of a VM environment.

## Editor Settings

We have kept editor configuration files out of this repo to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/b1ff3be0-c904-438e-aad3-2b5379f0e0c3/concepts/1a2255a0-e23c-44cf-8d41-39b8a3c8264a)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. We omitted IDE profiles to ensure
students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. Most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio and develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
