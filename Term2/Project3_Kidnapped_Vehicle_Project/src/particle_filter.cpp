/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * DONE: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * DONE: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // DONE: Set the number of particles
  // From lesson 14, chapter 5:
  std::default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std[0]);
  // Create normal distributions for y and theta
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  weights.reserve(num_particles);
  weights.resize(num_particles);
  weights.assign(num_particles, 1.0);

  particles.reserve(num_particles);
  particles.resize(num_particles);


  for(int i=0; i<num_particles; ++i)
  {
    double sample_x, sample_y, sample_theta;
    // Sample from these normal distributions like this:
    // sample_x = dist_x(gen);
    // where "gen" is the random engine initialized earlier.
    sample_x            = dist_x(gen);
    sample_y            = dist_y(gen);
    sample_theta        = dist_theta(gen);
    particles[i].x      = sample_x;
    particles[i].y      = sample_y;
    particles[i].theta  = sample_theta;
    particles[i].weight = 1.0;
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * DONE: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // Lesson 14, chapter 8,9
  std::default_random_engine gen;

  for(int i=0; i<num_particles; ++i)
  {
    double sample_x, sample_y, sample_theta;

    std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

    sample_x     = dist_x(gen);
    sample_y     = dist_y(gen);
    sample_theta = dist_theta(gen);

    // calculate prediction
    double xpred = velocity/yaw_rate * (sin(sample_theta + yaw_rate*delta_t) - sin(sample_theta));
    double ypred = velocity/yaw_rate * (cos(sample_theta - cos(sample_theta + yaw_rate*delta_t)));
    double thetapred = sample_theta + yaw_rate * delta_t;

    sample_x += xpred;
    sample_y += ypred;
    sample_theta += thetapred;

    particles[i].x     = sample_x;
    particles[i].y     = sample_y;
    particles[i].theta = sample_theta;
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * DONE: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  // Iterate through each transformed observation to associate to a landmark
  double product = 1.0;
  unsigned int num_obs = observations.size();
  unsigned int num_landmarks = predicted.size();
  for (int i = 0; i < num_obs; ++i) {
    int closest_landmark = 0;
    int closest_mapId = -1;
    int min_dist = 999999;
    int curr_dist;
    // Iterate through all landmarks to check which is closest
    for (int j = 0; j < num_landmarks; ++j) {
      // Calculate Euclidean distance
      curr_dist = sqrt(pow(predicted[i].x - observations[j].x, 2)
                       + pow(predicted[i].y - observations[j].y, 2));
      // Compare to min_dist and update if closest
      if (curr_dist < min_dist) {
        min_dist = curr_dist;
        closest_landmark = j;
        closest_mapId = predicted[closest_landmark].id;
      }
    }
    // update observation identifier:
    observations[i].id = closest_mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

// --------------------------------------------------------------------------------------------
// private methods
// --------------------------------------------------------------------------------------------
void ParticleFilter::transform2d(double x_part, double y_part, double x_obs, double y_obs, double theta,
                 double& x_map, double& y_map)
{
  // transform to map x coordinate
  x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
  // transform to map y coordinate
  y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
}
