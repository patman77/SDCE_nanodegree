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
  num_particles = 100;  // DONE: Set the number of particles
  // From lesson 14, chapter 5:
  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std[0]);
  // Create normal distributions for y and theta
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  weights.reserve(num_particles);    // faster than push_back
  weights.resize(num_particles);
  weights.assign(num_particles, 1.0);

  particles.reserve(num_particles);  // faster than push_back
  particles.resize(num_particles);

  for(int i=0; i<num_particles; ++i)
  {
    // Sample from these normal distributions like this:
    // sample_x = dist_x(gen);
    // where "gen" is the random engine initialized earlier.
    particles[i].id     = i;
    particles[i].x      = dist_x(gen);
    particles[i].y      = dist_y(gen);
    particles[i].theta  = dist_theta(gen);
    particles[i].weight = 1.0;
  }
  is_initialized = true;
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

  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for(int i=0; i<num_particles; ++i)
  {
    // calculate prediction
    if(fabs(yaw_rate) < 0.0001)
    {
      particles[i].x += cos(particles[i].theta) * velocity * delta_t;
      particles[i].y += sin(particles[i].theta) * velocity * delta_t;
    }
    else
    {
      double delta_theta = yaw_rate * delta_t;
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + delta_theta) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + delta_theta));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add random noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].y += dist_theta(gen);
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
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs& observation = observations[i];

    double dist_to_predict = -1;
    int predict_id = -1;
    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs predict = predicted[j];

      double distance = dist(observation.x, observation.y, predict.x, predict.y);
      if (dist_to_predict == -1 || distance < dist_to_predict) {
        dist_to_predict = distance;
        predict_id = predict.id;
      }
    }
    observation.id = predict_id;
  }    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                      const vector<LandmarkObs> &observations,
                                      const Map &map_landmarks) {
  /**
   * DONE: Update the weights of each particle using a mult-variate Gaussian
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
  // The Multivariate-Gaussian is evaluated at the point of the transformed measurement's position.
  std::cout<<"entering updateWeights, sensor_range="<<sensor_range<<", #obs="<<observations.size()
  <<", #map landmarks="<<map_landmarks.landmark_list.size()<<std::endl;
  for(int i=0; i<particles.size(); ++i)
  {
    std::cout<<"particle id="<<particles[i].id<<", particle x="<<particles[i].x<<", particle y="<<particles[i].y<<std::endl;
    //vector<LandmarkObs> transformed_observations;
    Particle particle = particles[i];
    double new_weight = 1.0;
    for(int j=0; j<observations.size(); ++j)
    {
      double x_map, y_map;
      transform2d(particles[i].x, particles[i].y,
                  observations[j].x, observations[j].y, particles[i].theta,
                  x_map, y_map);
      LandmarkObs lmo;
      lmo.x = x_map;
      lmo.y = y_map;
      lmo.id = observations[j].id;
      Particle map_coordinates;
      map_coordinates.x = lmo.x;
      map_coordinates.y = lmo.y;

      Map::single_landmark_s closest_landmark = findClosestLandmark(map_coordinates, map_landmarks);
      double prob = multivariate_gaussian_2d(map_coordinates.x, map_coordinates.y,
                                             closest_landmark.x_f, closest_landmark.y_f,
                                             std_landmark[0], std_landmark[1]);
      new_weight *= prob;
    }
    particle.weight = new_weight;
    weights[i]      = new_weight;
  }
}

void ParticleFilter::updateWeightsOld(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * DONE: Update the weights of each particle using a mult-variate Gaussian
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
  // The Multivariate-Gaussian is evaluated at the point of the transformed measurement's position.
  std::cout<<"entering updateWeights, sensor_range="<<sensor_range<<", #obs="<<observations.size()
  <<", #map landmarks="<<map_landmarks.landmark_list.size()<<std::endl;
  for(int i=0; i<particles.size(); ++i)
  {
    std::cout<<"particle id="<<particles[i].id<<", particle x="<<particles[i].x<<", particle y="<<particles[i].y<<std::endl;
    vector<LandmarkObs> transformed_observations;

    for(int j=0; j<observations.size(); ++j)
    {
      double x_map, y_map;
      transform2d(particles[i].x, particles[i].y,
                  observations[j].x, observations[j].y, particles[i].theta,
                  x_map, y_map);
      LandmarkObs lmo;
      lmo.x = x_map;
      lmo.y = y_map;
      transformed_observations.push_back(lmo);
    }
    std::cout<<"transformed observations from vehicle to map coord system"<<std::endl;

    // collect landmarks in sensor range
    vector<LandmarkObs> potential_landmarkobs;
    for(int lm = 0; lm < map_landmarks.landmark_list.size(); lm++)
    {
      if(dist(particles[i].x, particles[i].y,
                                        map_landmarks.landmark_list[lm].x_f,
                                        map_landmarks.landmark_list[lm].y_f) <= sensor_range)
      {
        Map::single_landmark_s landmark = map_landmarks.landmark_list[lm];
        LandmarkObs landmarkobs;
        landmarkobs.id = landmark.id_i;
        landmarkobs.x  = landmark.x_f;
        landmarkobs.y  = landmark.y_f;
        potential_landmarkobs.push_back(landmarkobs);
      }
    }
    std::cout<<"collected landmarks in sensor range"<<std::endl;

    //dataAssociation(transformed_observations, potential_landmarkobs);
    if (potential_landmarkobs.size() > 0)
    {
      vector<int> associations;
      vector<double> obs_x;
      vector<double> obs_y;
      for (int k = 0; k < transformed_observations.size(); k++) {
        double minDist = std::numeric_limits<double>::max();
        int bestIndex = 0;
        for (int p = 0; p < potential_landmarkobs.size(); p++) {
          double d = dist(potential_landmarkobs[p].x, potential_landmarkobs[p].y,
                         transformed_observations[k].x, transformed_observations[k].y);
          if (d < minDist) {
            minDist = d;
            bestIndex = p;
          }
        }
        associations.push_back(potential_landmarkobs[bestIndex].id);
        obs_x.push_back(transformed_observations[k].x);
        obs_y.push_back(transformed_observations[k].y);
      }

      particles[i].associations = associations;
      particles[i].sense_x = obs_x;
      particles[i].sense_y = obs_y;
      std::cout<<"finished associations"<<std::endl;

      // Calc particle's new weight
      double particle_weight = 1.0;
      for (int j = 0; j < particles[i].associations.size(); j++) {
        int landmark_id = particles[i].associations[j];
        // NOTE: obeye zero-based vs 1-based index
        Map::single_landmark_s landmark = map_landmarks.landmark_list[landmark_id-1];
        double x_lm = landmark.x_f;
        double y_lm = landmark.y_f;
        double x_obs = particles[i].sense_x[j];
        double y_obs = particles[i].sense_y[j];

        double prob = multivariate_gaussian_2d(x_obs, y_obs, x_lm, y_lm, std_landmark[0], std_landmark[1]);

        particle_weight *= prob;
      } // traverse particles[j] associations

      particles[i].weight = particle_weight;
      weights[i]          = particle_weight;
    } // any potential landmark observations?

  } // traverse all particles
}

void ParticleFilter::resample() {
  /**
   * DONE: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // inspired by Lesson 13, Particle Filters, lesson 20> Resampling Wheel
  //std::random_device rd;
  //std::mt19937 gen(rd());
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;

  for(int i=0; i<num_particles; ++i)
  {
    resampled_particles.push_back(particles[distribution(gen)]);
  }
  particles = resampled_particles;
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

Map::single_landmark_s ParticleFilter::findClosestLandmark(Particle map_coordinates, Map map_landmarks)
{
  Map::single_landmark_s closest_landmark = map_landmarks.landmark_list.at(0);
  double distance = dist(closest_landmark.x_f, closest_landmark.y_f, map_coordinates.x, map_coordinates.y);

  for(int i=1; i<map_landmarks.landmark_list.size(); ++i) {
    Map::single_landmark_s current_landmark = map_landmarks.landmark_list.at(i);
    double current_distance = dist(current_landmark.x_f, current_landmark.y_f, map_coordinates.x, map_coordinates.y);

    if(current_distance < distance) {
      distance = current_distance;
      closest_landmark = current_landmark;
    }
  }
  return closest_landmark;
}
