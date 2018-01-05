/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    num_particles = 100;

    std::random_device rd{};
    default_random_engine gen(rd());
    std::normal_distribution<double> x0(x,std_x);
    std::normal_distribution<double> y0(y,std_y);
    std::normal_distribution<double> theta0(theta,std_theta);

    for(int i=0;i<num_particles;i++){
        Particle p;

        p.id = i;
        p.x = x0(gen);
        p.y = y0(gen);
        p.theta = theta0(gen);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    std::random_device rd{};
    std::default_random_engine gen(rd());
    std::normal_distribution<double> x_noise(0,std_x);
    std::normal_distribution<double> y_noise(0,std_y);
    std::normal_distribution<double> yaw_noise(0,std_theta);

    for(int i=0;i<num_particles;i++){
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;
        if (abs(yaw_rate) > 1e-9){
            x += velocity/yaw_rate * ( sin(theta + yaw_rate*delta_t)- sin(theta) );
            y += velocity/yaw_rate * ( cos(theta) - cos(theta+yaw_rate*delta_t) );
            theta += yaw_rate*delta_t;
        }
        else{
            x += velocity*cos(theta)*delta_t;
            y += velocity*sin(theta)*delta_t;
        }

        particles[i].x = x + x_noise(gen);
        particles[i].y = y + y_noise(gen);
        particles[i].theta = theta + yaw_noise(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for(int i=0;i<observations.size();i++){

        double o_x = observations[i].x;
        double o_y = observations[i].y;

        double d_min = 1e9;
        int id_min = -1;

        for(int j=0;j<predicted.size();j++){

            double p_x = predicted[j].x;
            double p_y = predicted[j].y;
            int p_id = predicted[j].id;

            double d_j = dist(p_x,p_y,o_x,o_y);

            if(d_j <= d_min){

                d_min = d_j;
                id_min = p_id;
            }
        }

        observations[i].id = id_min;
//        cout<<observations[i].id<<endl;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // find partiles within the sensor range for each landmark

    for(int i=0;i<num_particles;i++){
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        std::vector<LandmarkObs> p_inRange;
        for(int j=0;j<map_landmarks.landmark_list.size();j++){

//            double dist_P2L = 0.0;
            double l_x = map_landmarks.landmark_list[j].x_f;
            double l_y = map_landmarks.landmark_list[j].y_f;

            double dist_P2L = dist(p_x,p_y,l_x,l_y);
            if (dist_P2L<=sensor_range){
                LandmarkObs obj_k;
                obj_k.x = l_x;
                obj_k.y = l_y;
                obj_k.id = map_landmarks.landmark_list[j].id_i;
                p_inRange.push_back(obj_k);
//                cout<<"p_inRange id: " <<obj_k.id<<endl;
            }
        }

        // convert the coordinates of each observations to the map coordinate system
        std::vector<LandmarkObs> Map_obs;
        for(int j=0;j<observations.size();j++){

            double o_x = observations[j].x;
            double o_y = observations[j].y;
            double map_x = p_x+cos(p_theta)*o_x-sin(p_theta)*o_y;
            double map_y = p_y+sin(p_theta)*o_x+cos(p_theta)*o_y;

            LandmarkObs obj_k;
            obj_k.x = map_x;
            obj_k.y = map_y;
            obj_k.id = observations[j].id;
            Map_obs.push_back(obj_k);
        }
        // find the nearest landmark for each observation, and assign the observation to its nearest landmark
        dataAssociation(p_inRange,Map_obs);

        // update weights
        double weight = 1.0;
        double std_x = std_landmark[0];
        double std_y = std_landmark[1];

        for(int j=0;j<p_inRange.size();j++){
            // search all landmarks in the sensor range for this particle
            double mu_x = p_inRange[j].x;
            double mu_y = p_inRange[j].y;
            int mu_id = p_inRange[j].id;
//            cout<<"mux muy myid "<<mu_x<<" "<<mu_y<<" "<<mu_id<<endl;
            for(int k=0;k<Map_obs.size();k++){
                // calculate probabilities using observations nearest to this
                int o_id = Map_obs[k].id;
//                cout<<"oid "<<o_id<<endl;
                if(o_id == mu_id){
                    double o_x = Map_obs[k].x;
                    double o_y = Map_obs[k].y;
                    weight *= (1/(2*M_PI*std_x*std_y))*
                              exp( -(o_x-mu_x)*(o_x-mu_x)/(2*std_x*std_x)  - (o_y-mu_y)*(o_y-mu_y)/(2*std_y*std_y) );
//                    cout<<"i j k"<<i<<" "<<" "<<j<<" "<<k<<" "<<weight<<" "<<o_id<<endl;
                }
            }
        }

        // update the weight for this particle
        particles[i].weight = weight;
        weights[i] = weight;
    }

}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> new_particles;
//    cout<<"line 215"<<endl;
    std::random_device rd{};
    default_random_engine gen(rd());

    for ( int i = 0; i < num_particles; ++i ) {
//        cout<<"line 218"<<endl;
        discrete_distribution<int> index( weights.begin(), weights.end() );
//        cout<<"line 220"<<endl;
        new_particles.push_back(particles[index( gen )]);
//        cout<<"line 222"<<endl;
//        cout<<"new p "<<new_particles[i].id<<endl;
    }
    // Replace old particles with the resampled particles
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
