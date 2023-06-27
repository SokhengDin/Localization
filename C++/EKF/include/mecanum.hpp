#ifndef MECANUM_HPP__
#define MECANUM_HPP__

#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <chrono>


class MECANUM_EKF
{
    private:
        // MECANUM Parameters
        double Lx_;
        double Ly_;
        double r_;

        // Simulation Parameters
        double dt_; // Sampling Time
        double sim_time_; // Simulation Time

        // EKF Pameters
        Eigen::Matrix4d Q_; // Convariance Matrix for State
        Eigen::Matrix2d R_; // Convariance Matrix for Input

        // Input noise

        Eigen::Matrix2d INPUT_NOISE_;
        Eigen::Matrix2d GPS_NOISE_;

    public:
        MECANUM_EKF(
            double Lx, double Ly, double r,
            double dt, double sim_time,
            Eigen::Matrix4d Q, Eigen::Matrix2d R,
            Eigen::Matrix2d INPUT_NOISE, Eigen::Matrix2d GPS_NOISE
        );

        virtual ~MECANUM_EKF();

        void calc_input(double v, double yawrate);
        void observation(Eigen::Matrix4d xTrue, Eigen::Matrix4d xd, Eigen::Matrix2d u);
        void ekf_estimation(Eigen::Matrix4d xTrue, Eigen::Matrix4d PEst, Eigen::Matrix2d z, Eigen::Matrix2d u);
        void plot_covariance_ellipse(Eigen::Matrix4d xEst, Eigen::Matrix2d PEst);

        Eigen::Matrix4d motion_model(Eigen::Matrix4d x, Eigen::Matrix4d u);
        Eigen::Matrix2d observation_model(Eigen::Matrix4d x);
        Eigen::Matrix4d jacob_f(Eigen::Matrix4d x, Eigen::Matrix2d u);
        Eigen::Matrix2d jacob_h();

};

#endif