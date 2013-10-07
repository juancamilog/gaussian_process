#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <cmath>
#include "optimization.h"
#include<limits>
#include<execinfo.h>

#define PI 3.14159265358979

using namespace Eigen;

typedef double kernel_func(VectorXd&,VectorXd&, alglib::real_1d_array &);
typedef void gradient_func(const alglib::real_1d_array &, double &, alglib::real_1d_array &, void *);

class kernel_object{
    public:
        kernel_object();
        kernel_object(std::function<kernel_func> &k,
                std::function<gradient_func> &g,
                int n_params);

        std::function<kernel_func> function;
        std::function<gradient_func> gradient;

        /* for dlib */
        alglib::real_1d_array parameters;
        double observation_noise;
        double search_step_size;
        int iters;
        double best_log_l;
        alglib::real_1d_array best_parameters;
        double best_observation_noise;

        alglib::minlbfgsstate state;
};

class gaussian_process{
    public:
        gaussian_process();
        gaussian_process(MatrixXd &Xin, MatrixXd &Yin);
        void init(const alglib::real_1d_array &x, double observation_noise);
        VectorXd kernel_vector(VectorXd &x);
        MatrixXd kernel_matrix(MatrixXd &X);
        VectorXd compute_marginal_covariance(VectorXd &x);
        double log_marginal_likelihood();
        void prediction(VectorXd &x, VectorXd &mean, double &variance);
        void optimize_kernel_parameters(double step_size=1e-5, double stopping_criterion=1e-7);

        /* square exponential (RBF) kernel */
        void set_SE_kernel();

        /*
        void set_matern_kernel();
        void set_gamma_exp_kernel();
        void set_RQ_kernel();
        void set_pp_kernel();
        void set_dot_product_kernel();*/

        kernel_object kernel;

    private:
        MatrixXd K;
        VectorXd KY;
        MatrixXd Kinv;
        LLT<MatrixXd,Lower> llt_of_K;
        double log_det_K;
        double normalization_const;
        MatrixXd X;
        VectorXd Y;
        MatrixXd KinvY;
};
