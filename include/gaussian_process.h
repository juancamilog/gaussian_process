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
typedef void kernel_func_alglib(const alglib::real_1d_array &, double &, void *);

class kernel_object{
    public:
        kernel_object();
        kernel_object(std::function<kernel_func> &k,
                std::function<gradient_func> &g,
                int n_params);

        std::function<kernel_func> function;
        std::function<gradient_func> gradient;
        std::function<kernel_func_alglib> function_alglib;

        /* for dlib */
        alglib::real_1d_array parameters;
        double observation_noise;
        int iters;
        double best_log_l;
        alglib::real_1d_array best_parameters;
        double best_observation_noise;

        alglib::minlbfgsstate bfgsstate;
        alglib::mincgstate cgstate;
        alglib::minbleicstate bleicstate;
};

class gaussian_process{
    public:
        gaussian_process(int input_dimensions);
        gaussian_process(MatrixXd &Xin, MatrixXd &Yin);
        gaussian_process(MatrixXd &Xin, VectorXd &Yin);
        void init(const alglib::real_1d_array &x, double observation_noise, bool noise_free=false);

        void add_sample(VectorXd &X, double value);
        double get_maximum_variance();
        double compute_maximum_variance();
        int dataset_size();
        int input_dimensions();

        VectorXd kernel_vector(VectorXd &x);
        MatrixXd kernel_matrix(MatrixXd &X, bool noise_free=false);
        VectorXd compute_marginal_covariance(VectorXd &x);
        double log_marginal_likelihood();
        void prediction(VectorXd &x, VectorXd &mean, double &variance);

        void set_opt_starting_point(VectorXd point);
        void optimize_parameters(double stopping_criterion=1e-7,int solver = 0);

        /* square exponential (RBF) kernel */
        void set_SE_kernel(int input_dimensions);

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
        double maximum_variance;
        MatrixXd X;
        VectorXd Y;
        MatrixXd KinvY;
};
