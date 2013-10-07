#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <cmath>
#include "dlib/optimization.h"
#include<limits>
#include<execinfo.h>

#define PI 3.14159265358979

using namespace Eigen;

typedef dlib::matrix<double,0,1> column_vector;
typedef double kernel_func(VectorXd&,VectorXd&, const column_vector&);
typedef double logl_func(const column_vector&);
typedef column_vector gradient_func(const column_vector&);

class kernel_object{
    public:
        kernel_object();
        kernel_object(std::function<kernel_func> &k,
                std::function<gradient_func> &g,
                int n_params);

        std::function<kernel_func> function;
        std::function<logl_func> log_likelihood;
        std::function<gradient_func> gradient;

        /* for dlib */
        column_vector parameters;
        double observation_noise;

        double best_log_likelihood;
        column_vector best_parameters;

        double search_step_size;

};

class gaussian_process{
    public:
        gaussian_process();
        gaussian_process(MatrixXd &Xin, MatrixXd &Yin);
        void init(const column_vector &x, double observation_noise);
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
        MatrixXd X;
        VectorXd Y;
        MatrixXd KinvY;
};
