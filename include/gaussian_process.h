#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <cmath>
#include "optimization.h"
#include<limits>
//#include<execinfo.h>
#include <random>
#include <chrono>
#include <omp.h>

#define PI 3.14159265358979


using namespace Eigen;

typedef double kernel_func(VectorXd&,VectorXd&, alglib::real_1d_array &);
typedef void gradient_func(const alglib::real_1d_array &, double &, alglib::real_1d_array &, void *);
typedef void kernel_func_alglib(const alglib::real_1d_array &, double &, void *);

enum{
   ALGLIB_SOLVER_LBFGS,
   ALGLIB_SOLVER_NUM_LBFGS,
   ALGLIB_SOLVER_CONSTRAINED_CG,
   ALGLIB_SOLVER_CG
};

enum{
   KERNEL_RBF,
   KERNEL_SQUARED_EXPONENTIAL
};

class kernel_object{
    public:
        kernel_object();
        kernel_object(std::function<kernel_func> &k,
                std::function<gradient_func> &g,
                std::function<kernel_func_alglib> &f);

        void init(std::function<kernel_func> &k,
                std::function<gradient_func> &g,
                std::function<kernel_func_alglib> &f);

        std::function<kernel_func> function;
        std::function<gradient_func> gradient;
        std::function<kernel_func_alglib> function_alglib;

        int id;

        /* for alglib */
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
        gaussian_process();
        gaussian_process(int input_dimensions);
        gaussian_process(MatrixXd &Xin, MatrixXd &Yin);
        gaussian_process(MatrixXd &Xin, VectorXd &Yin);
        void init(const alglib::real_1d_array x, double observation_noise, bool noise_free=false);
        void init(const alglib::real_1d_array x);

        void add_sample(VectorXd &X, double value);
        void remove_sample(int sample_id);

        int dataset_size();
        int input_dimensions();

        VectorXd kernel_vector(VectorXd &x);
        MatrixXd kernel_matrix(MatrixXd &X, bool noise_free=false);
        VectorXd compute_marginal_covariance(VectorXd &x);
        double log_marginal_likelihood();
        double leave_one_out_log_probability();
        void prediction(VectorXd &x, VectorXd &mean, double &variance);
        void predictive_error_and_variance(VectorXd &error, VectorXd &variance, int type = 0);

        void set_opt_starting_point(VectorXd point);
        void set_opt_random_start(double scale=1.0, double offset = 0.0);
        void optimize_parameters(double stopping_criterion=1e-7,int solver = 0);
        void optimize_parameters_random_restarts(double stopping_criterion=1e-7,int solver = 0, int restarts=2,double scale=1.0);


        /* square exponential (RBF) kernel */
        void set_SE_kernel(int input_dimensions);
        void set_RBF_kernel();

        void set_debug_print(bool dbg_prnt);

        /*
        void set_matern_kernel();
        void set_gamma_exp_kernel();
        void set_RQ_kernel();
        void set_pp_kernel();
        void set_dot_product_kernel();*/

        kernel_object* kernel;

    private:
        MatrixXd K;
        VectorXd KinvY;
        MatrixXd Kinv;
        LLT<MatrixXd,Lower> llt_of_K;
        double log_det_K;
        double normalization_const;
        MatrixXd X;
        VectorXd Y;

        bool debug_print;
};
