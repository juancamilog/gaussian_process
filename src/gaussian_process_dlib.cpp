#include "gaussian_process.h"

// empty constructor
gaussian_process::gaussian_process(){
    set_SE_kernel();
}

// constructor with dataset
gaussian_process::gaussian_process(MatrixXd &Xin, MatrixXd &Yin){
    X = Xin;
    Y = Yin;
    set_SE_kernel();
}

// precomputations
void gaussian_process::init(const column_vector &x, double observation_noise){
    kernel.parameters = column_vector(x);
    kernel.observation_noise = observation_noise;

    int n = X.cols();
    K = kernel_matrix(X);
    llt_of_K = LLT<MatrixXd,Lower>(K);
    Kinv = llt_of_K.solve(MatrixXd::Identity(n,n));
    KY = llt_of_K.solve(Y);
    log_det_K = 1;
    for (int i =0; i < n ; i++){
        log_det_K *=  (llt_of_K.matrixL())(i,i)*(llt_of_K.matrixL())(i,i);
    }
    log_det_K = std::log(std::abs(log_det_K));
}

// compute the covariance vector between a sample point x, and the samples in the dataset X
VectorXd gaussian_process::kernel_vector(VectorXd &x){
    int n = X.cols();
    VectorXd kx = VectorXd(n);
    VectorXd x_i;
    for(int i=0; i<n; i++){
        x_i = X.col(i);
        kx[i]=kernel.function(x_i,x,kernel.parameters);
    }
    return kx;
}

// computes the covariance matrix between the samples in the dataset
MatrixXd gaussian_process::kernel_matrix(MatrixXd &X){
    int n = X.cols();
    MatrixXd K = MatrixXd(n,n);
    VectorXd x_i,x_j;

    for (int j=0; j<n ; j++){
        for (int i=j; i<n ; i++){
            x_i = X.col(i);
            x_j = X.col(j);
            K(i,j) = kernel.function(x_i,x_j,kernel.parameters);
            if (i==j){
                K(i,j) += kernel.observation_noise;
            }else{
                K(j,i) = K(i,j);
            }
        }
    }
    return K;
}

// predict the function value for a new sample x
void gaussian_process::prediction(VectorXd &x, VectorXd &mean, double &variance){
    //compute the correlation vector for the input x
    VectorXd kx = kernel_vector(x);
    // compute predictive mean
    mean = kx.transpose()*KY;
    // compute variance estimate
    VectorXd v = llt_of_K.matrixL().solve(kx);
    variance = kernel.function(kx,kx,kernel.parameters) - v.transpose()*v + kernel.observation_noise;
}

// compute the log marginal likelihood of the function values Y, given the inputs X and the kernel parameters
double gaussian_process::log_marginal_likelihood(){
    int n = X.cols();
    return  -0.5*Y.dot(KY) - 0.5*log_det_K - 0.5*n*std::log(2*PI);    
}

// search for the maximum likelihood parameters of the kernel
void gaussian_process::optimize_kernel_parameters(double step_size, double stopping_criterion){

    std::cout<<"++kernel params before optimization++\t"<<dlib::trans(kernel.parameters)<<std::endl;

    kernel.search_step_size = step_size;

    //dlib::find_max(dlib::bfgs_search_strategy(),
    dlib::find_max(dlib::cg_search_strategy(),
                   dlib::objective_delta_stop_strategy(stopping_criterion),
                   //dlib::gradient_norm_stop_strategy(stopping_criterion),
                   kernel.log_likelihood,
                   kernel.gradient,
                   kernel.parameters,10);
    
    /*dlib::find_max_using_approximate_derivatives(dlib::cg_search_strategy(),
                   dlib::objective_delta_stop_strategy(stopping_criterion),
                   //dlib::gradient_norm_stop_strategy(stopping_criterion),
                   kernel.log_likelihood,
                   kernel.parameters,10);*/

    std::cout<<"++kernel params after optimization++\t"<<dlib::trans(kernel.parameters)<<std::endl;
    std::cout<<"++likelihood++\t"<<kernel.log_likelihood(kernel.parameters)<<std::endl;

}

//================================== kernel functions ===================================//
void gaussian_process::set_SE_kernel(){
    
    // parameters correspond to (sigma_f^2, 1/(2*l_1), ... , 1/(2*l_d), sigma_n^2)
    // kernel function for evaluations
    std::function<kernel_func> se_kernel = [this](VectorXd x_i, VectorXd &x_j, const column_vector &parameters){
        //TODO receive distance function as parameter
        int d = x_i.size();
        double dist;
        for (int i=0; i<d; i++){
            dist += (x_i[i]-x_j[i])*(x_i[i]-x_j[i])*parameters(i+1);
        }
        return parameters(0)*parameters(0)*std::exp(-0.5*dist);
    };
    
    // evaluation of log likelihood
    std::function<logl_func> se_log_likelihood = [this](const column_vector &x){
        int d = this->X.rows();
        if(kernel.parameters!=x){
            // update K and Kinv
            this->init(x, kernel.parameters(d+1)*kernel.parameters(d+1));
        }
        return this->log_marginal_likelihood();
    };

    // gradient of log likelihood
    std::function<gradient_func> se_gradient = [this](const column_vector &x){
        int d = this->X.rows();
        int n = this->X.cols();

        if(kernel.parameters!=x){
            // update K and Kinv
            this->init(x, kernel.parameters(d+1)*kernel.parameters(d+1));
        }

        column_vector grad = dlib::zeros_matrix<double>(d+2,1);
        // compute ((K^{-1}*y)*(K^{-1}*y)^T- K^{-1})^{T}
        MatrixXd K_a = this->llt_of_K.solve(Y);
        K_a = ((K_a)*(K_a).transpose() - this->Kinv).transpose();
        //K_a = ((K_a)*(K_a).transpose()).transpose();

        // compute dK/dsigma_f^2
        VectorXd x_i,x_j;

        double sigma_f = x(0);
        double sigma_n = x(d+1);
        double quart_sigma_f_sq = (0.25*x(0)*x(0));

        VectorXd length_scales(d);
        for (int i=0; i<d; i++){
            length_scales[i] = x(i+1);
        }

        double dist, exp_ij;

        // here we are computing 1/2*trace{ ( (K^{-1}*y)*(K^{-1}*y)^T- K^{-1} )*dK/d_param ) }
        //                       = 1/2*trace { K_a^{T}*dK/d_param }
        //                       = 1/2*sum_row { sum_col { { K_a ** dK/d_param } }  (** means element-wise product)
        //                       = sum_row { sum_col { { K_a ** ( 0.5*dK/d_param) } }  (** means element-wise product)
        for (int j=0; j<n ; j++){
            for (int i=0; i<n ; i++){
                x_i = this->X.col(i);
                x_j = this->X.col(j);

                dist = ((x_i-x_j).cwiseProduct(length_scales)).squaredNorm();
                exp_ij = std::exp(-0.5*dist);

                // irst accumulate the gradient for sigma_f^2
                grad(0) += K_a(j,i)*(  sigma_f*exp_ij );

                // the accumulate the gradient for the length scales
                for ( int k=0; k<d; k++){
                    grad(k+1) += K_a(j,i)*( -quart_sigma_f_sq*(x_i[k]-x_j[k])*(x_i[k]-x_j[k])*exp_ij );
                }

                // finally accumulate the gradient for the noise parameter
                grad(d+1) += K_a(j,i)*( (i==j)?sigma_n:0 );
            }
        }
        //dlib::matrix<double> g = this->kernel.search_step_size*dlib::normalize(grad);
        dlib::matrix<double> g = this->kernel.search_step_size*grad;

        return g;
    };

    kernel.function = se_kernel;
    kernel.log_likelihood = se_log_likelihood;
    kernel.gradient = se_gradient;

    int d = X.rows();
    // number of parameters is d+2
    column_vector parameters = dlib::ones_matrix<double>(d+2,1);
    double observation_noise = parameters(d+1)*parameters(d+1);
    init(parameters,observation_noise);
}

//==================================== kernel object =====================================//

kernel_object::kernel_object( ){

}

kernel_object::kernel_object(std::function<kernel_func> &k,
                std::function<gradient_func> &g,
                int n_params){
    function = k;
    gradient = g;
}
