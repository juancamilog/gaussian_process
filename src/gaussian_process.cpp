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
void gaussian_process::init(const alglib::real_1d_array &x, double observation_noise){
    kernel.parameters = alglib::real_1d_array(x);
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
    normalization_const = 0.5*n*std::log(2*PI);
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
                K(i,j) += kernel.observation_noise*kernel.observation_noise;
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
    double model_fit_error = 0.5*Y.dot(KY);
    double complexity_penalty = 0.5*log_det_K;

     // function trimming to avoid singularities
    if (std::isnan(model_fit_error)){
        model_fit_error = 1e300;
    }
    if (std::isnan(complexity_penalty)){
        complexity_penalty = 1e300;
    }
    if(std::isinf(model_fit_error)){
        model_fit_error = (model_fit_error<0)?0:1e300;
    }
    if ( std::isinf(complexity_penalty)){
        complexity_penalty = (complexity_penalty<0)?0:1e300;
    } 
    // end of function trimming

    return -model_fit_error - complexity_penalty - normalization_const;
}

// search for the maximum likelihood parameters of the kernel
void gaussian_process::optimize_kernel_parameters(double step_size, double stopping_criterion){
    int d = kernel.parameters.length();

    std::cout<<"kernel params before optimization\t";
    for(int i=0; i<d; i++){ 
       std::cout<<" "<<kernel.parameters[i];
    }
    std::cout<<std::endl;

    kernel.search_step_size = step_size;
    kernel.best_log_l = std::numeric_limits<float>::max();
    kernel.iters = 0;

    double epsg = stopping_criterion;
    double epsf = 0;
    double epsx = 0;
    alglib::ae_int_t maxits = 1000;
    alglib::minlbfgsreport rep;
    alglib::minlbfgscreate(4, kernel.parameters, kernel.state);
    alglib::minlbfgssetcond(kernel.state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize2(kernel.state, kernel.gradient);
    alglib::minlbfgsresults(kernel.state, kernel.parameters, rep);

    std::cout<<"kernel params after optimization\t";
    for(int i=0; i<d; i++){ 
       std::cout<<" "<<kernel.best_parameters[i];
       kernel.parameters[i] = kernel.best_parameters[i];
    }
    kernel.observation_noise = kernel.best_observation_noise;
    init(kernel.parameters,kernel.observation_noise);
    std::cout<<std::endl;
    std::cout<<"likelihood: \t"<<log_marginal_likelihood()<<std::endl;
}


//================================== kernel functions ===================================//
void gaussian_process::set_SE_kernel(){
    // parameters correspond to (sigma_f^2, 1/(2*l_1), ... , 1/(2*l_d), sigma_n^2)
    // kernel function for evaluations
    std::function<kernel_func> se_kernel = [this](VectorXd x_i, VectorXd &x_j, const alglib::real_1d_array &parameters){
        //TODO receive distance function as parameter
        int d = x_i.size();
        double dist=0;
        for (int i=0; i<d; i++){
            dist += (x_i[i]-x_j[i])*(x_i[i]-x_j[i])*parameters(i+1);
        }
        return parameters(0)*parameters(0)*std::exp(-0.5*dist);
    };

    // gradient of log likelihood
    std::function<gradient_func> se_gradient = [this](const alglib::real_1d_array &x, double &func, alglib::real_1d_array &grad, void *ptr){
        int d = this->X.rows();
        int n = this->X.cols();
        int param_d = x.length();
        double two_sigma_f = 2*x(0);
        double two_sigma_n = 2*x(d+1);
        double minus_half_sigma_f_sq = (-0.5*x(0)*x(0));
        double dist, exp_ij;
        VectorXd x_i,x_j,length_scales(d);
        
        // update K and Kinv
        this->init(x, x(d+1));
        this->kernel.iters++;
        func = -this->log_marginal_likelihood();
        if (!std::isinf(func) && func<=this->kernel.best_log_l){
            std::cout<<"======== iteration "<<this->kernel.iters<<" ========"<<std::endl;
            this->kernel.best_parameters = alglib::real_1d_array(x);
            this->kernel.best_log_l = func;
            this->kernel.best_observation_noise= kernel.observation_noise;
            std::cout<<"new best solution:\t";
            for(int i=0; i<param_d; i++){ 
               std::cout<<" "<<this->kernel.best_parameters[i];
            }
            std::cout<<", value: "<<this->kernel.best_log_l<<std::endl;
        }
        // compute (-(K^{-1}*y)*(K^{-1}*y)^T- K^{-1})^{T}
        MatrixXd K_a = this->llt_of_K.solve(Y);
        K_a = (this->Kinv - (K_a)*(K_a).transpose()).transpose();
        //K_a = (0 - (K_a)*(K_a).transpose()).transpose();
        K_a = 0.5*this->kernel.search_step_size*K_a;

        for (int i=0; i<d; i++){
            length_scales[i] = x(i+1);
        }

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

                // first accumulate the gradient for sigma_f^2
                grad(0) = K_a(j,i);(  two_sigma_f*exp_ij );

                // the accumulate the gradient for the length scales
                for ( int k=0; k<d; k++){
                    grad(k+1) = K_a(j,i)*( minus_half_sigma_f_sq*(x_i[k]-x_j[k])*(x_i[k]-x_j[k])*exp_ij );
                }

                // finally accumulate the gradient for the noise parameter
                grad(d+1) = K_a(j,i)*( (i==j)?two_sigma_n:0 );
            }
        }
    };

    kernel.function = se_kernel;
    kernel.gradient = se_gradient;

    int d = X.rows();
    // number of parameters is d+2
    alglib::real_1d_array parameters;
    parameters.setlength(d+2);
    for (int i=0; i<parameters.length(); i++){
        parameters(i) = 1;
    }
    
    kernel.best_parameters = alglib::real_1d_array(parameters);
    double observation_noise = parameters(d+1);
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
