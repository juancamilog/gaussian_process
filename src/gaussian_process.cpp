#include "gaussian_process.h"

// empty constructor
gaussian_process::gaussian_process(){
    set_SE_kernel();
    maximum_variance=0;
}

// constructor with dataset
gaussian_process::gaussian_process(MatrixXd &Xin, MatrixXd &Yin){
    X = Xin;
    Y = Yin;
    set_SE_kernel();
    maximum_variance=0;
}

// precomputations
void gaussian_process::init(const alglib::real_1d_array &x, double observation_noise, bool noise_free){
    kernel.parameters = alglib::real_1d_array(x);
    kernel.observation_noise = observation_noise;

    int n = X.cols();
    // kernel matrix ( !noise_free => K= K_noise_free + sigma_n*I
    K = kernel_matrix(X,noise_free);
    llt_of_K = LLT<MatrixXd,Lower>(K);
    Kinv = llt_of_K.solve(MatrixXd::Identity(n,n));
    KY = llt_of_K.solve(Y);
    log_det_K = 0;
    for (int i =0; i < n ; i++){
        log_det_K +=  std::log((llt_of_K.matrixL())(i,i)*(llt_of_K.matrixL())(i,i));
    }
    normalization_const = 0.5*n*std::log(2*PI);
}

void gaussian_process::add_sample(VectorXd &x, double value){
    if (X.cols()==0){
       X = x;
    }
    else{
       X.conservativeResize(X.rows(),X.cols()+1);
       X.col(X.cols()-1) = x;
       Y.conservativeResize(Y.size()+1);
       Y(Y.size()-1) = value;
    }
}

int gaussian_process::dataset_size(){
    return X.cols();
}

double gaussian_process::get_maximum_variance(){
    return maximum_variance;
}

// compute the maximum variance in the model predictions for samples in the dataset (should be close to 0 if noise is 0)
double gaussian_process::compute_maximum_variance(){
    int n = X.cols();
    VectorXd x_i;
    VectorXd v;
    VectorXd mean;
    double variance;

    for(int i=0; i<n; i++){
        x_i = X.col(i);
        prediction(x_i,mean,variance);
        if (variance> maximum_variance){
            maximum_variance = variance;
        }
    }
    return maximum_variance;
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
MatrixXd gaussian_process::kernel_matrix(MatrixXd &X, bool noise_free){
    int n = X.cols();
    K.setZero(n,n);
    VectorXd x_i,x_j;

    for (int j=0; j<n ; j++){
        for (int i=j; i<n ; i++){
            x_i = X.col(i);
            x_j = X.col(j);
            K(i,j) = kernel.function(x_i,x_j,kernel.parameters);
            if (i==j){
                if(!noise_free){
                    K(i,j) += kernel.observation_noise*kernel.observation_noise;
                }
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
    variance = kernel.function(x,x,kernel.parameters) - v.dot(v);
}

// compute the log marginal likelihood of the function values Y, given the inputs X and the kernel parameters
double gaussian_process::log_marginal_likelihood(){
    double model_fit_error = 0.5*Y.dot(KY);
    double complexity_penalty = 0.5*log_det_K;
    if (std::isnan(model_fit_error+complexity_penalty)||std::isinf(model_fit_error+complexity_penalty)){
        std::cout<<"x: ";
        for(int i=0; i<kernel.parameters.length(); i++){ 
           std::cout<<" "<<kernel.parameters(i);
        }
        std::cout<<" K: \n"<<K<<std::endl;
        std::cout<<" Kinv: \n"<<Kinv<<std::endl;
        std::cout<<" Y: \n"<<Y.transpose()<<std::endl;
        std::cout<<" KY: \n"<<KY.transpose()<<std::endl;

        std::cout<<"fit error: "<<model_fit_error<<" complexity penalty: "<<complexity_penalty<<" normalization constant: "<<normalization_const<<std::endl;
    }
    return -model_fit_error - complexity_penalty - normalization_const;
}

//select random staring point
void gaussian_process::set_opt_starting_point(VectorXd point){
    int d = kernel.parameters.length();
    for(int i=0; i<d; i++){ 
       kernel.parameters[i] = point[i];
    }
}

// search for the maximum likelihood parameters of the kernel
void gaussian_process::optimize_parameters(double stopping_criterion, int solver){
    int d = kernel.parameters.length();

    std::cout<<"Kernel parameters before optimization\t";
    for(int i=0; i<d; i++){ 
       std::cout<<" "<<kernel.parameters[i];
    }
    std::cout<<std::endl;

    kernel.best_log_l = std::numeric_limits<float>::max();
    kernel.iters = 0;

    double epsg = stopping_criterion;
    double epsf = 0;
    double epsx = 0;
    alglib::ae_int_t maxits = 0;
    try{
        if (solver == 0){
            std::cout<<"Optimizing with the L-BFGS algorithm"<<std::endl;
            alglib::minlbfgsreport rep;
            alglib::minlbfgscreate(4,kernel.parameters, kernel.bfgsstate);
            alglib::minlbfgssetcond(kernel.bfgsstate, epsg, epsf, epsx, maxits);
            alglib::minlbfgsoptimize2(kernel.bfgsstate, kernel.gradient);
            alglib::minlbfgsresults(kernel.bfgsstate, kernel.parameters, rep);
            std::cout<<"Iterations: "<<rep.iterationscount<<", Function Evaluations: "<<rep.nfev<<", VarIdx: "<<rep.varidx<<", Termination Type: "<<rep.terminationtype<<std::endl;
        }else if (solver == 1){
            std::cout<<"Optimizing with the L-BFGS algorithm (with numerical differentiation)"<<std::endl;
            alglib::minlbfgsreport rep;
            alglib::minlbfgscreatef(4,kernel.parameters,1e-5, kernel.bfgsstate);
            alglib::minlbfgssetcond(kernel.bfgsstate, epsg, epsf, epsx, maxits);
            alglib::minlbfgsoptimize2(kernel.bfgsstate, kernel.function_alglib);
            alglib::minlbfgsresults(kernel.bfgsstate, kernel.parameters, rep);
            std::cout<<"Iterations: "<<rep.iterationscount<<", Function Evaluations: "<<rep.nfev<<", VarIdx: "<<rep.varidx<<", Termination Type: "<<rep.terminationtype<<std::endl;
        }else if (solver == 2){
            std::cout<<"Optimizing with constrained CG algorithm"<<std::endl;
            alglib::real_1d_array lbound,ubound;
            lbound.setlength(d);
            ubound.setlength(d);
            for (int i=0; i<d; i++){
                lbound[i] = 0.0;
                ubound[i] = 1e3;
            }
            alglib::minbleicreport rep;
            alglib::minbleiccreate(kernel.parameters, kernel.bleicstate);
            alglib::minbleicsetbc(kernel.bleicstate,lbound,ubound);
            alglib::minbleicsetcond(kernel.bleicstate, epsg, epsf, epsx, maxits);
            alglib::minbleicoptimize2(kernel.bleicstate, kernel.gradient);
            alglib::minbleicresults(kernel.bleicstate, kernel.parameters, rep);
            std::cout<<"Iterations: "<<rep.iterationscount<<", Function Evaluations: "<<rep.nfev<<", VarIdx: "<<rep.varidx<<", Termination Type: "<<rep.terminationtype<<std::endl;
        } else {
            std::cout<<"Optimizing with the CG algorithm"<<std::endl;
            alglib::mincgreport rep;
            alglib::mincgcreate(kernel.parameters, kernel.cgstate);
            alglib::mincgsetcond(kernel.cgstate, epsg, epsf, epsx, maxits);
            alglib::mincgoptimize2(kernel.cgstate, kernel.gradient);
            alglib::mincgresults(kernel.cgstate, kernel.parameters, rep);
            std::cout<<"Iterations: "<<rep.iterationscount<<", Function Evaluations: "<<rep.nfev<<", VarIdx: "<<rep.varidx<<", Termination Type: "<<rep.terminationtype<<std::endl;
        }
    } catch (alglib::ap_error e){
        std::cout<<"Caught exception: "<<e.msg<<std::endl;
    }
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
            dist += (x_i[i]-x_j[i])*(x_i[i]-x_j[i])*std::fabs(parameters(i+1));
        }
        return parameters(0)*parameters(0)*std::exp(-0.5*dist);
    };

    // gradient of log likelihood
    std::function<kernel_func_alglib> se_func= [this](const alglib::real_1d_array &params, double &func, void *ptr){
        int d = this->X.rows();
        // update K and Kinv
        this->init(params, params(d+1));
        this->kernel.iters++;
        func = -this->log_marginal_likelihood();
        if (!std::isinf(func) && func<=this->kernel.best_log_l){
            this->kernel.best_parameters = alglib::real_1d_array(params);
            this->kernel.best_log_l = func;
            this->kernel.best_observation_noise= params(d+1);
        }
    };

    // gradient of log likelihood
    std::function<gradient_func> se_gradient = [this](const alglib::real_1d_array &params, double &func, alglib::real_1d_array &grad, void *ptr){
        int d = this->X.rows();
        int n = this->X.cols();
        int param_d = params.length();
        double two_sigma_f = 2.0*std::fabs(params(0));
        double sigma_n_sq = params(d+1)*params(d+1);
        double two_sigma_n = 2.0*std::fabs(params(d+1));
        double one_over_sigma_f_sq = 1.0/(params(0)*params(0));
        double minus_half_sigma_f_sq = (-0.5*params(0)*params(0));
        double exp_ij;
        VectorXd x_i,x_j;
        // update K and Kinv
        this->init(params, params(d+1));
        this->kernel.iters++;
        //compute negative log likelihood
        func = -this->log_marginal_likelihood();

        //check if this is the best set of parameters we have obtained so far
        if (!std::isinf(func) && func<=this->kernel.best_log_l){
            this->kernel.best_parameters = alglib::real_1d_array(params);
            this->kernel.best_log_l = func;
            this->kernel.best_observation_noise= params(d+1);
        }

        // compute (-(K^{-1}*y)*(K^{-1}*y)^T- K^{-1})^{T}
        MatrixXd  K_a = (KY*(KY.transpose()) - this->Kinv);

        // here we are computing the (negative) partial derivatives as
        //                         -1/2*trace{ ( (K^{-1}*y)*(K^{-1}*y)^T- K^{-1} )*dK/d_param ) }
        //                       = -1/2*trace { K_a*dK/d_param }
        //                       = -1/2*trace { K_a^{T}*dK/d_param }  (K_a is symmetric)
        //                       = -1/2*sum_row { sum_col { { K_a ** dK/d_param } }  (** means element-wise product)
        
        // initialize partial derivatives to 0
        for(int i=0; i<param_d; i++){ 
             grad(i)=0.0;
        }

        // accumulate the trace value
        for (int j=0; j<n ; j++){
            for (int i=0; i<n ; i++){
                x_i = this->X.col(i);
                x_j = this->X.col(j);
                if (i==j){
                    exp_ij = (K(i,j)-sigma_n_sq)*one_over_sigma_f_sq;
                }
                else{
                    exp_ij = (K(i,j))*one_over_sigma_f_sq;
                }
                // first accumulate the gradient for sigma_f^2
                grad(0) += K_a(j,i)*( two_sigma_f*exp_ij );
                // then accumulate the gradient for the length scales
                for ( int k=0; k<d; k++){
                    grad(k+1) += K_a(j,i)*( minus_half_sigma_f_sq*(x_i[k]-x_j[k])*(x_i[k]-x_j[k])*exp_ij );
                }
                // finally accumulate the gradient for the noise parameter
                grad(d+1) += K_a(j,i)*( (i==j)?two_sigma_n:0 );
            }
        }
        for(int i=0; i<param_d; i++){ 
           grad(i)=-0.5*grad(i);
        }
        // debug info:
        /*
        IOFormat fmt(FullPrecision, 0, ", ", ",\n", "{", "}", "{", "}");
        std::cout<<"log likelihood: "<<-func<<std::endl;
        std::cout<<"x: ";
            for(int i=0; i<param_d; i++){ 
               std::cout<<" "<<params(i);
            }
        std::cout<<std::endl;
        std::cout<<"gradient: ";
            for(int i=0; i<param_d; i++){ 
               std::cout<<" "<<grad(i);
            }
        std::cout<<std::endl;
        std::cout<<" K: \n"<<this->K<<std::endl;
        std::cout<<" Kinv: \n"<<this->Kinv<<std::endl;
        std::cout<<" Ka: \n"<<K_a<<std::endl;
        std::cout<<" X: \n"<<this->X.format(fmt)<<std::endl;
        std::cout<<" Y: \n"<<this->Y.transpose().format(fmt)<<std::endl;
        std::cout<<" KY: \n"<<this->KY.transpose()<<std::endl;
        */
    };

    kernel.function = se_kernel;
    kernel.function_alglib = se_func;
    kernel.gradient = se_gradient;

    int d = X.rows();
    // number of parameters is d+2
    alglib::real_1d_array parameters;
    parameters.setlength(d+2);
    for (int i=0; i<parameters.length(); i++){
        parameters(i) = 1.0;
    }
    
    kernel.best_parameters = alglib::real_1d_array(parameters);
    init(parameters,parameters(d+1));
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
