#include "gaussian_process.h"

// empty constructor
gaussian_process::gaussian_process(){
    debug_print=false;
    std::cout<<"CALLED THE EMPTY CONSTRUCTOR"<<std::endl;
}

gaussian_process::gaussian_process(int input_dimensions){
    debug_print=false;
    set_SE_kernel(input_dimensions);
}

// constructor with dataset
gaussian_process::gaussian_process(MatrixXd &Xin, MatrixXd &Yin){
    debug_print=false;
    X = Xin;
    Y = Yin;
    set_SE_kernel(X.rows());
}

// precomputations
void gaussian_process::init(const alglib::real_1d_array x, double observation_noise, bool noise_free){
    //    std::chrono::time_point<std::chrono::system_clock> start,end;
    //    std::chrono::duration<double> secs;
    kernel->parameters = alglib::real_1d_array(x);
    kernel->observation_noise = observation_noise;

    int n = X.cols();
    // kernel matrix ( !noise_free => K= K_noise_free + sigma_n*I
    //    start = std::chrono::system_clock::now();
    K = kernel_matrix(X,noise_free);
    //    end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"K: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
    llt_of_K = LLT<MatrixXd,Lower>(K);
    //    end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"llt of K: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
    if ( Kinv.cols() < n){
        Kinv = MatrixXd::Zero(n,n);
    } else {
        Kinv.setZero();
    }
    
    static Ref<MatrixXd> tmp = Kinv;
    static const LLT<MatrixXd,Lower>& llt_of_tmp = llt_of_K;
#pragma omp parallel shared(tmp,llt_of_tmp)
    {
#pragma omp for schedule (static)
        for (int i =0; i < n ; i++){
            tmp(i,i) = 1.0;
            tmp.col(i) = llt_of_tmp.solve(tmp.col(i));
        }
    }
    //    end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"Kinv: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
    //Kinv = llt_of_K.solve(MatrixXd::Identity(n,n));
    //    end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"Kinv: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
    KinvY = llt_of_K.solve(Y);
    //    end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"KinvY: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
    //log_det_K = 0;
    float logdetk = 0;
    //#pragma omp parallel for reduction(+:logdetk) num_threads(omp_get_num_procs()) schedule(static)
    for (int i =0; i < n ; i++){
        //log_det_K +=  std::log((llt_of_K.matrixL())(i,i));
        logdetk +=  std::log((llt_of_K.matrixL())(i,i));
    }
    //log_det_K = 2*log_det_K;
    log_det_K = 2*logdetk;
    //    end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"logdetk: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
    normalization_const = 0.5*n*std::log(2*PI);
}

void gaussian_process::init(const alglib::real_1d_array x){
    int d = kernel->parameters.length();
    init(kernel->parameters,kernel->parameters(d-1));
}

void gaussian_process::add_sample(VectorXd &x, double value){
    if (X.cols()==0){
        X = x;
        Y = VectorXd(1);
        Y[0]= value;
    }
    else{
        X.conservativeResize(X.rows(),X.cols()+1);
        X.col(X.cols()-1) = x;
        Y.conservativeResize(Y.size()+1);
        Y(Y.size()-1) = value;
    }
    // update the likelihood of the dataset given the best parameters found so far
    kernel->best_log_l = log_marginal_likelihood();
    init(kernel->parameters);
}

int gaussian_process::dataset_size(){
    return X.cols();
}

int gaussian_process::input_dimensions(){
    return X.rows();
}


// compute the maximum variance in the model predictions for samples in the dataset (should be close to 0 if noise is 0)
void gaussian_process::predictive_error_and_variance(VectorXd &error, VectorXd &variance, int type){
    int n = X.cols();
    VectorXd x_i;

    error.conservativeResize(n);
    variance.conservativeResize(n);
    std::cout<<"n: "<<n<<std::endl;
    // compute the GP prediction with the full dataset. If the noise is 0, the error and the variance should be 0
    if (type == 0){
        VectorXd mean;
        for(int i=0; i<n; i++){
            x_i = X.col(i);
            prediction(x_i,mean,variance(i));
            error(i) =  (Y(i) - mean(0))*(Y(i) -mean(0));
        }
    }
    // compute the LOO GP prediction (i.e. for each datapoint, compute the prediction and variance of a GP that does not include the data point)
    else if (type == 1){
        double mean_i;
        for(int i=0; i<n; i++){
            mean_i = Y(i) - KinvY(i)/Kinv(i,i);
            variance(i) = 1.0/Kinv(i,i);
            error(i) =  (Y(i) - mean_i)*(Y(i) -mean_i);
        }
    }
}

// compute the covariance vector between a sample point x, and the samples in the dataset X
VectorXd gaussian_process::kernel_vector(VectorXd &x){
    static int n = X.cols();
    VectorXd kx = VectorXd(n);
    //#pragma omp parallel for num_threads(omp_get_num_procs()) schedule(static)
    for(int i=0; i<n; i++){
        VectorXd x_i = X.col(i);
        kx[i]=kernel->function(x_i,x,kernel->parameters);
    }
    return kx;
}

// computes the covariance matrix between the samples in the dataset
MatrixXd gaussian_process::kernel_matrix(MatrixXd &X, bool noise_free){
    static int n = X.cols();
    K.setZero(n,n);

#pragma omp parallel for num_threads(omp_get_num_procs()) schedule(guided)
    for (int j=0; j<n ; j++){
        VectorXd x_j = X.col(j);
        for (int i=j; i<n ; i++){
            VectorXd x_i = X.col(i);
            K(i,j) = kernel->function(x_i,x_j,kernel->parameters);
            if (i==j){
                if(!noise_free){
                    K(i,j) += kernel->observation_noise*kernel->observation_noise;
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
    mean = kx.transpose()*KinvY;
    // compute variance estimate
    VectorXd v = llt_of_K.matrixL().solve(kx);
    variance = kernel->function(x,x,kernel->parameters) - v.dot(v);
    if (variance < 0.0)
        variance = 0.0;
}

// compute the log marginal likelihood of the function values Y, given the inputs X and the kernel parameters
double gaussian_process::log_marginal_likelihood(){
    double model_fit_error = 0.5*Y.dot(KinvY);
    double complexity_penalty = 0.5*log_det_K;
    if ((std::isnan(model_fit_error+complexity_penalty)||std::isinf(model_fit_error+complexity_penalty)) && debug_print){
        std::cout<<"x: ";
        for(int i=0; i<kernel->parameters.length(); i++){ 
            std::cout<<" "<<kernel->parameters(i);
        }
        std::cout<<std::endl;
        std::cout<<"fit error: "<<model_fit_error<<" complexity penalty: "<<complexity_penalty<<" normalization constant: "<<normalization_const<<std::endl;
    }
    return -model_fit_error - complexity_penalty - normalization_const;
}

double gaussian_process::leave_one_out_log_probability(){
    double loo_log_probability = 0;
    static int n = X.cols();
    double mean_i;
    double variance_i;

    for(int i=0; i<n; i++){
        mean_i = Y(i) - KinvY(i)/Kinv(i,i);
        variance_i = 1/Kinv(i,i);
        loo_log_probability += -std::log(variance_i) - (Y(i) - mean_i)*(Y(i) - mean_i)/variance_i;
    }
    loo_log_probability *= 0.5;
    loo_log_probability -= normalization_const;
    return loo_log_probability;

}

//select random staring point
void gaussian_process::set_opt_starting_point(VectorXd point){
    static int d = kernel->parameters.length();
    for(int i=0; i<d; i++){ 
        kernel->parameters[i] = point[i];
    }
}

void gaussian_process::set_opt_random_start(double scale, double offset){
    VectorXd init_params = VectorXd::Random(kernel->parameters.length())*scale;
    for (int i=0; i<init_params.size(); i++){
        init_params(i) += offset;
    }
    set_opt_starting_point(init_params);
}
// search for the maximum likelihood parameters of the kernel
void gaussian_process::optimize_parameters(double stopping_criterion, int solver){
    int d = kernel->parameters.length();

    if (debug_print){
        std::cout<<"Best kernel parameters before optimization: \t";
        for(int i=0; i<d; i++){ 
            std::cout<<" "<<kernel->best_parameters[i];
        }
        std::cout<<std::endl;
        std::cout<<"likelihood: \t"<<kernel->best_log_l<<std::endl;

        std::cout<<"Starting point: \t";
        for(int i=0; i<d; i++){ 
            std::cout<<" "<<kernel->parameters[i];
        }
        std::cout<<std::endl;
    }
    kernel->iters = 0;

    double epsg = stopping_criterion;
    double epsf = 0;
    double epsx = 0;
    static alglib::ae_int_t maxits = 50;
    try{
        if (solver == ALGLIB_SOLVER_LBFGS){
            if (debug_print)
                std::cout<<"\tOptimizing with the L-BFGS algorithm"<<std::endl;
            alglib::minlbfgsreport rep;
            alglib::minlbfgscreate(d,kernel->parameters, kernel->bfgsstate);
            alglib::minlbfgssetcond(kernel->bfgsstate, epsg, epsf, epsx, maxits);
            alglib::minlbfgsoptimize2(kernel->bfgsstate, kernel->gradient);
            alglib::minlbfgsresults(kernel->bfgsstate, kernel->parameters, rep);
            if (debug_print){
                std::cout<<"\tIterations: "<<rep.iterationscount
                    <<", Function Evaluations: "<<rep.nfev
                    <<", VarIdx: "<<rep.varidx
                    <<", Termination Type: "<<rep.terminationtype<<std::endl;
            }
        }else if (solver == ALGLIB_SOLVER_NUM_LBFGS){
            if (debug_print)
                std::cout<<"\tOptimizing with the L-BFGS algorithm (with numerical differentiation)"<<std::endl;
            alglib::minlbfgsreport rep;
            alglib::minlbfgscreatef(d,kernel->parameters,1e-5, kernel->bfgsstate);
            alglib::minlbfgssetcond(kernel->bfgsstate, epsg, epsf, epsx, maxits);
            alglib::minlbfgsoptimize2(kernel->bfgsstate, kernel->function_alglib);
            alglib::minlbfgsresults(kernel->bfgsstate, kernel->parameters, rep);
            if (debug_print){
                std::cout<<"\tIterations: "<<rep.iterationscount
                    <<", Function Evaluations: "<<rep.nfev
                    <<", VarIdx: "<<rep.varidx
                    <<", Termination Type: "<<rep.terminationtype<<std::endl;
            }
        }else if (solver == ALGLIB_SOLVER_CONSTRAINED_CG){
            if (debug_print)
                std::cout<<"\tOptimizing with constrained CG algorithm"<<std::endl;
            alglib::real_1d_array lbound,ubound;
            lbound.setlength(d);
            ubound.setlength(d);
            for (int i=0; i<d; i++){
                lbound[i] = 1e-9;
                ubound[i] = 1e9;
            }
            alglib::minbleicreport rep;
            alglib::minbleiccreate(kernel->parameters, kernel->bleicstate);
            alglib::minbleicsetbc(kernel->bleicstate,lbound,ubound);
            alglib::minbleicsetcond(kernel->bleicstate, epsg, epsf, epsx, maxits);
            alglib::minbleicoptimize2(kernel->bleicstate, kernel->gradient);
            alglib::minbleicresults(kernel->bleicstate, kernel->parameters, rep);
            if (debug_print){
                std::cout<<"\tIterations: "<<rep.iterationscount
                    <<", Function Evaluations: "<<rep.nfev
                    <<", VarIdx: "<<rep.varidx
                    <<", Termination Type: "<<rep.terminationtype<<std::endl;
            }
        } else {
            if (debug_print)
                std::cout<<"\tOptimizing with the CG algorithm"<<std::endl;
            alglib::mincgreport rep;
            alglib::mincgcreate(kernel->parameters, kernel->cgstate);
            alglib::mincgsetcond(kernel->cgstate, epsg, epsf, epsx, maxits);
            alglib::mincgoptimize2(kernel->cgstate, kernel->gradient);
            alglib::mincgresults(kernel->cgstate, kernel->parameters, rep);
            if (debug_print){
                std::cout<<"\tIterations: "<<rep.iterationscount
                    <<", Function Evaluations: "<<rep.nfev
                    <<", VarIdx: "<<rep.varidx
                    <<", Termination Type: "<<rep.terminationtype<<std::endl;
            }
        }
    } catch (alglib::ap_error e){
        std::cout<<"Caught exception: "<<e.msg<<std::endl;
    }

    // set current parameters to the best found so far
    init(kernel->best_parameters,kernel->best_parameters(d-1));
    kernel->best_log_l = log_marginal_likelihood();

    if (debug_print){
        std::cout<<"kernel params after optimization\t";
        for(int i=0; i<d; i++){ 
            std::cout<<" "<<kernel->best_parameters[i];
        }
        std::cout<<std::endl;
        std::cout<<"likelihood: \t"<<log_marginal_likelihood()<<std::endl;
    }
}

void gaussian_process::optimize_parameters_random_restarts(double stopping_criterion, int solver, int restarts, double scale){
    static int param_d = kernel->parameters.length();

    for(int i=0; i<restarts;i++){
        // set starting point
        VectorXd starting_point = 0.5*VectorXd::Random(param_d);
        for (int i=0; i<param_d; i++){
            starting_point[i] *= scale*kernel->parameters[i];
            // in these cases, the parameters should be always positive
            if (solver == ALGLIB_SOLVER_CONSTRAINED_CG 
                    || kernel->id == KERNEL_SQUARED_EXPONENTIAL
                    || kernel->id == KERNEL_RBF
               ){
                starting_point[i] *= starting_point[i]>0?1.0:-1.0;
            }
        }
        set_opt_starting_point(starting_point);

        // optimize
        if (debug_print){
            std::cout<<"------> Run #"<<i<<std::endl;
        }
        optimize_parameters(stopping_criterion, solver);

    }
};

void gaussian_process::set_debug_print(bool dbg_prnt){
    debug_print = dbg_prnt;
}

//================================== kernel functions ===================================//
void gaussian_process::set_SE_kernel(int input_dimensions){
    // parameters correspond to (sigma_f^2, 1/(2*l_1), ... , 1/(2*l_d), sigma_n^2)
    // kernel function for evaluations
    std::function<kernel_func> se_kernel = [this](VectorXd x_i, VectorXd &x_j, const alglib::real_1d_array &parameters){
        //TODO receive distance function as parameter
        static int d = x_i.size();
        double dist=0;
        for (int i=0; i<d; i++){
            //dist += (x_i[i]-x_j[i])*(x_i[i]-x_j[i])*std::fabs(parameters(i+1));
            dist += (x_i[i]-x_j[i])*(x_i[i]-x_j[i])/std::fabs(parameters(i+1));
        }
        return parameters(0)*parameters(0)*std::exp(-0.5*dist);
    };

    // gradient of log likelihood
    std::function<kernel_func_alglib> se_func= [this](const alglib::real_1d_array &params, double &func, void *ptr){
        static int d = this->X.rows();
        // update K and Kinv
        this->init(params, params(d+1));
        this->kernel->iters++;
        func = -this->log_marginal_likelihood();
        if (!std::isinf(func) && func<=-1.0*this->kernel->best_log_l){
            this->kernel->best_parameters = alglib::real_1d_array(params);
            this->kernel->best_log_l = -1.0*func;
        }
    };

    // gradient of log likelihood
    std::function<gradient_func> se_gradient = [this](const alglib::real_1d_array &params, double &func, alglib::real_1d_array &grad, void *ptr){
        //        std::chrono::time_point<std::chrono::system_clock> start,end;
        //        std::chrono::duration<double> secs;

        static int d = this->X.rows();
        static int n = this->X.cols();
        static int param_d = params.length();
        double two_sigma_f = 2.0*std::fabs(params(0));
        double sigma_n_sq = params(d+1)*params(d+1);
        double two_sigma_n = 2.0*std::fabs(params(d+1));
        double one_over_sigma_f_sq = 1.0/(params(0)*params(0));
        double minus_half_sigma_f_sq = (-0.5*params(0)*params(0));
        //double exp_ij;
        //VectorXd x_i,x_j;
        //        start = std::chrono::system_clock::now();
        // update K and Kinv
        this->init(params, params(d+1));
        //        end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"init: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
        this->kernel->iters++;
        //compute negative log likelihood
        func = -this->log_marginal_likelihood();
        //        end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"lml: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();

        //check if this is the best set of parameters we have obtained so far
        if (!std::isinf(func) && func<=-1.0*this->kernel->best_log_l){
            this->kernel->best_parameters = alglib::real_1d_array(params);
            this->kernel->best_log_l = -1.0*func;
        }
        //        end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"updatebest: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();

        // compute (-(K^{-1}*y)*(K^{-1}*y)^T- K^{-1})^{T}
        MatrixXd  K_a = (this->KinvY*(this->KinvY.transpose()) - this->Kinv);
        //        end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"Ka: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();

        // here we are computing the (negative) partial derivatives as
        //                         -1/2*trace{ ( (K^{-1}*y)*(K^{-1}*y)^T- K^{-1} )*dK/d_param ) }
        //                       = -1/2*trace { K_a*dK/d_param }
        //                       = -1/2*trace { K_a^{T}*dK/d_param }  (K_a is symmetric)
        //                       = -1/2*sum_row { sum_col { { K_a ** dK/d_param } }  (** means element-wise product)

        // initialize partial derivatives to 0
        for(int i=0; i<param_d; i++){ 
            grad(i)=0;
        }

        // accumulate the trace value
#pragma omp parallel shared(grad) num_threads(omp_get_num_procs())
        {
            alglib::real_1d_array tmp;
            tmp.setlength(param_d);
            for(int i=0; i<param_d; i++){ 
                tmp(i)=0;
            }
            VectorXd x_i;
            VectorXd x_j;
            double exp_ij;
#pragma omp for collapse(2) schedule(static)
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
                    tmp(0) += K_a(j,i)*( two_sigma_f*exp_ij );

                    // then accumulate the gradient for the length scales
                    for ( int k=0; k<d; k++){
                        //grad(k+1) += K_a(j,i)*( minus_half_sigma_f_sq*(x_i[k]-x_j[k])*(x_i[k]-x_j[k])*exp_ij );
                        tmp(k+1) += K_a(j,i)*( -(1/(params(k+1)*params(k+1)))*minus_half_sigma_f_sq*(x_i[k]-x_j[k])*(x_i[k]-x_j[k])*exp_ij );
                    }
                    // finally accumulate the gradient for the noise parameter
                    tmp(d+1) += K_a(j,i)*( (i==j)?two_sigma_n:0 );
                }
            }
#pragma omp critical
            for(int i=0; i<param_d; i++){ 
                grad(i)+=-0.5*tmp(i);
            }
        }
        //        end = std::chrono::system_clock::now(); secs = end - start; std::cout<<"grad: "<<secs.count()<<" secs."<<std::endl; start = std::chrono::system_clock::now();
    };

    kernel = new kernel_object(se_kernel, se_gradient, se_func);
    kernel->id = KERNEL_SQUARED_EXPONENTIAL;

    // number of parameters is d+2
    alglib::real_1d_array parameters;
    parameters.setlength(input_dimensions+2);
    for (int i=0; i<parameters.length(); i++){
        parameters(i) = 1.0;
    }

    kernel->best_parameters = alglib::real_1d_array(parameters);
    init(parameters,parameters(input_dimensions+1));
    kernel->best_log_l = log_marginal_likelihood();
}

void gaussian_process::set_RBF_kernel(){
    // parameters correspond to (sigma_f^2, 1/(2*l_1), ... , 1/(2*l_d), sigma_n^2)
    // kernel function for evaluations
    std::function<kernel_func> rbf_kernel = [this](VectorXd x_i, VectorXd &x_j, const alglib::real_1d_array &parameters){
        //TODO receive distance function as parameter
        int d = x_i.size();
        double dist=0;
        for (int i=0; i<d; i++){
            //dist += (x_i[i]-x_j[i])*(x_i[i]-x_j[i])*std::fabs(parameters(1));
            dist += (x_i[i]-x_j[i])*(x_i[i]-x_j[i])/std::fabs(parameters(1));
        }
        return parameters(0)*parameters(0)*std::exp(-0.5*dist);
    };

    // gradient of log likelihood
    std::function<kernel_func_alglib> rbf_func= [this](const alglib::real_1d_array &params, double &func, void *ptr){
        // update K and Kinv
        this->init(params, params(2));
        this->kernel->iters++;
        func = -this->log_marginal_likelihood();
        if (!std::isinf(func) && func<=-1.0*this->kernel->best_log_l){
            this->kernel->best_parameters = alglib::real_1d_array(params);
            this->kernel->best_log_l = -1.0*func;
        }
    };

    // gradient of log likelihood
    std::function<gradient_func> rbf_gradient = [this](const alglib::real_1d_array &params, double &func, alglib::real_1d_array &grad, void *ptr){
        int d = this->X.rows();
        int n = this->X.cols();
        int param_d = params.length();
        double two_sigma_f = 2.0*std::fabs(params(0));
        double sigma_n_sq = params(2)*params(2);
        double two_sigma_n = 2.0*std::fabs(params(2));
        double one_over_sigma_f_sq = 1.0/(params(0)*params(0));
        double minus_half_sigma_f_sq = (-0.5*params(0)*params(0));
        //double exp_ij;
        //VectorXd x_i,x_j;

        // update K and Kinv
        this->init(params, params(2));
        this->kernel->iters++;
        //compute negative log likelihood
        func = -this->log_marginal_likelihood();

        //check if this is the best set of parameters we have obtained so far
        if (!std::isinf(func) && func<= -1.0*this->kernel->best_log_l){
            this->kernel->best_parameters = alglib::real_1d_array(params);
            this->kernel->best_log_l = -1.0*func;
        }

        // compute (-(K^{-1}*y)*(K^{-1}*y)^T- K^{-1})^{T}
        MatrixXd  K_a = (this->KinvY*(this->KinvY.transpose()) - this->Kinv);

        // here we are computing the (negative) partial derivatives as
        //                         -1/2*trace{ ( (K^{-1}*y)*(K^{-1}*y)^T- K^{-1} )*dK/d_param ) }
        //                       = -1/2*trace { K_a*dK/d_param }
        //                       = -1/2*trace { K_a^{T}*dK/d_param }  (K_a is symmetric)
        //                       = -1/2*sum_row { sum_col { { K_a ** dK/d_param } }  (** means element-wise product)

        // initialize partial derivatives to 0
        for(int i=0; i<param_d; i++){ 
            grad(i)=0;
        }

        // accumulate the trace value
#pragma omp parallel shared(grad) num_threads(omp_get_num_procs())
        {
            alglib::real_1d_array tmp;
            tmp.setlength(param_d);
            for(int i=0; i<param_d; i++){ 
                tmp(i)=0;
            }
            VectorXd x_i;
            VectorXd x_j;
            double exp_ij;
#pragma omp for collapse(2) schedule(static)
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
                    tmp(0) += K_a(j,i)*( two_sigma_f*exp_ij );
                    // then accumulate the gradient for the length scale
                    for ( int k=0; k<d; k++){
                        //grad(1) += K_a(j,i)*( minus_half_sigma_f_sq*(x_i[k]-x_j[k])*(x_i[k]-x_j[k])*exp_ij );
                        tmp(1) += K_a(j,i)*( -(1/(params(1)*params(1)))*minus_half_sigma_f_sq*(x_i[k]-x_j[k])*(x_i[k]-x_j[k])*exp_ij );
                    }
                    // finally accumulate the gradient for the noise parameter
                    tmp(2) += K_a(j,i)*( (i==j)?two_sigma_n:0 );
                }
            }
#pragma omp critical
            for(int i=0; i<param_d; i++){ 
                grad(i)+=-0.5*tmp(i);
            }
        }
    };
    kernel = new kernel_object(rbf_kernel, rbf_gradient, rbf_func);
    kernel->id = KERNEL_RBF;

    // number of parameters is d+2
    alglib::real_1d_array parameters;
    parameters.setlength(3);
    for (int i=0; i<parameters.length(); i++){
        parameters(i) = 1.0;
    }

    kernel->best_parameters = alglib::real_1d_array(parameters);
    init(parameters,parameters(2));
    kernel->best_log_l = log_marginal_likelihood();
}

//==================================== kernel object =====================================//

kernel_object::kernel_object( ){

}

kernel_object::kernel_object(std::function<kernel_func> &k,
        std::function<gradient_func> &g,
        std::function<kernel_func_alglib> &f){
    function = k;
    function_alglib = f;
    gradient = g;
}
