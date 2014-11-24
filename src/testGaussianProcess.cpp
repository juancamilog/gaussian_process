#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <gaussian_process.h>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
//#include "mgl2/mgl.h"

std::random_device rd;
std::mt19937 gen(rd());


double step_size = 0.05;
double noise = 1.0;
int n_train = 500;
int optimization_algorithm = 0;
typedef std::pair<VectorXd,double> data_point;

double func(double x, double y, double observation_noise=0){
    std::normal_distribution<double> dist(0,observation_noise);
    return 255*(0.8*std::exp(-0.5*(x*x+y*y))  + 0.2*std::sin(0.25*(x+y))) + dist(gen);
};


void parse_args(int argc,char* argv[]){
    int i =0;
    while (i< argc){
        std::string t = std::string(argv[i]);
        if( t == "-t" || t == "--n_train"){
            n_train = atoi(argv[i+1]);
            i=i+2;
        }
        else if (t == "-s" || t =="--noise"){
            noise = atof(argv[i+1]);
            i=i+2;
        }
        else if (t == "-c" || t =="--cell_size"){
            step_size = atof(argv[i+1]);
            i=i+2;
        }
        else if (t == "-a" || t =="--optimization-algorithm"){
            optimization_algorithm = atoi(argv[i+1]);
            i=i+2;
        }
        else{
            i=i+1;
        }
    }
};

int main(int argc, char* argv[])
{
    Eigen::setNbThreads(2);
    parse_args(argc,argv);

    std::vector<data_point> f;
    // this is our data
    for (double x0 = -20; x0<20; x0=x0+step_size){
      for (double x1 = -20; x1<20; x1=x1+step_size){
          VectorXd xp(2);
          xp(0)=x0; xp(1)=x1;
          f.push_back(data_point(xp,func(x0,x1,noise)));
      }
    } 
    //std::cout<< f<<endl;
    int n = f.size();
    // pick n_train samples from f
    std::shuffle(f.begin(),f.end(),gen);
    std::cout<<"n: "<<n<<"n_train: "<<n_train<<std::endl;
    MatrixXd X(2,n_train);
    MatrixXd Y(n_train,1);
    for (int i = 0; i<n_train; i++){
        X(0,i) = f[i].first[0];
        X(1,i) = f[i].first[1];
        Y(i,0) = f[i].second;
    } 
    
    // this is only one way of doing it, you can also add samples one by one, with GP.add_sample(VectorXd x, double y)
    gaussian_process GP(X,Y);

    GP.set_debug_print(true);

    GP.set_SE_kernel(X.rows());

    std::cout.precision(5);

    std::chrono::time_point<std::chrono::system_clock> start,end; 
    std::chrono::duration<double> secs;

    start = std::chrono::system_clock::now();
    //Vector4d init_params = VectorXd::Random(4).cwiseAbs()*10;
    //GP.set_opt_starting_point(init_params);
    std::srand(std::time(NULL));
    GP.optimize_parameters_random_restarts(1e-6,optimization_algorithm, 10,5);

    end = std::chrono::system_clock::now();

    secs = end - start;
    std::cout<<"Took "<< secs.count()<<" seconds."<<std::endl;

    // compare actual values with prediction
    double variance;
    VectorXd mean;
    for (int i = n_train; i<n_train+100; i++){
        VectorXd x = f[i].first;
        GP.prediction(x,mean,variance);
        std::cout<<std::fixed
                 <<"x: "<<x.transpose()
                 <<",\tf(x): "<<func(x[0], x[1])
                 <<",\tf*(x): "<<mean
                 <<",\tstd(x): "<<std::sqrt(variance)<<std::endl;
    } 
    std::cout<<"n="<<n<<", n_train="<<n_train<<std::endl;

    VectorXd err, var;
    
    GP.predictive_error_and_variance(err,var,1);

    //std::cout<<"Dataset Error:"<<err.transpose()<<std::endl;
    //std::cout<<"Dataset Variance:"<<var.transpose()<<std::endl;
   
    VectorXd xp(2);
    xp(0)=100; xp(1)=100;
    GP.prediction(xp,mean,variance);

    std::cout<<std::fixed
             <<"x: "<<xp.transpose()
             <<",\tf(x): "<<func(xp[0], xp[1])
             <<",\tf*(x): "<<mean
             <<",\tstd(x): "<<std::sqrt(variance)<<std::endl;
}
