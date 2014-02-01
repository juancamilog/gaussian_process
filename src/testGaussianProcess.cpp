#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <gaussian_process.h>
#include <random>
#include <vector>
#include <algorithm>
//#include "mgl2/mgl.h"

std::random_device rd;
std::mt19937 gen(rd());

typedef std::pair<VectorXd,double> data_point;

double sine(double x, double y, double observation_noise=0){
    std::normal_distribution<double> dist(0,observation_noise);
    return 10.0*std::sin((-(x-1)*(x-1)-y*y/400)/100) + dist(gen);
};


int main()
{
    double step_size = 0.1;
    double noise = 1.0;
    std::vector<data_point> f;
    // this is our data
    for (double x0 = -10; x0<10; x0=x0+step_size){
      for (double x1 = -10; x1<10; x1=x1+step_size){
          VectorXd xp(2);
          xp(0)=x0; xp(1)=x1;
          f.push_back(data_point(xp,sine(x0,x1,noise)));
      }
    } 
    //std::cout<< f<<endl;
    int n_train = 500;
    int n = f.size();
    // pick n_train samples from f
    std::shuffle(f.begin(),f.end(),gen);

    MatrixXd X(2,n_train);
    MatrixXd Y(n_train,1);
    for (int i = 0; i<n_train; i++){
        X(0,i) = f[i].first[0];
        X(1,i) = f[i].first[1];
        Y(i,0) = f[i].second;
    } 
    
    gaussian_process GP(X,Y);
    GP.set_SE_kernel();
    std::cout.precision(12);

    GP.optimize_kernel_parameters(1,1e-12,1);
    // compare actual values with prediction
    double variance;
    VectorXd mean;
    for (int i = n_train; i<n_train+100; i++){
        VectorXd x = f[i].first;
        GP.prediction(x,mean,variance);
        std::cout<<std::fixed
                 <<"x: "<<x.transpose()
                 <<",\tf(x): "<<sine(x[0], x[1])
                 <<",\tf*(x): "<<mean.transpose()
                 <<",\tvar(x): "<<variance<<std::endl;
    } 
    std::cout<<"n="<<n<<", n_train="<<n_train<<std::endl;
}
