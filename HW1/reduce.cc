#include <stdlib.h>
// #include <cstdio>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>


int main(int argc, char** argv) 
{

  if(argc < 2) std::cout<<"Usage : ./reduce num_items"<<std::endl;
  int N = atoi(argv[1]);

  //0. Initialize

  int *arrayIn = new int[N];
  {
    std::chrono::duration<double> diff;
    auto start = std::chrono::steady_clock::now();
    for(int i=0;i<N;i++) {
      arrayIn[i] = i;
    }
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"init took "<<diff.count()<<" sec"<<std::endl;
  }

  {
    long long serial_sum=0;
    //1. Serial
    std::chrono::duration<double> diff;
    auto start = std::chrono::steady_clock::now();
    for(int i=0;i<N;i++) {
      serial_sum+=arrayIn[i];
    }
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"serial reduce took "<<diff.count()<<" sec"<<std::endl;
    std::cout<<"answer: "<<serial_sum<<std::endl;
  }

  {
    long long sum=0;
    //2. parallel reduce
    std::chrono::duration<double> diff;
    auto start = std::chrono::steady_clock::now();
    /* TODO: put your own parallelized reduce here */



    /* TODO: put your own parallelized reduce here */
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"parallel reduce took "<<diff.count()<<" sec"<<std::endl;
    std::cout<<"sum: "<<sum<<std::endl;

  }
  return 0;
}
