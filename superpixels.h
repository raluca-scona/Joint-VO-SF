#ifndef SUPERPIXELS
#define SUPERPIXELS

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <functional>
#include <queue>
#include <vector>

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;


class Superpixels {
public:

    Superpixels(Eigen::MatrixXf im_r, Eigen::MatrixXf im_g, Eigen::MatrixXf im_b,  Eigen::MatrixXf xx, Eigen::MatrixXf yy, Eigen::MatrixXf depth, unsigned int superpixelsNo, float m);

    Eigen::MatrixXi getLabelsImage();

    void rgbToLab(Eigen::MatrixXf im_r, Eigen::MatrixXf im_g, Eigen::MatrixXf im_b);
    void findSeeds(const int width, const int height, int& numk, vector<int>& kx, vector<int>& ky, vector<float>& kz_metric, vector<float>& kx_metric, vector<float>& ky_metric);
    void runSNIC(const int width, const int height, int* outnumk, const int innumk, const double compactness);


private:
    int superpixelsNo;
    float m;
    float noOfPixels;
    int s;
    int rows;
    int cols;
    cv::Mat colorImage;
    cv::Mat depthImage;
    Eigen::MatrixXi labelsImage;
    Eigen::MatrixXi reachedByQueue;

    Eigen::MatrixXd lmat;
    Eigen::MatrixXd amat;
    Eigen::MatrixXd bmat;


    Eigen::MatrixXf xmat;
    Eigen::MatrixXf ymat;
    Eigen::MatrixXf zmat;

    std::ofstream printSuperpixels;


    std::chrono::duration<double, std::milli> segmentationDuration;
    std::chrono::high_resolution_clock::time_point segmentationT0;
    std::chrono::high_resolution_clock::time_point segmentationT1;

    //std::vector<std::pair<queueElement, int>> clusterCenters;
   // std::priority_queue<queueElement, std::vector<queueElement>, closestElementByDistance> queue;

};


#endif // SUPERPIXELS
