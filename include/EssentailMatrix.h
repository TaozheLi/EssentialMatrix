#include <opencv2/opencv.hpp>
#include <vector>
#ifndef OPTICALFLOWTEST_ESSENTAILMATRIX_H
#define OPTICALFLOWTEST_ESSENTAILMATRIX_H
void CalEssentialMatrix(std::vector<bool> &vStatus, const std::vector<int> &originalIndex, const cv::Mat & originalImage, const std::vector<cv::Point2f> &prevKeyPoints, const std::vector<cv::Point2f> &currentKeyPoints, const float & threshold, const int & maxIterations, const float &fx, const float &fy, const float &cx, const float & cy ) {
    cv::Mat K = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
    std::cout << "generate K matrix successfully" << std::endl;
    K.at<float>(0, 0) = fx;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 1) = fy;
    K.at<float>(1, 2) = cy;
    K.at<float>(2, 2) = 1.0f;
    std::vector<int> mask;
    std::vector<uchar> mask2;
    cv::Mat EssentialMatrix = cv::findEssentialMat(prevKeyPoints, currentKeyPoints, K, cv::RANSAC, 0.999, threshold,
                                                   maxIterations, mask2);
    for (int i = 0; i < mask2.size(); ++i) {
        if (!mask2[i]) {
            int index = originalIndex[i];
            vStatus[index] = false;
//            std::cout << "original index: " << index << std::endl;
        }
    }
}
#endif //OPTICALFLOWTEST_ESSENTAILMATRIX_H
