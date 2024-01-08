#include <iostream>
#include <opencv2/optflow.hpp>
#include <opencv2/opencv.hpp>
#include <OpticalFlow.h>
#include <EssentailMatrix.h>
int main(int argc, char ** argv) {
    std::string imgPath1, imgPath2;
    cv::Mat prev, current, prevGray, currentGray;
    if(argc < 3){
        std::cout<<"please prepare parameters in this way ./OpticalFlowTest -imgPath1 -imgPath2";
        std::exit(-1);
    }
    imgPath1 = argv[1];
    imgPath2 = argv[2];
    prev = cv::imread(imgPath1, cv::ImreadModes::IMREAD_UNCHANGED);
    current = cv::imread(imgPath2, cv::ImreadModes::IMREAD_UNCHANGED);
    if(prev.empty() || current.empty()){
        std::cout<<"read image failed, please check input image path"<<std::endl;
        std::exit(-1);
    }
    std::cout<<"image size: "<<current.size<<std::endl;
    if(prev.channels() != 1){
        cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
    }
    if(current.channels()!=1){
        cv::cvtColor(current, currentGray, cv::COLOR_BGR2GRAY);
    }

    CalculateOpticalFlow(prevGray, currentGray, prev, current, true);
    std::vector<cv::Point2f> prevGoodFeatures;
    std::vector<cv::Point2f> currentGoodFeatures;
    cv::goodFeaturesToTrack(prevGray, prevGoodFeatures, 1000, 0.01, 10);
//    cv::goodFeaturesToTrack(currentGray, currentGoodFeatures, 100, 0.01, 10);
    std::vector<double> depth; depth.reserve(prevGoodFeatures.size());
    std::cout<<"feature nums before process: "<<prevGoodFeatures.size()<<std::endl;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray, currentGray, prevGoodFeatures, currentGoodFeatures, status, err, cv::Size(21, 21), 1);
    //assign same depth
    for(int i=0; i<prevGoodFeatures.size(); ++i){
//        if(i%3 == 0) depth.push_back(2.0);
//        if(i%3 == 1) depth.push_back(10.0);
//        if(i%3 == 2) depth.push_back(15.0);
        depth.push_back(15.0);
    }
    cv::Mat currentCopy = current.clone();
    for(int i=0; i<status.size(); ++i){
        if(status[i]){
            if(cv::norm(prevGoodFeatures[i] - currentGoodFeatures[i]) < 5) continue;
            cv::circle(currentCopy, cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), 3, cv::Scalar(0, 0, 255));
            cv::line(currentCopy, cv::Point(prevGoodFeatures[i].x, prevGoodFeatures[i].y),
                     cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), cv::Scalar(0, 255, 0));
        }
//        if(i == 500) break;
    }
    cv::imshow("original optical flow  ", currentCopy);
    std::vector<bool> Vstatus1(prevGoodFeatures.size(), true);
    std::vector<bool> Vstatus2(prevGoodFeatures.size(), true);
    double originalLength = 0;
    for(int i=0; i<status.size(); ++i){
        if(!status[i]) {
            Vstatus1[i] = false;
            Vstatus2[i] = false;
        }
        originalLength ++;
    }
    std::vector<int> groups;
    int classes = 20;
    groups = classifyBasedOnDepth(classes, depth);
//    RemovePointsThroughDepth(classes, groups, prevGoodFeatures, currentGoodFeatures, 3, 3, Vstatus);
    double a = 3;
    double b = 3;
    int parts = 5;
    bool useGlobalInformation = false;
    ClassifyBasedOnXY(classes, a, b, groups, prevGoodFeatures, currentGoodFeatures, Vstatus1, parts, useGlobalInformation);
    double length = 0;
    for(const auto & x: Vstatus1){
        if(x) length++;
    }

    cv::Mat originalCurrentCopyImage = current.clone();
    for(int i=0; i<Vstatus1.size(); ++i){

        if(Vstatus1[i]){
            if(cv::norm(prevGoodFeatures[i] - currentGoodFeatures[i]) < 5) continue;
            cv::circle(originalCurrentCopyImage, cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), 3, cv::Scalar(0, 0, 255));
            cv::line(originalCurrentCopyImage, cv::Point(prevGoodFeatures[i].x, prevGoodFeatures[i].y),
                     cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), cv::Scalar(0, 255, 0));
        }
//        if(i == 100) break;
    }
    cv::imshow("Length Method", originalCurrentCopyImage);
//    cv::waitKey(-1);
//    cv::destroyAllWindows();

    const float xmax = 10;
    const float xmin = 5;
    const float ymax = 10;
    const float ymin = 5;
    //situation 1
//    cv::Point2f p1(xmax, ymin);
//    cv::Point2f p2(xmin, ymax);
//    std::cout<<"\ncase 01: "<<ComputeAngleAtan2(p1, p2)<<std::endl;
//
//    //situation 2
//    cv::Point2f p3(xmin, ymin);
//    cv::Point2f p4(xmax, ymax);
//    std::cout<<"case 02: "<<ComputeAngleAtan2(p3, p4)<<std::endl;
//
//    //situation 3
//    cv::Point2f p5(xmin, ymax);
//    cv::Point2f p6(xmax, ymin);
//    std::cout<<"case 03: "<<ComputeAngleAtan2(p5, p6)<<std::endl;
//
//    //situation 4
//    cv::Point2f p7(xmax, ymax);
//    cv::Point2f p8(xmin, ymin);
//    std::cout<<"case 04: "<<ComputeAngleAtan2(p7, p8)<<std::endl;
    std::vector<cv::Point2f> selectedPrevFeatures, selectedCurrentFeatures;
    std::vector<int> originalIndex;
    for(int i=0; i<Vstatus2.size(); ++i){
        if(Vstatus2[i]){
            selectedPrevFeatures.push_back(prevGoodFeatures[i]);
            selectedCurrentFeatures.push_back(currentGoodFeatures[i]);
            originalIndex.push_back(i);
        }
    }
    float fx,fy,cx,cy, threshold;
    int maxIterations;
    fx = 718.856;
    fy = 718.856;
    cx = 607.1928;
    cy = 185.2157;
    threshold = 1.0f;
    maxIterations = 1000;
    std::cout<<"start to Calculate EssentialMatrix"<<std::endl;
    CalEssentialMatrix(Vstatus2, originalIndex, originalCurrentCopyImage, selectedPrevFeatures, selectedCurrentFeatures, threshold, maxIterations, fx, fy, cx, cy);
    cv::Mat imgAfterEssentialMatrixRemoving = current.clone();
    for(int i=0; i<Vstatus2.size(); ++i){
        if(Vstatus2[i]){
            if(cv::norm(prevGoodFeatures[i] - currentGoodFeatures[i]) < 5) continue;
            cv::circle(imgAfterEssentialMatrixRemoving, cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), 3, cv::Scalar(0, 0, 255));
            cv::line(imgAfterEssentialMatrixRemoving, cv::Point(prevGoodFeatures[i].x, prevGoodFeatures[i].y),
                     cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), cv::Scalar(0, 255, 0));
        }
//        if(i == 100) break;
    }
    cv::imshow("EssentailMatrix Method", imgAfterEssentialMatrixRemoving);
    cv::waitKey(-1);
    cv::destroyAllWindows();
    std::cout<<"calculate essential matrix successfully"<<std::endl;
    std::cout<<"original length: "<<originalLength<<" after length method final length: "<<length<<" ratio: "<<length / originalLength<<std::endl;
    double length2 = 0;
    for(const auto & x: Vstatus2){
        if(x) length2++;
    }
    std::cout<<"original length: "<<originalLength<<" after essential matrix final length: "<<length2<<" ratio: "<<length2 / originalLength<<std::endl;
}
