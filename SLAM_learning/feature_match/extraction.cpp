#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "erro 01" << endl;
        return 1;
    }

    cv::Mat img_1 = cv::imread(argv[1]);
    cv::Mat img_2 = cv::imread(argv[2]);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    matcher->match(descriptors_1, descriptors_2, matches);

    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    std::vector<cv::DMatch> optimized_mactch_result;
    for ( int i = 0; i<descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max( 2*min_dist,30.0))
        {
            optimized_mactch_result.push_back(matches[i]);
        }
    }

    cv::Mat output_img_1;
    cv::drawKeypoints(img_1, keypoints_1, output_img_1, cv::Scalar::all(-1));
    cv::imshow("ORB", output_img_1);

    cv::Mat output_img_match;
    cv::Mat output_img_opti_match;
    cv::drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,output_img_match);
    cv::drawMatches(img_1,keypoints_1,img_2,keypoints_2,optimized_mactch_result,output_img_opti_match);
    cv::imshow("match",output_img_match);
    cv::imshow("good_match",output_img_opti_match);
    cv::waitKey(0);
    return 0;
}