#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include<cstring>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

int main(int argc, char **argv)
{
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";
    

    ifstream fin(associate_file);
    if (!fin)
    {
        cerr << "file not found" << endl;
        return 1;
    }

    string rgb_time, rgb_file, depth_time, depth_file;
    vector<cv::Point2f> keypoints;
    cv::Mat color, depth, last_color;

    for (int i = 0; i < 10; i++)
    {
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        color = cv::imread(path_to_dataset + "/" + rgb_file);
        depth = cv::imread(path_to_dataset + "/" + depth_file);
        string outstr = "pic"+to_string(i);
        if (color.data == nullptr || depth.data == nullptr)
            continue;
        if (i == 0)
        {
            vector<cv::KeyPoint> tem_keypoints;
            cv::Ptr<cv::FastFeatureDetector> __detector = cv::FastFeatureDetector::create();
            __detector->detect(color, tem_keypoints);
            for (auto p : tem_keypoints)
            {
                cv::Point2f tem_p(p.pt.x, p.pt.y);
                keypoints.push_back(tem_p);
            }
            last_color = color;
            cv::Mat img_show = color.clone();
            for(auto p:keypoints)
                cv::circle(img_show,p,5,cv::Scalar(0,240,0),0.8);
            cv::imshow(outstr,img_show);
            cv::waitKey(0);
            continue;
        }

        vector<cv::Point2f> next_keypoints;
        vector<unsigned char> status;
        vector<float> error;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK(last_color, color, keypoints, next_keypoints,
                                 status, error);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << time_used.count() << endl;

        int j = 0;
        for (auto iter = keypoints.begin(); iter != keypoints.end(); j++)
        {
            if (status[j] == 0)
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[j];
            iter++;
        }
        cout << "tracked points" << keypoints.size() << endl;
        if (keypoints.size() == 0)
        {
            cout << "all keypoints are lost." << endl;
            break;
        }
        cv::Mat img_show = color.clone();
        for (auto p : keypoints)
        {
            cv::circle(img_show, p, 5,cv::Scalar(0,240,0),0.8);
        }
        
        cv::imshow(outstr,img_show);
        last_color = color;
        cv::waitKey(0);
    }
    return 0;
}