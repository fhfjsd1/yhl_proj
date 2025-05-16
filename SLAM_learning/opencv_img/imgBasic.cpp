#include <iostream>
#include <chrono>
using namespace std;

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>

int main(int argc,char** argv)
{
    cv::Mat image;
    image = cv::imread(argv[1]);
    if (image.data == nullptr)
    {
        cerr<<"文件不存在！"<<endl;
        return 0;
    }

    cout<<"width:"<<image.cols<<"height:"<<image.rows<<"channel:"
    <<image.channels()<<endl;
    cv::imshow("image",image);
    cv::waitKey(0);
    if (image.type()!=CV_8UC1 && image.type()!=CV_8UC3)
    {
        cout<<"input illegal!"<<endl;
        return 0;
    }

    cv::Mat img_another = image;
    img_another(cv::Rect(0,0,200,200)).setTo(0);
    cv::imshow("image",image);
    cv::waitKey(0);

    cv::Mat img_clone = image.clone();
    img_clone(cv::Rect(0,0,200,200)).setTo(255);
    cv::imshow("image",image);
    cv::imshow("image_clone",img_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}