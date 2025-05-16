#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 打开默认摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    // 创建窗口
    namedWindow("Camera", WINDOW_AUTOSIZE);

    while (true) {
        Mat frame;
        // 捕获一帧图像
        cap >> frame;
        if (frame.empty()) {
            cerr << "无法捕获视频帧" << endl;
            break;
        }

        // 创建白色区域的掩膜: 定义 BGR 中接近白色的范围
        Mat mask;
        inRange(frame, Scalar(200, 200, 200), Scalar(255, 255, 255), mask);

        // 寻找掩膜中的轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 遍历所有轮廓，标注面积较大的白色区域
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > 100) {  // 忽略较小的区域
                Rect boundingBox = boundingRect(contours[i]);
                rectangle(frame, boundingBox, Scalar(0, 0, 255), 2);
                putText(frame, "White", boundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            }
        }

        // 在窗口中显示图像
        imshow("Camera", frame);

        // 按任意键退出
        if (waitKey(30) >= 0)
            break;
    }

    // 释放摄像头资源并销毁所有窗口
    cap.release();
    destroyAllWindows();
    return 0;
}