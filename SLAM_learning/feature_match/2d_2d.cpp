#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include<cmath>

using namespace std;

void feature_macth(const cv::Mat &img_1, const cv::Mat &img_2,
                   std::vector<cv::KeyPoint> &keypoints_1,
                   std::vector<cv::KeyPoint> &keypoints_2,
                   std::vector<cv::DMatch> &matches)
{
    cv::Mat descriptors_1, descriptors_2;

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
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
        {
            optimized_mactch_result.push_back(matches[i]);
        }
    }
    matches = optimized_mactch_result;
    cout << "一共找到了" << matches.size() << "组匹配点" << endl
         << endl;
    cv::Mat output_img_with_matches;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, output_img_with_matches);
    cv::imshow("match", output_img_with_matches);
}

cv::Point2d pixel2cam(const cv::Point2d &pixel_points, const cv::Mat &K)
{
    return cv::Point2f((pixel_points.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                       (pixel_points.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(const std::vector<cv::KeyPoint> &keypoints_1,
                          const std::vector<cv::KeyPoint> &keypoints_2, const std::vector<cv::DMatch> &matches,
                          cv::Mat &R, cv::Mat &t)
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    cv::Mat F_mat = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    cout << "F_mat is: " << endl
         << F_mat << endl;

    cv::Mat E_mat = cv::findEssentialMat(points1, points2, K);
    cout << "E_mat is: " << endl
         << E_mat << endl;

    cv::Mat H_mat = cv::findHomography(points1, points2, cv::RANSAC);

    cv::recoverPose(E_mat, points1, points2, K, R, t);

    cout << "R：" << endl
         << R << endl
         << endl;
    cout << "t:" << endl
         << t << endl
         << endl;
}

void triangulation(const vector<cv::KeyPoint> &keypoints_1,
                   const vector<cv::KeyPoint> &keypoints_2,
                   const vector<cv::DMatch> &matches,
                   const cv::Mat &R, const cv::Mat &t,
                   vector<cv::Point3d> &points_with_depth)
{
    cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                  R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                  R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point2f> pts_cam_1, pts_cam_2;
    for (cv::DMatch m : matches)
    {
        pts_cam_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, K));
        pts_cam_2.push_back(pixel2cam(keypoints_2[m.trainIdx].pt, K));
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_cam_1, pts_cam_2, pts_4d);
    for (int i = 0; i < pts_4d.cols; i++)
    {
        cv::Point3d p;
        double s = pts_4d.at<float>(3, i);
        p.x = pts_4d.at<float>(0, i) / s;
        p.y = pts_4d.at<float>(1, i) / s;
        p.z = pts_4d.at<float>(2, i) / s;
        points_with_depth.push_back(p);

    }
}

void bundle_adjustment(const vector<cv::Point3f> points_3d,
                       const vector<cv::Point2f> points_2d, const cv::Mat &K, cv::Mat &R, cv::Mat &t)
{
    using Block = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;                                  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block *solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));         // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    Eigen::Vector3d t_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    t_mat << t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0);
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, t_mat));
    optimizer.addVertex(pose);

    int index = 1;
    for (const cv::Point3f p : points_3d)
    {
        g2o::VertexPointXYZ *point = new g2o::VertexPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    g2o::CameraParameters *camera =
        new g2o::CameraParameters(K.at<double>(0, 0),
                                  Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    index = 1;
    for (const cv::Point2f p : points_2d)
    {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "优化用时：" << time_used.count() << " 秒" << endl;

    cout << endl
         << "优化后：" << endl;
    cout << "T = " << endl
         << Eigen::Isometry3d(pose->estimate()).matrix() << endl;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "错误：命令行输入参数数量不对！" << endl;
        return 1;
    }
    std::cout << "当前OpenCV版本：" << CV_VERSION << std::endl;

    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;

    cv::Mat R, t;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    cout << "基于ORB进行特征匹配……" << endl;
    feature_macth(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "使用对极几何进行2D-2D位姿估计：" << endl
         << endl;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    vector<cv::Point3d> points_with_depth;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points_with_depth);
    cout << "三角化恢复深度信息……" << endl;
    cout << "验证重投影误差……" << endl;
    double x_bias_1, y_bias_1, x_bias_2, y_bias_2;
    for (int i = 0; i < matches.size(); i++)
    {
        cv::Point2d point_cam_from3d(points_with_depth[i].x / points_with_depth[i].z,
                                     points_with_depth[i].y / points_with_depth[i].z);
        cv::Point2f point_cam_frompixel = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        x_bias_1 += pow(points_with_depth[i].x / points_with_depth[i].z - point_cam_frompixel.x,2);
        y_bias_1 += pow(points_with_depth[i].y / points_with_depth[i].z - point_cam_frompixel.y,2);

        // cout << "从三角化得到的3d点转到归一化相机坐标系：" << point_cam_from3d << ", d=" << points_with_depth[i].z << endl;
        // cout << "从像素坐标经内参转到归一化相机坐标系：" << point_cam_frompixel << endl;
    }
    cout << "第一张图：x方向MSE：" << x_bias_1 / matches.size() << "     y方向MSE：" << y_bias_1 / matches.size() << endl
         << endl;

    std::vector<cv::Point2f> points_pix_uv;
    for (int i = 0; i < (int)matches.size(); i++)
    {
        cv::Point2f point_cam_from_pixel_2 = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        cv::Mat point_cam_from_3d_2 = R * (cv::Mat_<double>(3, 1) << points_with_depth[i].x,
                                           points_with_depth[i].y, points_with_depth[i].z) +
                                      t;
        cv::Point2f points_uv = keypoints_2[matches[i].trainIdx].pt;
        points_pix_uv.push_back(points_uv);
        point_cam_from_3d_2 /= point_cam_from_3d_2.at<double>(2, 0);
        x_bias_2 += pow(point_cam_from_pixel_2.x - point_cam_from_3d_2.at<double>(0, 0),2);
        y_bias_2 += pow(point_cam_from_pixel_2.y - point_cam_from_3d_2.at<double>(1, 0),2);

        // cout << "从三角化得到的3d点转到归一化相机坐标系：" << point_cam_from_3d_2.t() << ", d=" << point_cam_from_3d_2.at<double>(2,0) << endl;
        // cout << "从像素坐标经内参转到归一化相机坐标系：" << point_cam_from_pixel_2 << endl;
    }
    cout << "第二张图：x方向MSE：" << x_bias_2 / matches.size() << "     y方向MSE：" << y_bias_2 / matches.size() << endl
         << endl;

    std::vector<cv::Point3f> points_with_depth_under_world_3f;
    for (cv::Point3d p : points_with_depth)
    {
        cv::Point3f p_tem((float)p.x, (float)p.y, (float)p.z);
        points_with_depth_under_world_3f.push_back(p_tem);
    }

    cv::Mat r;
    cout << "利用PNP求解3D-2D位姿：" << endl;
    cv::solvePnP(points_with_depth_under_world_3f, points_pix_uv, K, cv::Mat(), r, t);
    cv::Rodrigues(r, R);
    cout << "R：" << endl
         << R << endl
         << endl;
    cout << "t:" << endl
         << t << endl
         << endl;
    cout << "Bundle Adjustment优化位姿结果：" << endl;
    bundle_adjustment(points_with_depth_under_world_3f, points_pix_uv, K, R, t);
    cout << "R：" << endl
         << R << endl
         << endl;
    cout << "t:" << endl
         << t << endl
         << endl;

    cv::waitKey(0);
    return 0;
}