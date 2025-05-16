#include<iostream>
#include<fstream>
using namespace std;

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<Eigen/Geometry> 

#include<boost/format.hpp>

#include<pcl-1.12/pcl/point_types.h>
#include<pcl-1.12/pcl/io/pcd_io.h>
#include<pcl-1.12/pcl/visualization/pcl_visualizer.h>

int main(int argc,char** argv)
{
    vector<cv::Mat> colorimgs,depthimgs;
    vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    ifstream fin("./pose.txt");
    if (!fin)
    {
        cerr<<"eee"<<endl;
        return 1;
    }

    for (int i=0;i<5;i++)
    {
        boost::format fmt("./%s/%d.%s");
        colorimgs.push_back(cv::imread((fmt%"color"%(i+1)%"png").str()));
        depthimgs.push_back(cv::imread((fmt%"depth"%(i+1)%"pgm").str(),-1));

        double data[7] = {0};
        for (auto& d:data)
            fin>>d;
        Eigen::Quaternion<double> q(data[6],data[3],data[4],data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0],data[1],data[2]));
        poses.push_back(T);
    }

    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthscale = 1000.0;

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr pointcloud(new PointCloud);
    for (int i=0;i<5;i++)
    {
        cout<<"转换图像中: "<<i+1<<endl;
        cv::Mat color = colorimgs[i];
        cv::Mat depth = depthimgs[i];
        Eigen::Isometry3d T = poses[i];
        for(int v=0;v<color.rows;v++)
        {
            for(int u=0;u<color.cols;u++)
            {
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if(d==0) continue;
                Eigen::Vector3d point;
                point[2] = double(d)/depthscale;
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy;
                Eigen::Vector3d point_under_world = T*point;

                PointT p;
                p.x = point_under_world[0];
                p.y = point_under_world[1];
                p.z = point_under_world[2];
                p.b = color.data[v*color.step + u*color.channels()];
                p.g = color.data[v*color.step + u*color.channels() + 1];
                p.r = color.data[v*color.step + u*color.channels() + 2];
                pointcloud->points.push_back(p);
            }
        }
    }

    pointcloud->is_dense = false;
    cout<<pointcloud->size()<<endl;
    pcl::io::savePCDFileBinary("map.pcd",*pointcloud);
    return 0;
}

