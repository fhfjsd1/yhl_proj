#include <iostream>
#include "ICP.h"
#include "io_pc.h"
#include "FRICP.h"

#include <random>

// ×××××××××××××××××× 控制开关 ××××××××××××××××××
//#define DEBUG_CHECKPOINT

// 设置单类别或多类别
//#define SINGLE
#define MULTI

// 设置Robust还是普通
//#define WELSCH_define
#define NONE_define

// 设置是否取消缩放
//#define SCALE

// 降采样模式（正常配准需要注释掉）
//#define SAMPLE

// ××××××××××××××××××××××××××××××××××××××××××

const int N = 3;
typedef Eigen::Matrix<Scalar, N, Eigen::Dynamic> MatrixNX;

typedef Eigen::Transform<Scalar, N, Eigen::Affine> AffineNd;

double FRICP_simulation(AffineNd T_simulation, ICP::Parameters& extern_pars, double& class_distance, double& distance);

double pointCloud_distance(MatrixNX X1, MatrixNX X2, MatrixNX X3, MatrixNX Y1, MatrixNX Y2, MatrixNX Y3, Eigen::Affine3d T, bool isColor = true); // isColor为true则为按类别搜索

double pointCloud_distance(MatrixNX X, MatrixNX Y, Eigen::Affine3d T);

void pointCloud_sample(std::string pointCloud_file_name, int sample_num);

int main(int argc, char const ** argv)
{
#ifdef SAMPLE

    std::cout << "请输入需要下采样点文件路径：" << std::endl;
    char file_path_char[100];
    std::cin.getline(file_path_char, 100);
    std::string file_path(file_path_char, file_path_char + strlen(file_path_char));
    //std::cin >> file_path;
    std::cout << file_path << std::endl;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    Vertices vertices, normal, src_vert;
    read_file(vertices, normal, src_vert, file_path);
    std::cout << "点个数：" << vertices.cols() << std::endl;
    std::cout << "请输入下采样到的点个数：" << std::endl;
    int sample_num = 0;
    std::cin >> sample_num;
    pointCloud_sample(file_path, sample_num);
    exit(0);
#endif // SAMPLE

    //AffineNd T1;
    //T1 = AffineNd::Identity();
    //Eigen::Matrix3d Rx1;
    //Rx1 << 1, 0, 0,
    //    0, 0, 0,
    //    1, 1, 0;
    //std::cout << Rx1 << std::endl;
    //std::cout << Rx1.norm() << std::endl; // ：矩阵的Frobenius范数
    //MatrixXX Rx = MatrixXX(Rx1);
    //T1.matrix() = Rx;
    //std::cout << T1.matrix().norm() << std::endl;

    double step = 10;
    double pi = 3.1415926;

    int count = 0;
    int success_count = 0;
    double total_class_distance = 0;
    double total_distance = 0;

    std::string path = "成功iter数统计.txt";
    std::ofstream file(path);
    std::string path2 = "结果和角度统计.txt";
    std::ofstream file2(path2);

    double all_init_time = 0;
    double all_registration_time = 0;
    double all_total_time = 0;

    for (double alpha = 0; alpha < 2 * pi - pi / step; alpha += 2 * pi / step) {
        for (double beta = 0; beta < 2 * pi - pi / step; beta += 2 * pi / step) {
            for (double theta = 0; theta < 2 * pi - pi / step; theta += 2 * pi / step) {
                Eigen::Matrix3d Rx;
                Rx << 1, 0, 0,
                    0, cos(theta), -sin(theta),
                    0, sin(theta), cos(theta);
                Eigen::Matrix3d Ry;
                Ry << cos(alpha), 0, sin(alpha),
                    0, 1, 0,
                    -sin(alpha), 0, cos(alpha);
                Eigen::Matrix3d Rz;
                Rz << cos(beta), -sin(beta), 0,
                    sin(beta), cos(beta), 0,
                    0, 0, 1;

                std::cout << "alpha: " << alpha << " beta: " << beta << " theta: " << theta << std::endl;

                Eigen::Matrix3d R_simulation = Rz * Ry * Rx;

                AffineNd T_simulation;
                T_simulation.linear() = R_simulation;
                T_simulation.translation() << 0, 0, 0;
                //std::cout << T.matrix() << std::endl;

                ICP::Parameters extern_pars;
                double class_distance = 0;
                double distance = 0;
                double matrix_diff = FRICP_simulation(T_simulation, extern_pars, class_distance, distance);
                std::cout << "init_time = " << extern_pars.init_time << ",registration_time = " << extern_pars.registration_time << ",iter_num = " << extern_pars.iter_num << std::endl;
                all_init_time += extern_pars.init_time;
                all_registration_time += extern_pars.registration_time;
                all_total_time += extern_pars.init_time + extern_pars.registration_time;

                count++;
#ifdef DEBUG
                std::cout << "matrix_diff" << matrix_diff << std::endl;
#endif // DEBUG
                if (matrix_diff < 0.01) {
                    success_count++;
                    //file << extern_pars.iter_num << " ";
                }
                file << extern_pars.iter_num << " ";
                file2 << extern_pars.iter_num << " " << theta << " " << beta << " " << alpha << std::endl;

                std::cout << "success_count/count: " << success_count << " / " << count << std::endl;
                total_class_distance += class_distance;
                total_distance += distance;
                std::cout << "totol_class_distance: " << total_class_distance << "totol_distance: " << total_distance << std::endl;
            }
        }
    }

    file.close();
    file2.close();
    std::cout << "init_time = " << all_init_time << ",registration_time = " << all_registration_time << ",all_total_time = " << all_total_time << std::endl;
    std::cout << std::endl;
    system("pause");
    system("pause");
    system("pause");
}


double FRICP_simulation(AffineNd T_simulation, ICP::Parameters& extern_pars, double& class_distance, double& distance)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;
    std::string file_source;
    std::string file_target;
    std::string file_init = "./data/bunny/";
    std::string res_trans_path;
    std::string out_path;
    bool use_init = false;
    MatrixXX res_trans;
    enum Method { ICP, AA_ICP, FICP, RICP, PPL, RPPL, SparseICP, SICPPPL } method = RICP;

    int dim = 3;

    out_path = "./output/";
    //method = Method(0);


#ifdef MULTI
    

    std::string file_target1 = "monkey_left.ply";
    std::string file_target2 = "monkey_middle.ply";
    std::string file_target3 = "monkey_right.ply";
    std::string file_source1 = "monkey_left.ply";
    std::string file_source2 = "monkey_middle.ply";
    std::string file_source3 = "monkey_right.ply";
    
#endif // MULTI

#ifdef SINGLE
    

    std::string file_target1 = "monkey.ply";
    std::string file_target2 = "monkey.ply";
    std::string file_target3 = "monkey.ply";
    std::string file_source1 = "monkey.ply";
    std::string file_source2 = "monkey.ply";
    std::string file_source3 = "monkey.ply";
#endif // SINGLE

    //单类别
    //--- Model that will be rigidly transformed
    Vertices vertices_source, normal_source, src_vert_colors;
    //read_file(vertices_source, normal_source, src_vert_colors, file_source);
    //std::cout << "source: " << vertices_source.rows() << "x" << vertices_source.cols() << std::endl;

    //--- Model that source will be aligned to
    Vertices vertices_target, normal_target, tar_vert_colors;
    //read_file(vertices_target, normal_target, tar_vert_colors, file_target);
    //std::cout << "target: " << vertices_target.rows() << "x" << vertices_target.cols() << std::endl;

    Vertices vertices_source1, normal_source1, src_vert_colors1;
    read_file(vertices_source1, normal_source1, src_vert_colors1, file_source1);
#ifdef DEBUG_CHECKPOINT
    std::cout << "source: " << vertices_source1.rows() << "x" << vertices_source1.cols() << std::endl;
#endif // DEBUG_CHECKPOINT
    Vertices vertices_source2, normal_source2, src_vert_colors2;
    read_file(vertices_source2, normal_source2, src_vert_colors2, file_source2);
#ifdef DEBUG_CHECKPOINT
    std::cout << "source: " << vertices_source2.rows() << "x" << vertices_source2.cols() << std::endl;
#endif // DEBUG_CHECKPOINT
    Vertices vertices_source3, normal_source3, src_vert_colors3;
    read_file(vertices_source3, normal_source3, src_vert_colors3, file_source3);
#ifdef DEBUG_CHECKPOINT
    std::cout << "source: " << vertices_source3.rows() << "x" << vertices_source3.cols() << std::endl;
#endif // DEBUG_CHECKPOINT

    Vertices vertices_target1, normal_target1, tar_vert_colors1;
    read_file(vertices_target1, normal_target1, tar_vert_colors1, file_target1);
    Vertices vertices_target2, normal_target2, tar_vert_colors2;
    read_file(vertices_target2, normal_target2, tar_vert_colors2, file_target2);
    Vertices vertices_target3, normal_target3, tar_vert_colors3;
    read_file(vertices_target3, normal_target3, tar_vert_colors3, file_target3);

#ifdef SINGLE
    // 单类别置0
    vertices_source2 = Vertices::Identity(3, 0);
    vertices_target2 = Vertices::Identity(3, 0);
    vertices_source3 = Vertices::Identity(3, 0);
    vertices_target3 = Vertices::Identity(3, 0);
#endif // SINGLE

    // ：矩阵拼接为一个大矩阵
    vertices_source = Vertices::Zero(3, vertices_source1.cols() + vertices_source2.cols() + vertices_source3.cols());
    for (int i = 0; i < vertices_source1.cols(); ++i) {
        vertices_source.col(i) = vertices_source1.col(i);
        //std::cout << vertices_source.col(i).transpose() << std::endl;
    }
    for (int i = 0; i < vertices_source2.cols(); ++i) {
        vertices_source.col(vertices_source1.cols() + i) = vertices_source2.col(i);
    }
    for (int i = 0; i < vertices_source3.cols(); ++i) {
        vertices_source.col(vertices_source1.cols() + vertices_source2.cols() + i) = vertices_source3.col(i);
    }
    vertices_target = Vertices::Zero(3, vertices_target1.cols() + vertices_target2.cols() + vertices_target3.cols());
    for (int i = 0; i < vertices_target1.cols(); ++i) {
        vertices_target.col(i) = vertices_target1.col(i);
    }
    for (int i = 0; i < vertices_target2.cols(); ++i) {
        vertices_target.col(vertices_target1.cols() + i) = vertices_target2.col(i);
    }
    for (int i = 0; i < vertices_target3.cols(); ++i) {
        vertices_target.col(vertices_target1.cols() + vertices_target2.cols() + i) = vertices_target3.col(i);
    }

    //vertices_source << vertices_source1, vertices_source2, vertices_source3;
    //vertices_target << vertices_target1, vertices_target2, vertices_target3;

    // scaling（点云尺寸必须归一化，否则对距离取权重的时候，大尺寸的距离过大，会导致权重过小甚至为0）
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
#ifdef DEBUG_CHECKPOINT
    std::cout << "scale = " << scale << std::endl;
#endif // DEBUG_CHECKPOINT

#ifdef SCALE
    scale = 1;  //取消缩放
#endif // SCALE
    vertices_source /= scale;   // 关于原点进行尺寸缩放（目的是让结束配准的条件固定）
    vertices_target /= scale;
    vertices_source1 /= scale;
    vertices_target1 /= scale;
    vertices_source2 /= scale;
    vertices_target2 /= scale;
    vertices_source3 /= scale;
    vertices_target3 /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
#ifdef DEBUG_CHECKPOINT
    std::cout << source_mean << std::endl;
#endif // DEBUG_CHECKPOINT

    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;
    vertices_source1.colwise() -= source_mean;
    vertices_source2.colwise() -= source_mean;
    vertices_source3.colwise() -= source_mean;
    vertices_target1.colwise() -= target_mean;
    vertices_target2.colwise() -= target_mean;
    vertices_target3.colwise() -= target_mean;

    //vertices_source = vertices_source1;
    //vertices_target = vertices_target1;
#ifdef DEBUG_CHECKPOINT
    std::cout << "初始平移：" << std::endl;
    std::cout << target_mean - source_mean << std::endl;
#endif // DEBUG_CHECKPOINT

    //std::string path = "test_result.txt";
    //std::ofstream test(path);
    //for (int i = 0; i < vertices_target.cols(); i++) {
    //    test << vertices_target.col(i).transpose() << std::endl;
    //}

    // 应用仿真变换
    vertices_target = T_simulation * vertices_target;
    vertices_target1 = T_simulation * vertices_target1;
    vertices_target2 = T_simulation * vertices_target2;
    vertices_target3 = T_simulation * vertices_target3;

    double time;
    // set ICP parameters
    ICP::Parameters pars;
    
    //pars.print_energy = true;   // ：测试
    pars.print_energy = false;
    
    // ：测试
#ifdef DEBUG_CHECKPOINT
    std::cout << "begin registration..." << std::endl;
#endif // DEBUG_CHECKPOINT

    FRICP<3> fricp;
    //double begin_reg = omp_get_wtime();
    double converge_rmse = 0;

    #ifdef DEBUG_CHECKPOINT
       std::ofstream test("target1.txt");
       for (int i = 0; i < vertices_source2.cols(); i++) {
           test << vertices_source2.col(i)[0] << " " << vertices_source2.col(i)[1] << " " << vertices_source2.col(i)[2] << std::endl;
       }
       test.close();
    #endif

        // 必要参数
        // 多类别测试
#ifdef WELSCH_define
    pars.f = ICP::WELSCH;
#endif // WELSCH

#ifdef NONE_define
    pars.f = ICP::NONE;
#endif // NONE
    
    pars.use_AA = false;

#ifdef MULTI
    fricp.point_to_point(vertices_source, vertices_target, vertices_source1, vertices_source2, vertices_source3, vertices_target1, vertices_target2, vertices_target3, source_mean, target_mean, pars);
#endif // MULTI

#ifdef SINGLE
    fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
#endif // SINGLE

    // 单类别测试
    //pars.f = ICP::WELSCH;
    //pars.use_AA = false;
    //fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
    //std::cout << "init_time = " << pars.init_time << "registration_time = " << pars.registration_time << "iter_num = " << pars.iter_num << std::endl;

    res_trans = pars.res_trans;
#ifdef DEBUG_CHECKPOINT
    std::cout << "Registration done!" << std::endl;
#endif // DEBUG_CHECKPOINT
    //double end_reg = omp_get_wtime();
    //time = end_reg - begin_reg;
    vertices_source = scale * vertices_source;  // ：恢复缩放
    vertices_target = scale * vertices_target;
    vertices_source1 = scale * vertices_source1;
    vertices_target1 = scale * vertices_target1;
    vertices_source2 = scale * vertices_source2;
    vertices_target2 = scale * vertices_target2;
    vertices_source3 = scale * vertices_source3;
    vertices_target3 = scale * vertices_target3;

    //out_path = out_path + "m" + std::to_string(method);
    //Eigen::Affine3d res_T;
    //res_T.linear() = res_trans.block(0, 0, 3, 3); // ：矩阵分块，从4x4矩阵中分别提取旋转和平移矩阵
    //res_T.translation() = res_trans.block(0, 3, 3, 1);
    ////res_trans_path = out_path + "trans.txt";
    ////std::ofstream out_trans(res_trans_path);
    ////res_trans.block(0, 3, 3, 1) *= scale * 0.001; // ：平移矩阵 * 0.001是 geomagic 需要的
    ////out_trans << res_trans << std::endl;
    ////out_trans.close();
    
    //out_path = out_path + "m" + std::to_string(method);
    Eigen::Affine3d res_T;
    //res_trans_path = out_path + "trans.txt";
    //std::ofstream out_trans(res_trans_path);
    res_trans.block(0, 3, 3, 1) *= scale;

    res_T.linear() = res_trans.block(0, 0, 3, 3);
    res_T.translation() = res_trans.block(0, 3, 3, 1);
    //out_trans << res_trans << std::endl;
    //out_trans.close();

#ifdef DEBUG_CHECKPOINT
    std::cout << "仿真旋转平移矩阵：" << std::endl << T_simulation.matrix() << std::endl;

    std::cout << "旋转平移矩阵：" << std::endl << res_T.matrix() << std::endl;
#endif // DEBUG_CHECKPOINT

    Eigen::Matrix3d R_diff = res_T.matrix().block(0, 0, 3, 3) - T_simulation.matrix().block(0, 0, 3, 3);
    //std::cout << R_diff << std::endl;

    // 配准完成后复原平移，变回原始点云关系（计算出的旋转矩阵是修正过的原始变换）
    vertices_source.colwise() += source_mean;
    vertices_target.colwise() += target_mean;
    vertices_source1.colwise() += source_mean;
    vertices_source2.colwise() += source_mean;
    vertices_source3.colwise() += source_mean;
    vertices_target1.colwise() += target_mean;
    vertices_target2.colwise() += target_mean;
    vertices_target3.colwise() += target_mean;

    class_distance = pointCloud_distance(vertices_source1, vertices_source2, vertices_source3, vertices_target1, vertices_target2, vertices_target3, res_T);
    distance = pointCloud_distance(vertices_source, vertices_target, res_T);
    std::cout << "类别距离：" << class_distance << "，距离：" << class_distance << std::endl;

    extern_pars = pars; // 传回时间参数

    return R_diff.norm();

    ///--- Write result to file
    //std::string file_source_reg = out_path + "reg_pc.ply";
    //write_file(file_source, vertices_source, normal_source, src_vert_colors, file_source_reg);

}


double pointCloud_distance(MatrixNX X1, MatrixNX X2, MatrixNX X3, MatrixNX Y1, MatrixNX Y2, MatrixNX Y3, Eigen::Affine3d T, bool isColor) // isColor为true则为按类别搜索
{
    const int N = 3;
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, N, Eigen::Dynamic> MatrixNX;
    typedef Eigen::Matrix<Scalar, N, N> MatrixNN;
    typedef Eigen::Matrix<Scalar, N + 1, N + 1> AffineMatrixN;
    typedef Eigen::Transform<Scalar, N, Eigen::Affine> AffineNd;
    typedef Eigen::Matrix<Scalar, N, 1> VectorN;
    typedef nanoflann::KDTreeAdaptor<MatrixNX, N, nanoflann::metric_L2_Simple> KDtree;
    typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

    KDtree kdtree1(Y1);
    if (X2.cols() == 0){
        Y2 = Y1;    
    }
    if (X3.cols() == 0) {
        Y3 = Y1;    // 若为空会报错，赋值后面也用不上
    }
    KDtree kdtree2(Y2);
    KDtree kdtree3(Y3);

    double all_distance = 0;

    // 初始化方式， norm 就是二范数
    //VectorN test1;
    //test1 << 1, 2, 3;
    //VectorN test2;
    //test2 << 3, 5, 10;
    //std::cout << test1 << std::endl;
    //std::cout << test2 << std::endl;
    //std::cout << test2 - test1 << std::endl;
    //std::cout << (test2 - test1).norm() << std::endl;

#pragma omp parallel for
    for (int i = 0; i < X1.cols(); ++i) {
        VectorN cur_p1 = T * X1.col(i);
        //VectorN cur_p1 = X1.col(i);
        //std::cout << "cur_p1: " << cur_p1.transpose() << std::endl;
        //std::cout << "Y1: " << Y1.col(i).transpose() << std::endl;
        VectorN cur_Q = Y1.col(kdtree1.closest(cur_p1.data()));// ：Q是Y中与经过T旋转后的当前Q（cur_q） 最近的一个点
        //std::cout << "kdtree11: " << kdtree11.closest(cur_p1.data()) << std::endl;
        //std::cout << "cur_Q: " << cur_Q.transpose() << std::endl;
        double distance = (cur_p1 - cur_Q).norm();   // ：计算每对点的距离，并储存
        all_distance += distance;
    }
    for (int i = 0; i < X2.cols(); ++i) {
        VectorN cur_p2 = T * X2.col(i);
        VectorN cur_Q = Y2.col(kdtree2.closest(cur_p2.data()));// ：Q是Y中与经过T旋转后的当前Q（cur_q） 最近的一个点
        double distance = (cur_p2 - cur_Q).norm();   // ：计算每对点的距离，并储存
        all_distance += distance;
    }
    for (int i = 0; i < X3.cols(); ++i) {
        VectorN cur_p3 = T * X3.col(i);
        VectorN cur_Q = Y3.col(kdtree3.closest(cur_p3.data()));// ：Q是Y中与经过T旋转后的当前Q（cur_q） 最近的一个点
        double distance = (cur_p3 - cur_Q).norm();   // ：计算每对点的距离，并储存
        all_distance += distance;
    }

    return all_distance / (X1.cols() + X2.cols() + X3.cols());
}


// 单类别距离
double pointCloud_distance(MatrixNX X, MatrixNX Y, Eigen::Affine3d T)
{
    const int N = 3;
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, N, Eigen::Dynamic> MatrixNX;
    typedef Eigen::Matrix<Scalar, N, N> MatrixNN;
    typedef Eigen::Matrix<Scalar, N + 1, N + 1> AffineMatrixN;
    typedef Eigen::Transform<Scalar, N, Eigen::Affine> AffineNd;
    typedef Eigen::Matrix<Scalar, N, 1> VectorN;
    typedef nanoflann::KDTreeAdaptor<MatrixNX, N, nanoflann::metric_L2_Simple> KDtree;
    typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

    KDtree kdtree(Y);

    double all_distance = 0;

#pragma omp parallel for
    for (int i = 0; i < X.cols(); ++i) {
        VectorN cur_p = T * X.col(i);
        VectorN cur_Q = Y.col(kdtree.closest(cur_p.data()));// ：Q是Y中与经过T旋转后的当前Q（cur_q） 最近的一个点
        double distance = (cur_p - cur_Q).norm();   // ：计算每对点的距离，并储存
        all_distance += distance;
    }
    
    return all_distance / X.cols();
}


void pointCloud_sample(std::string pointCloud_file_name, int sample_num) {
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    Vertices vertices, normal, src_vert;
    read_file(vertices, normal, src_vert, pointCloud_file_name);
    std::cout << "点个数：" << vertices.cols() << std::endl;
    //std::cout << normal.size() << std::endl;
    //std::cout << src_vert.size() << std::endl;

    if (vertices.cols() < sample_num) {
        std::cout << "采样大小超过总点个数" << std::endl;
        return;
    }

    int lower_bound = 0; // 下界
    int upper_bound = vertices.cols() - 1; // 上界
    int sample_size = sample_num; // 需要的随机数个数

    // 创建一个包含所有整数的向量
    std::vector<int> numbers;
    for (int i = lower_bound; i <= upper_bound; ++i) {
        numbers.push_back(i);
    }

    // 使用随机数引擎
    std::random_device rd; // 获得一个随机数种子
    std::mt19937 gen(rd()); // 使用种子初始化Mersenne Twister引擎

    // Fisher-Yates洗牌算法的变体
    for (int i = 0; i < sample_size; ++i) {
        std::uniform_int_distribution<> distrib(i, upper_bound - lower_bound);
        int j = distrib(gen);
        std::swap(numbers[i], numbers[j]);
    }

    // 取前 sample_size 个数字
    std::vector<int> sample(numbers.begin(), numbers.begin() + sample_size);

    // 对随机选择的数字进行排序
    std::sort(sample.begin(), sample.end());

    // 输出排序后的数字
    //for (int num : sample) {
    //    std::cout << num << " ";
    //}


    Vertices sample_vertices(3, sample_num);
    for (int i = 0; i < sample_num; ++i) {
        sample_vertices.col(i) = vertices.col(sample[i]);
    }

    write_file(pointCloud_file_name, sample_vertices, normal, src_vert, "sample.ply");
}
