#ifndef MEDIAN_H
#define MEDIAN_H
// 计算中位数
// 本文件是 libigl（一个简单的 C++ 几何处理库）的一部分。
// 
// 版权所有 (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// 本源代码形式受 Mozilla 公共许可证 第 2.0 版条款约束。
// 如果本文件未附带该许可证副本，您可以通过 http://mozilla.org/MPL/2.0/ 获取。
#include "eigen-3.4.0/Eigen/Dense"
#include <vector>
namespace igl
{
  template <typename DerivedM>
  void matrix_to_list(const Eigen::DenseBase<DerivedM>& M,
                      std::vector<typename DerivedM::Scalar> &V)
  {
      using namespace std;
      V.resize(M.size());
      // loop over cols then rows
      for(int j =0; j<M.cols();j++)
      {
          for(int i = 0; i < M.rows();i++)
          {
              V[i+j*M.rows()] = M(i,j);
          }
      }
  }

  // 计算一个 Eigen 向量的中位数
  //
  // 输入:
  //   V  #V 未排序值的列表
  // 输出:
  //   m  这些值的中位数
  // 返回:
  //   成功时返回 true，失败时返回 false
  template <typename DerivedV, typename mType>
  bool median(
    const Eigen::MatrixBase<DerivedV> & V, mType & m)
  {
    using namespace std;
    if(V.size() == 0)
    {
      return false;
    }
    vector<typename DerivedV::Scalar> vV;
    matrix_to_list(V,vV);
    // http://stackoverflow.com/a/1719155/148668
    size_t n = vV.size()/2;
    nth_element(vV.begin(),vV.begin()+n,vV.end());
    if(vV.size()%2==0)
    {
      nth_element(vV.begin(),vV.begin()+n-1,vV.end());
      m = 0.5*(vV[n]+vV[n-1]);
    }else
    {
      m = vV[n];
    }
    return true;
  }
}
#endif
