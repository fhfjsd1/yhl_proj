/**
 * @file Types.h
 * @brief 通用类型定义及基于 Eigen 的小型线性代数工具
 *
 * 本头文件定义：
 *   - 可配置标量类型（float 或 double）。
 *   - Eigen 类型的对齐策略。
 *   - 固定和动态尺寸矩阵/向量别名。
 *   - 通用三元向量类型与 Eigen::Vector3 之间的转换。
 *   - 块矩阵复合类（Matrix3333 和 Matrix2222）及其算术和收缩操作。
 *   - 2×2 和 3×3 块矩阵的 Kronecker（直积）函数。
 */

/**
 * @typedef Scalar
 * @brief 库中使用的基础算术类型。
 * @details 当定义 USE_FLOAT_SCALAR 时为 float，否则为 double。
 */

/**
 * @def EIGEN_ALIGNMENT
 * @brief Eigen 对齐宏选项。
 * @details 若定义 EIGEN_DONT_ALIGN，展开为 Eigen::DontAlign；否则为 Eigen::AutoAlign。
 */

/**
 * @tparam Rows   行数（编译时或动态）。
 * @tparam Cols   列数（编译时或动态）。
 * @tparam Options 存储选项（如列优先和对齐选项）。
 * @brief Eigen::Matrix<Scalar,Rows,Cols,Options> 的别名模板。
 */

/**
 * @typedef Vector2
 * @brief 2 维 Scalar 向量。
 */

/**
 * @typedef Matrix22
 * @brief 2×2 Scalar 矩阵。
 */

/**
 * @typedef Matrix23
 * @brief 2×3 Scalar 矩阵。
 */

/**
 * @typedef Vector3
 * @brief 3 维 Scalar 向量。
 */

/**
 * @typedef Matrix32
 * @brief 3×2 Scalar 矩阵。
 */

/**
 * @typedef Matrix33
 * @brief 3×3 Scalar 矩阵。
 */

/**
 * @typedef Matrix34
 * @brief 3×4 Scalar 矩阵。
 */

/**
 * @typedef Vector4
 * @brief 4 维 Scalar 向量。
 */

/**
 * @typedef Matrix44
 * @brief 4×4 Scalar 矩阵。
 */

/**
 * @typedef Matrix4X
 * @brief 4×N 动态列数 Scalar 矩阵。
 */

/**
 * @typedef Matrix3X
 * @brief 3×N 动态列数 Scalar 矩阵。
 */

/**
 * @typedef MatrixX3
 * @brief N×3 动态行数 Scalar 矩阵。
 */

/**
 * @typedef Matrix2X
 * @brief 2×N 动态列数 Scalar 矩阵。
 */

/**
 * @typedef MatrixX2
 * @brief N×2 动态行数 Scalar 矩阵。
 */

/**
 * @typedef VectorX
 * @brief N×1 动态长度列向量。
 */

/**
 * @typedef MatrixXX
 * @brief 完全动态尺寸 N×M 矩阵。
 */

/**
 * @typedef EigenMatrix12
 * @brief 无对齐标志的固定 12×12 Scalar 矩阵。
 */

/**
 * @typedef EigenAngleAxis
 * @brief 基于 Scalar 的 Eigen::AngleAxis。
 */

/**
 * @typedef EigenQuaternion
 * @brief 无对齐填充的基于 Scalar 的 Eigen::Quaternion。
 */

/**
 * @brief 将通用 3 元向量类型转换为 Eigen::Vector3。
 * @tparam Vec_T  支持 operator[] (0..2) 的任意类型。
 * @param vec     输入向量，至少包含 3 个元素。
 * @return        由 vec[0], vec[1], vec[2] 构成的 Eigen::Vector3。
 */

/**
 * @brief 将 Eigen::Vector3 转换回通用 3 元向量类型。
 * @tparam Vec_T  支持默认构造且支持 operator[] (0..2) 的任意类型。
 * @param vec     Eigen::Vector3 实例。
 * @return        令 v[0]=vec(0), v[1]=vec(1), v[2]=vec(2) 的 Vec_T 对象。
 */

/**
 * @class Matrix3333
 * @brief 每个块均为 3×3 Scalar 矩阵的 3×3 块矩阵。
 *
 * 支持块访问、加法、减法、标量及矩阵乘法、
 * 转置和收缩操作。
 */

/**
 * @brief 默认构造函数。块未初始化。
 */

/**
 * @brief 拷贝构造函数。
 * @param other  源 Matrix3333 对象。
 */

/**
 * @brief 析构函数。
 */

/**
 * @brief 将所有 3×3 块置零。
 */

/**
 * @brief 将主对角线块置单位矩阵，非对角线块置零。
 */

/**
 * @brief 按 (row,col) 访问 3×3 块，索引范围 [0..2]。
 * @param row   块行索引。
 * @param col   块列索引。
 * @return      对应的 Matrix33 块引用。
 */

/**
 * @brief 块加法。
 * @param plus  右操作数。
 * @return      相加后的 Matrix3333。
 */

/**
 * @brief 块减法。
 * @param minus 右操作数。
 * @return      相减后的 Matrix3333。
 */

/**
 * @brief 每个块右乘同一个 3×3 矩阵。
 * @param multi 右乘的 Matrix33。
 * @return      结果 Matrix3333。
 */

/**
 * @brief 左乘 Matrix3333 的友元函数版本。
 * @param multi1 左侧 Matrix33。
 * @param multi2 右侧 Matrix3333。
 * @return       结果 Matrix3333。
 */

/**
 * @brief 块乘以标量。
 * @param multi  标量因子。
 * @return       缩放后的 Matrix3333。
 */

/**
 * @brief 块乘以标量的友元函数版本。
 * @param multi1 标量因子。
 * @param multi2 要缩放的 Matrix3333。
 * @return       缩放后的 Matrix3333。
 */

/**
 * @brief 转置块矩阵：交换块位置并转置每个 3×3 块。
 * @return      转置后的 Matrix3333。
 */

/**
 * @brief 将本 3×3 块矩阵与 3×3 矩阵收缩：Σ_i,j mat[i][j] * multi(j,i)。
 * @param multi  要收缩的 Matrix33。
 * @return       单个 3×3 Matrix33 结果。
 */

/**
 * @brief 将两个 3×3 块矩阵收缩为另一个 3×3 块矩阵。
 * @param multi  右操作数 Matrix3333。
 * @return       收缩后的 Matrix3333。
 */

/**
 * @class Matrix2222
 * @brief 每个块均为 2×2 Scalar 矩阵的 2×2 块矩阵。
 *
 * 支持与 Matrix3333 类似的操作。
 */

/**
 * @brief 默认构造函数。块未初始化。
 */

/**
 * @brief 拷贝构造函数。
 * @param other 源 Matrix2222 对象。
 */

/**
 * @brief 析构函数。
 */

/**
 * @brief 将所有 2×2 块置零。
 */

/**
 * @brief 将主对角线块置单位矩阵，非对角线块置零。
 */

/**
 * @brief 按 (row,col) 访问 2×2 块，索引范围 [0..1]。
 * @param row   块行索引。
 * @param col   块列索引。
 * @return      对应的 Matrix22 块引用。
 */

/**
 * @brief 块加法。
 * @param plus  右操作数。
 * @return      相加后的 Matrix2222。
 */

/**
 * @brief 块减法。
 * @param minus 右操作数。
 * @return      相减后的 Matrix2222。
 */

/**
 * @brief 每个块右乘同一个 2×2 矩阵。
 * @param multi 右乘的 Matrix22。
 * @return      结果 Matrix2222。
 */

/**
 * @brief 左乘 Matrix2222 的友元函数版本。
 * @param multi1 左侧 Matrix22。
 * @param multi2 右侧 Matrix2222。
 * @return       结果 Matrix2222。
 */

/**
 * @brief 块乘以标量。
 * @param multi  标量因子。
 * @return       缩放后的 Matrix2222。
 */

/**
 * @brief 块乘以标量的友元函数版本。
 * @param multi1 标量因子。
 * @param multi2 要缩放的 Matrix2222。
 * @return       缩放后的 Matrix2222。
 */

/**
 * @brief 转置块矩阵：交换块位置并转置每个 2×2 块。
 * @return      转置后的 Matrix2222。
 */

/**
 * @brief 将本 2×2 块矩阵与 2×2 矩阵收缩。
 * @param multi  要收缩的 Matrix22。
 * @return       单个 2×2 Matrix22 结果。
 */

/**
 * @brief 将两个 2×2 块矩阵收缩为另一个 2×2 块矩阵。
 * @param multi  右操作数 Matrix2222。
 * @return       收缩后的 Matrix2222。
 */

/**
 * @brief 计算两个 3×3 矩阵的 Kronecker 直积，结果存入块结构。
 * @param dst   输出 Matrix3333，接收 src1 ⊗ src2。
 * @param src1  左因子 3×3 矩阵。
 * @param src2  右因子 3×3 矩阵。
 */

/**
 * @brief 计算两个 2×2 矩阵的 Kronecker 直积，结果存入块结构。
 * @param dst   输出 Matrix2222，接收 src1 ⊗ src2。
 * @param src1  左因子 2×2 矩阵。
 * @param src2  右因子 2×2 矩阵。
 */

#ifndef TYPES_H

#define TYPES_H
#include "eigen-3.4.0/Eigen/Dense"

#ifdef USE_FLOAT_SCALAR
typedef float Scalar;
#else
typedef double Scalar;
#endif

#ifdef EIGEN_DONT_ALIGN
#define EIGEN_ALIGNMENT Eigen::DontAlign
#else
#define EIGEN_ALIGNMENT Eigen::AutoAlign
#endif

template <int Rows, int Cols, int Options = (Eigen::ColMajor | EIGEN_ALIGNMENT)>
using MatrixT = Eigen::Matrix<Scalar, Rows, Cols, Options>; ///< A typedef of the dense matrix of Eigen.

typedef MatrixT<2, 1> Vector2;                              ///< A 2d column vector.
typedef MatrixT<2, 2> Matrix22;                             ///< A 2 by 2 matrix.
typedef MatrixT<2, 3> Matrix23;                             ///< A 2 by 3 matrix.
typedef MatrixT<3, 1> Vector3;                              ///< A 3d column vector.
typedef MatrixT<3, 2> Matrix32;                             ///< A 3 by 2 matrix.
typedef MatrixT<3, 3> Matrix33;                             ///< A 3 by 3 matrix.
typedef MatrixT<3, 4> Matrix34;                             ///< A 3 by 4 matrix.
typedef MatrixT<4, 1> Vector4;                              ///< A 4d column vector.
typedef MatrixT<4, 4> Matrix44;                             ///< A 4 by 4 matrix.
typedef MatrixT<4, Eigen::Dynamic> Matrix4X;                ///< A 4 by n matrix.
typedef MatrixT<3, Eigen::Dynamic> Matrix3X;                ///< A 3 by n matrix.
typedef MatrixT<Eigen::Dynamic, 3> MatrixX3;                ///< A n by 3 matrix.
typedef MatrixT<2, Eigen::Dynamic> Matrix2X;                ///< A 2 by n matrix.
typedef MatrixT<Eigen::Dynamic, 2> MatrixX2;                ///< A n by 2 matrix.
typedef MatrixT<Eigen::Dynamic, 1> VectorX;                 ///< A nd column vector.
typedef MatrixT<Eigen::Dynamic, Eigen::Dynamic> MatrixXX;   ///< A n by m matrix.
typedef Eigen::Matrix<Scalar, 12, 12, 0, 12, 12> EigenMatrix12;

// 基于 Eigen 的四元数
typedef Eigen::AngleAxis<Scalar> EigenAngleAxis;
typedef Eigen::Quaternion<Scalar, Eigen::DontAlign> EigenQuaternion;

// 通用三维向量类型与 Eigen::Vector3 之间的转换
template <typename Vec_T>
inline Vector3 to_eigen_vec3(const Vec_T &vec)
{
    return Vector3(vec[0], vec[1], vec[2]);
}

template <typename Vec_T>
inline Vec_T from_eigen_vec3(const Vector3 &vec)
{
    Vec_T v;
    v[0] = vec(0);
    v[1] = vec(1);
    v[2] = vec(2);

    return v;
}

class Matrix3333 // 3x3 matrix: each element is a 3x3 matrix
{
public:
    Matrix3333();
    Matrix3333(const Matrix3333 &other);
    ~Matrix3333() {}

    void SetZero();     // [0 0 0; 0 0 0; 0 0 0]; 0 = 3x3 zeros
    void SetIdentity(); //[I 0 0; 0 I 0; 0 0 I]; 0 = 3x3 zeros, I = 3x3 identity

    // operators
    Matrix33 &operator()(int row, int col);
    Matrix3333 operator+(const Matrix3333 &plus);
    Matrix3333 operator-(const Matrix3333 &minus);
    Matrix3333 operator*(const Matrix33 &multi);
    friend Matrix3333 operator*(const Matrix33 &multi1, Matrix3333 &multi2);
    Matrix3333 operator*(Scalar multi);
    friend Matrix3333 operator*(Scalar multi1, Matrix3333 &multi2);
    Matrix3333 transpose();
    Matrix33 Contract(const Matrix33 &multi); // this operator is commutative
    Matrix3333 Contract(Matrix3333 &multi);

    // protected:

    Matrix33 mat[3][3];
};

class Matrix2222 // 2x2 matrix: each element is a 2x2 matrix
{
public:
    Matrix2222();
    Matrix2222(const Matrix2222 &other);
    ~Matrix2222() {}

    void SetZero();     // [0 0; 0 0]; 0 = 2x2 zeros
    void SetIdentity(); //[I 0; 0 I;]; 0 = 2x2 zeros, I = 2x2 identity

    // operators and basic functions
    Matrix22 &operator()(int row, int col);
    Matrix2222 operator+(const Matrix2222 &plus);
    Matrix2222 operator-(const Matrix2222 &minus);
    Matrix2222 operator*(const Matrix22 &multi);
    friend Matrix2222 operator*(const Matrix22 &multi1, Matrix2222 &multi2);
    Matrix2222 operator*(Scalar multi);
    friend Matrix2222 operator*(Scalar multi1, Matrix2222 &multi2);
    Matrix2222 transpose();
    Matrix22 Contract(const Matrix22 &multi); // this operator is commutative
    Matrix2222 Contract(Matrix2222 &multi);

protected:
    Matrix22 mat[2][2];
};

// dst = src1 \kron src2
void directProduct(Matrix3333 &dst, const Matrix33 &src1, const Matrix33 &src2);
void directProduct(Matrix2222 &dst, const Matrix22 &src1, const Matrix22 &src2);
#endif // TYPES_H

