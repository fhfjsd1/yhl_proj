/**
 * @class AndersonAcceleration
 * @brief 实现 Anderson 加速算法，加速固定点迭代的收敛。
 *
 * 本类维护最近 m 次函数评估和迭代的历史记录，
 * 然后构建并求解一个小规模线性系统以计算外推更新。
 * 它会动态重缩放差分以提高数值稳定性。
 *
 * 用法：
 *   1. 调用 init(m, d, u0) 设置记忆深度、变量维度和初始猜测。
 *   2. 在每次迭代中，使用当前函数值指针调用 compute(g)。
 *   3. 可选地调用 replace(u) 覆盖当前迭代值，或 reset(u) 重新开始。
 */


//void init(int m, int d, const Scalar* u0);
/**
 * @brief 分配工作空间并设置初始迭代值。
 *
 * @param m   要存储的过去迭代次数（记忆深度，必须 > 0）。
 * @param d   变量向量的维度。
 * @param u0  指向长度为 d 的数组的指针，包含初始迭代值。
 *
 * 调用 init() 后，iter_ 被设置为 0，历史缓冲区被置零。
 */

//void replace(const Scalar* u);
/**
 * @brief 在不影响历史记录的情况下替换当前存储的迭代值。
 *
 * @param u  指向长度为 dim_ 的数组的指针，表示新的当前迭代值。
 *
 * 对注入已知解或与外部状态同步很有用。
 */

// const VectorX& compute(const Scalar* g);
/**
 * @brief 对当前迭代执行一次 Anderson 加速更新。
 *
 * @param g  指向长度为 dim_ 的数组的指针，包含当前迭代处的函数评估值。
 * @return   更新后的迭代值的 Eigen 向量引用。
 *
 * 在第一次调用时（iter_ == 0），此函数仅返回 g。
 * 在随后的调用中，会更新差分历史、求解最小二乘问题以获得组合系数，
 * 并返回外推更新。
 */

//void reset(const Scalar* u);
/**
 * @brief 重置算法状态以使用新的初始猜测。
 *
 * @param u  指向长度为 dim_ 的数组的指针，表示新的初始迭代值。
 *
 * 清除迭代计数器和历史；等同于全新 init()，但重用当前维度。
 */

#ifndef ANDERSONACCELERATION_H_
#define ANDERSONACCELERATION_H_

#include "Types.h"
#include <cassert>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <fstream>

class AndersonAcceleration
{
public:
	AndersonAcceleration()
		:m_(-1), dim_(-1), iter_(-1), col_idx_(-1) {}

	void replace(const Scalar *u)
	{
		current_u_ = Eigen::Map<const VectorX>(u, dim_);
	}

	const VectorX& compute(const Scalar* g)
	{
		assert(iter_ >= 0);

		Eigen::Map<const VectorX> G(g, dim_);
		current_F_ = G - current_u_;

		if (iter_ == 0)
		{
			prev_dF_.col(0) = -current_F_;
			prev_dG_.col(0) = -G;
			current_u_ = G;
		}
		else
		{
			prev_dF_.col(col_idx_) += current_F_;
			prev_dG_.col(col_idx_) += G;

			Scalar eps = 1e-14;
			Scalar scale = std::max(eps, prev_dF_.col(col_idx_).norm());
			dF_scale_(col_idx_) = scale;
			prev_dF_.col(col_idx_) /= scale;

			int m_k = std::min(m_, iter_);


			if (m_k == 1)
			{
				theta_(0) = 0;
				Scalar dF_sqrnorm = prev_dF_.col(col_idx_).squaredNorm();
				M_(0, 0) = dF_sqrnorm;
				Scalar dF_norm = std::sqrt(dF_sqrnorm);

                if (dF_norm > eps) {
					theta_(0) = (prev_dF_.col(col_idx_) / dF_norm).dot(current_F_ / dF_norm);
				}
			}
			else
			{
				// Update the normal equation matrix, for the column and row corresponding to the new dF column
				VectorX new_inner_prod = (prev_dF_.col(col_idx_).transpose() * prev_dF_.block(0, 0, dim_, m_k)).transpose();
				M_.block(col_idx_, 0, 1, m_k) = new_inner_prod.transpose();
				M_.block(0, col_idx_, m_k, 1) = new_inner_prod;

				// Solve normal equation
				cod_.compute(M_.block(0, 0, m_k, m_k));
				theta_.head(m_k) = cod_.solve(prev_dF_.block(0, 0, dim_, m_k).transpose() * current_F_);
			}

			// Use rescaled theata to compute new u
			current_u_ = G - prev_dG_.block(0, 0, dim_, m_k) * ((theta_.head(m_k).array() / dF_scale_.head(m_k).array()).matrix());
			col_idx_ = (col_idx_ + 1) % m_;
			prev_dF_.col(col_idx_) = -current_F_;
			prev_dG_.col(col_idx_) = -G;
		}

		iter_++;
		return current_u_;
	}
    void reset(const Scalar *u)
    {
        iter_ = 0;
        col_idx_ = 0;
        current_u_ = Eigen::Map<const VectorX>(u, dim_);
    }

	// m: number of previous iterations used
	// d: dimension of variables
	// u0: initial variable values
	void init(int m, int d, const Scalar* u0)
	{
		assert(m > 0);
		m_ = m;
		dim_ = d;
		current_u_.resize(d);
		current_F_.resize(d);
		prev_dG_.resize(d, m);
		prev_dF_.resize(d, m);
		M_.resize(m, m);
		theta_.resize(m);
		dF_scale_.resize(m);
		current_u_ = Eigen::Map<const VectorX>(u0, d);
		iter_ = 0;
		col_idx_ = 0;
	}

private:
	VectorX current_u_;
	VectorX current_F_;
	MatrixXX prev_dG_;
	MatrixXX prev_dF_;
	MatrixXX M_;		// Normal equations matrix for the computing theta
	VectorX	theta_;	// theta value computed from normal equations
	VectorX dF_scale_;		// The scaling factor for each column of prev_dF
	Eigen::CompleteOrthogonalDecomposition<MatrixXX> cod_;

	int m_;		// Number of previous iterates used for Andreson Acceleration
	int dim_;	// Dimension of variables
	int iter_;	// Iteration count since initialization
	int col_idx_;	// Index for history matrix column to store the next value
	int m_k_;

	Eigen::Matrix4d current_T_;
	Eigen::Matrix4d current_F_T_;

	MatrixXX T_prev_dF_;
	MatrixXX T_prev_dG_;
};


#endif /* ANDERSONACCELERATION_H_ */
