#pragma once

namespace CuCG
{
	enum class KernelCoefficient { beta_and_omega, alpha_and_omega, alpha, omega };
	enum class ExtraAction { NONE, compute_rs_new_and_beta, compute_alpha, compute_omega, compute_buffer, compute_rs_old };
}
