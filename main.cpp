#define _CRT_SECURE_NO_WARNINGS // Boost apparently uses sprintf() in its code which is deprecated by MSVC

#include <boost/numeric/odeint.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <iostream>
#include <vector>
#include <complex>
#include <tuple>
#include "eq.h"
#include <algorithm>
#include <fstream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <chrono>

namespace py = pybind11;

typedef std::complex<val_t> cmpx;
typedef std::vector<cmpx> trace_t;
typedef std::vector<val_t> tlist_t;
typedef std::vector<cmpx> peaks_t;
typedef std::tuple<val_t, val_t, val_t, val_t, val_t, val_t, val_t> params_t;
typedef std::tuple<peaks_t, peaks_t> result_t;
typedef std::vector<val_t> zlist_t;
typedef std::vector<val_t> state2_t;
typedef std::vector<std::vector<cmpx>> op_t;
typedef std::vector<val_t> map_t;

struct push_back_state_and_time
{
	trace_t& m_sigmam1;
	trace_t& m_sigmam2;
	tlist_t& m_times;

	push_back_state_and_time(trace_t& sigmam1, trace_t& sigmam2, tlist_t& times)
		: m_sigmam1(sigmam1), m_sigmam2(sigmam2), m_times(times) { }

	void operator()(const state_t& x, val_t t)
	{
		m_sigmam1.push_back(cmpx(x[0], x[1]));
		m_sigmam2.push_back(cmpx(x[3], x[4]));
		m_times.push_back(t);
	}
};

struct push_back_state_and_time2
{
	peaks_t& q1;
	peaks_t& q2;
	val_t delta;
	val_t t_init_to_skip;

	push_back_state_and_time2(peaks_t& qubit1, peaks_t& qubit2, val_t df, val_t t_init_to_skip)
		: q1(qubit1), q2(qubit2), delta(df), t_init_to_skip(t_init_to_skip) { }

	void operator()(const state_t& x, val_t t)
	{
		if (t >= t_init_to_skip)
		{
			int N = q1.size();
			for (int i = 0; i < N; i++)
			{
				float order = static_cast<float>(2 * i - N + 1);
				q1[i] += cmpx(0.5f * x[0], (-0.5f) * x[1]) * std::exp(cmpx(0.f, delta * t * static_cast<float>(order)));
				q2[i] += cmpx(0.5f * x[3], (-0.5f) * x[4]) * std::exp(cmpx(0.f, delta * t * static_cast<float>(order)));
			}
		}
	}
};

struct push_back_state_and_time3
{
	std::vector<state2_t>& m_states;
	tlist_t& m_times;

	push_back_state_and_time3(std::vector<state2_t>& states, tlist_t& times)
		: m_states(states), m_times(times) { }

	void operator()(const state_t& x, val_t t)
	{
		m_states.push_back(state2_t(x.begin(), x.end()));
		m_times.push_back(t);
	}
};

struct compute_coherent_and_incoherent_energy
{
	val_t coh_en, incoh_en;
	val_t g1, g2, g1sqrt, g2sqrt;
	val_t t_init_to_skip;
	val_t t0;

	compute_coherent_and_incoherent_energy(val_t g1_, val_t g2_, val_t t_init_to_skip)
		: coh_en(0), incoh_en(0), g1(g1_), g2(g2_), g1sqrt(std::sqrtf(g1)), g2sqrt(std::sqrtf(g2)),
		  t_init_to_skip(t_init_to_skip), t0(t_init_to_skip) {}

	void operator()(const state_t& x, val_t t)
	{
		if (t > t_init_to_skip)
		{
			val_t dt = t - t0;
			t0 = t;
			val_t sx = g1sqrt * x[0] + g2sqrt * x[3];
			val_t sy = g1sqrt * x[1] + g2sqrt * x[4];
			coh_en += (sx * sx + sy * sy) * 0.25f * dt;
			incoh_en += (g1 * (1.f + x[2]) + g2 * (1.f + x[5]) + g1sqrt * g2sqrt * (x[6] + x[10])) * 0.5f * dt;
		}
	}
};

class qwm_solver
{
	/*
	t_init_to_skip is the parameter that allows to skip the fast-oscilating part in the beginning of the solution
		in the function rosenbrock_solve2()
	*/
	val_t eta1, g1, g2, alpha, domega, eps1, eps2;
	val_t t0, t1, dt, t_init_to_skip;

public:
	qwm_solver(params_t params, val_t T, val_t time_step, val_t t_init_to_skip):
		eta1(std::get<0>(params)), g1(std::get<1>(params)), g2(std::get<2>(params)), alpha(std::get<3>(params)),
		domega(std::get<4>(params)), eps1(std::get<5>(params)), eps2(std::get<6>(params)), t0(0.f), t1(T), dt(time_step),
		t_init_to_skip(t_init_to_skip)
	{}

	std::tuple<tlist_t, std::vector<state2_t>> rosenbrock_solve(const std::tuple<val_t, val_t> &amplitudes)
	{
		using namespace boost::numeric::odeint;
		Eq eq(std::get<0>(amplitudes), std::get<1>(amplitudes), eta1, g1, g2, alpha, domega, eps1, eps2);
		state_t x(N, 0.f);
		x[2] = -1.f;
		x[5] = -1.f;
		x[14] = 1.f;

		int total_steps = static_cast<int>(roundf((t1 - t0) / dt));

		tlist_t times;
		times.reserve(total_steps);
		std::vector<state2_t> states;
		states.reserve(total_steps);

		rosenbrock4<val_t> rb4stepper;
		integrate_const(rb4stepper, make_pair(eq, eq.system), x, t0, t1, dt,
			push_back_state_and_time3(states, times));

		return std::make_tuple(times, states);
	}

	result_t rosenbrock_solve2(const std::tuple<val_t, val_t>& amplitudes)
	{
		using namespace boost::numeric::odeint;
		Eq eq(std::get<0>(amplitudes), std::get<1>(amplitudes), eta1, g1, g2, alpha, domega, eps1, eps2);

		state_t x(N, 0.f);
		x[2] = -1.f;
		x[5] = -1.f;
		x[14] = 1.f;

		peaks_t qubit1(8, cmpx(0.f));
		peaks_t qubit2(8, cmpx(0.f));

		rosenbrock4<val_t> rb4stepper;
		integrate_const(rb4stepper, make_pair(eq, eq.system), x, t0, t1, dt,
			push_back_state_and_time2(qubit1, qubit2, domega, t_init_to_skip));

		return std::make_tuple(qubit1, qubit2);
	}
	
	std::tuple<val_t, val_t> solve_for_power(const std::tuple<val_t, val_t>& amplitudes)
	{
		using namespace boost::numeric::odeint;
		Eq eq(std::get<0>(amplitudes), std::get<1>(amplitudes), eta1, g1, g2, alpha, domega, eps1, eps2);

		state_t x(N, 0.f);
		x[2] = -1.f;
		x[5] = -1.f;
		x[14] = 1.f;

		peaks_t qubit1(8, cmpx(0.f));
		peaks_t qubit2(8, cmpx(0.f));

		compute_coherent_and_incoherent_energy computer(g1, g2, t_init_to_skip);

		rosenbrock4<val_t> rb4stepper;
		integrate_const(rb4stepper, make_pair(eq, eq.system), x, t0, t1, dt,
			push_back_state_and_time2(qubit1, qubit2, domega, t_init_to_skip));

		return std::make_tuple(computer.coh_en, computer.incoh_en);
	}

	result_t operator()(const std::tuple<val_t, val_t> &amplitudes)
	{
		return rosenbrock_solve2(amplitudes);
	}

	std::tuple<tlist_t, std::vector<state2_t>> rk45_solve(const std::tuple<val_t, val_t> &amplitudes)
	{
		using namespace boost::numeric::odeint;
		Eq eq(std::get<0>(amplitudes), std::get<1>(amplitudes), eta1, g1, g2, alpha, domega, eps1, eps2);
		state_t x(N, 0.f);
		x[2] = -1.f;
		x[5] = -1.f;
		x[14] = 1.f;

		int total_steps = static_cast<int>(roundf((t1 - t0) / dt));

		tlist_t times;
		times.reserve(total_steps);
		std::vector<state2_t> states;
		states.reserve(total_steps);

		//runge_kutta4_classic<state_t, val_t> stepper;
		integrate(eq, x, t0, t1, dt,
			push_back_state_and_time3(states, times));
		return std::make_tuple(times, states);
	}


	std::vector<result_t> rosenbrock_solve_multiple(std::vector<val_t> W_amplitudes, std::vector<val_t> E_amplitudes)
	{
		std::vector<std::tuple<val_t, val_t>> amplitudes;
		for (val_t& wamp : W_amplitudes)
			for (val_t& eamp : E_amplitudes)
				amplitudes.push_back({ wamp, eamp });
		std::vector<result_t> results(amplitudes.size());
		std::transform(amplitudes.begin(), amplitudes.end(), results.begin(), *this);
		return results;
	}

};

std::vector<result_t> qwm_parallel_solve(params_t params, val_t T, val_t time_step,
	std::vector<val_t> W_amplitudes, std::vector<val_t> E_amplitudes,
	val_t t_init_to_skip)
{
	qwm_solver qwmsolver(params, T, time_step, t_init_to_skip);
	std::vector<std::tuple<val_t, val_t>> amplitudes;
	for (val_t& wamp : W_amplitudes)
		for (val_t& eamp : E_amplitudes)
			amplitudes.push_back({ wamp, eamp });
	std::vector<result_t> results(amplitudes.size());

	struct Transform {
		void operator()(const tbb::blocked_range<size_t>& r) const {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				results[i] = solver.rosenbrock_solve2(ampls[i]);
			}
		}
		std::vector<std::tuple<val_t, val_t>>& ampls;
		std::vector<result_t>& results;
		qwm_solver& solver;
	} transform {amplitudes, results, qwmsolver};

	tbb::parallel_for(tbb::blocked_range<size_t>(0, amplitudes.size()), transform);

	return results;
}

std::tuple<map_t, map_t> coherent_and_incoherent_powers(params_t params, val_t T, val_t time_step,
	std::vector<val_t> W_amplitudes, std::vector<val_t> E_amplitudes,
	val_t t_init_to_skip)
{
	qwm_solver qwmsolver(params, T, time_step, t_init_to_skip);
	std::vector<std::tuple<val_t, val_t>> amplitudes;
	for (val_t& wamp : W_amplitudes)
		for (val_t& eamp : E_amplitudes)
			amplitudes.push_back({ wamp, eamp });
	map_t coherent_energy(amplitudes.size());
	map_t incoherent_energy(amplitudes.size());

	struct Transform {
		void operator()(const tbb::blocked_range<size_t>& r) const {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				auto res = solver.solve_for_power(ampls[i]);
				coh_en[i] = std::get<0>(res);
				incoh_en[i] = std::get<1>(res);
			}
		}
		std::vector<std::tuple<val_t, val_t>>& ampls;
		map_t& coh_en;
		map_t& incoh_en;
		qwm_solver& solver;
	} transform{ amplitudes, coherent_energy, incoherent_energy, qwmsolver };

	tbb::parallel_for(tbb::blocked_range<size_t>(0, amplitudes.size()), transform);

	return {coherent_energy, incoherent_energy};
}

PYBIND11_MODULE(QWM, m) {
	m.doc() = "Quantum Wave Mixing Solver"; // optional module docstring
	py::class_<qwm_solver>(m, "QWMSolver")
		.def(py::init<params_t, val_t, val_t, val_t>())
		.def("rosenbrock_solve", &qwm_solver::rosenbrock_solve,
			"Takes a tuple of amplitudes (W, E) and solves qwm system of equations with rosenbrock4 method")
		.def("rk45_solve", &qwm_solver::rk45_solve,
			"Takes a tuple of amplitudes (W, E) and solves qwm system of equations with Runge-Kutta 4 Cash-Karp method")
		.def("rosenbrock_solve_multiple", &qwm_solver::rosenbrock_solve_multiple,
			"Takes an array of W amplitudes and an array of E amplitudes and solves qwm system of equations on a grid of these amplitudes");
	m.def("qwm_parallel_solve", &qwm_parallel_solve);
	m.def("qwm_total_power", &coherent_and_incoherent_powers);
}
//void dump_in_file(const std::vector<result_t>& data)
//{
//	ofstream fout("result.dat", std::ofstream::out);
//	for (const auto& x : data)
//	{
//		for (int i = 0; i < 8; i++)
//			fout << std::get<0>(x)[i] << ' ';
//		fout << '\t';
//		for (int i = 0; i < 8; i++)
//			fout << std::get<1>(x)[i] << ' ';
//		fout << '\n';
//	}
//	fout.close();
//}
//
//int main()
//{
//	val_t pi = 3.14159265;
//	val_t eta1 = 2.f * pi * 0.1f;
//	val_t g1 = 2.f * pi * 2.f;
//	val_t g2 = 2.f * pi * 2.f;
//	val_t alpha = 0.4f;
//	val_t domega = 2.f * pi * 0.025f;
//	val_t eps1 = 0.f;
//	val_t eps2 = 0.f;
//
//	params_t params{ eta1, g1, g2, alpha, domega, eps1, eps2 };
//
//	val_t T = 160.f;
//	val_t dt = .1f;
//	
//	qwm_solver qwmsolver(params, T, dt);
//
//	std::vector<val_t> E_amps(51);
//	std::vector<val_t> W_amps(51);
//	std::generate(E_amps.begin(), E_amps.end(), [n = -2.f]() mutable { n += 0.1f;  return std::powf(10, n); });
//	std::generate(W_amps.begin(), W_amps.end(), [n = -2.f]() mutable { n += 0.1f;  return std::powf(10, n); });
//
//	auto start_time = std::chrono::high_resolution_clock::now();
//	//std::vector<result_t> results = qwmsolver.rosenbrock_solve_multiple(W_amps, E_amps);
//	std::vector<result_t> results = qwm_parallel_solve(params, T, dt, W_amps, E_amps);
//	auto end_time = std::chrono::high_resolution_clock::now();
//
//	dump_in_file(results);
//
//	std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
//
//	return 0;
//}