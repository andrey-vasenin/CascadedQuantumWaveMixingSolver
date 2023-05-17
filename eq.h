#pragma once

#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

using namespace std;

const int N = 15;

typedef float val_t;
typedef boost::numeric::ublas::vector<val_t> state_t;
typedef boost::numeric::ublas::matrix<val_t> matrix_t;
//typedef array<val_t, N> state_t;
//typedef array<array<val_t, N>, N> matrix_t;

void print_matrix(const matrix_t& M)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << M(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_vector(const state_t& v)
{
    for (int j = 0; j < N; j++)
    {
        std::cout << v(j) << " ";
    }
    std::cout << "\n\n";
}

class EqMatrix
{
    val_t t, W, E, eta1, g1, g2, alpha, domega, eps1, eps2;

    matrix_t M0, Ms, Mc;

public:
    state_t b;

    matrix_t M;

    EqMatrix(val_t W, val_t E, val_t eta1, val_t g1, val_t g2,
        val_t alpha, val_t domega, val_t eps1, val_t eps2) :
        t(-100.f), W(W), E(E), eta1(eta1), g1(g1), g2(g2), alpha(alpha), domega(domega), eps1(eps1), eps2(eps2), M0(N, N), Ms(N, N), Mc(N, N), b(N), M(N, N)
    {
        for (int i = 0; i < N; i++)
        {
            b[i] = 0.f;
            for (int j = 0; j < N; j++)
            {
                M0(i, j) = 0.f;
                Ms(i, j) = 0.f;
                Mc(i, j) = 0.f;
                M(i, j) = 0.f;
            }
        }
        set_M0();
        set_Mc();
        set_Ms();
    }

    void update_b_and_M(const float& _t)
    {
        if (t != _t)
        {
            t = _t;
            float cost = cos(domega * t);
            float sint = sin(domega * t);
            b[2] = -g1 - eta1;
            b[5] = -2.f * g2;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    M(i, j) = M0(i, j) + Mc(i, j) * cost + Ms(i, j) * sint;
        }
        //print_vector(b);
        //print_matrix(M);
    }

    void operator() (const state_t& x, matrix_t& J, const float& _t, state_t& dfdt)
    {
        update_b_and_M(_t);
        // Copying the Jacobian and computing dfdt
        float cost = cos(domega * t);
        float sint = sin(domega * t);

        for (int i = 0; i < N; i++)
        {
            dfdt[i] = 0.f;
            for (int j = 0; j < N; j++)
            {
                J(i, j) = M(i, j);
                dfdt[i] -= Mc(i, j) * sint * x[j];
                dfdt[i] += Ms(i, j) * cost * x[j];
            }
        }
    }

private:
    void set_M0()
    {
        M0(0, 0) = -(g1 / 2.f) - eta1 / 2.f;
        M0(0, 1) = -eps1;
        M0(1, 0) = eps1;
        M0(1, 1) = -(g1 / 2.f) - eta1 / 2.f;
        M0(2, 2) = -g1 - eta1;
        M0(3, 3) = -g2;
        M0(3, 4) = -eps2;
        M0(3, 8) = alpha * sqrt(g1 * g2);
        M0(4, 3) = eps2;
        M0(4, 4) = -g2;
        M0(4, 11) = alpha * sqrt(g1 * g2);
        M0(5, 5) = -2.f * g2;
        M0(5, 6) = (-alpha) * sqrt(g1 * g2);
        M0(5, 10) = (-alpha) * sqrt(g1 * g2);
        M0(6, 5) = alpha * sqrt(g1 * g2);
        M0(6, 6) = -(g1 / 2.f) - g2 - eta1 / 2.f;
        M0(6, 7) = -eps2;
        M0(6, 9) = -eps1;
        M0(6, 14) = alpha * sqrt(g1 * g2);
        M0(7, 6) = eps2;
        M0(7, 7) = -(g1 / 2.f) - g2 - eta1 / 2.f;
        M0(7, 10) = -eps1;
        M0(8, 0) = -2.f * g2;
        M0(8, 3) = (-alpha) * sqrt(g1 * g2);
        M0(8, 8) = -(g1 / 2.f) - 2.f * g2 - eta1 / 2.f;
        M0(8, 11) = -eps1;
        M0(8, 12) = (-alpha) * sqrt(g1 * g2);
        M0(9, 6) = eps1;
        M0(9, 9) = -(g1 / 2.f) - g2 - eta1 / 2.f;
        M0(9, 10) = -eps2;
        M0(10, 5) = alpha * sqrt(g1 * g2);
        M0(10, 7) = eps1;
        M0(10, 9) = eps2;
        M0(10, 10) = -(g1 / 2.f) - g2 - eta1 / 2.f;
        M0(10, 14) = alpha * sqrt(g1 * g2);
        M0(11, 1) = -2.f * g2;
        M0(11, 4) = (-alpha) * sqrt(g1 * g2);
        M0(11, 8) = eps1;
        M0(11, 11) = -(g1 / 2.f) - 2.f * g2 - eta1 / 2.f;
        M0(11, 13) = (-alpha) * sqrt(g1 * g2);
        M0(12, 3) = -g1 - eta1;
        M0(12, 8) = (-alpha) * sqrt(g1 * g2);
        M0(12, 12) = -g1 - g2 - eta1;
        M0(12, 13) = -eps2;
        M0(13, 4) = -g1 - eta1;
        M0(13, 11) = (-alpha) * sqrt(g1 * g2);
        M0(13, 12) = eps2;
        M0(13, 13) = -g1 - g2 - eta1;
        M0(14, 2) = -2.f * g2;
        M0(14, 5) = -g1 - eta1;
        M0(14, 6) = alpha * sqrt(g1 * g2);
        M0(14, 10) = alpha * sqrt(g1 * g2);
        M0(14, 14) = -g1 - 2.f * g2 - eta1;
    }

    void set_Mc()
    {
        Mc(0, 2) = 2.f * W * sqrt(eta1);
        Mc(2, 0) = -2.f * W * sqrt(eta1);
        Mc(3, 5) = 2.f * E * sqrt(g2);
        Mc(5, 3) = -2.f * E * sqrt(g2);
        Mc(6, 8) = 2.f * E * sqrt(g2);
        Mc(6, 12) = 2.f * W * sqrt(eta1);
        Mc(7, 13) = 2.f * W * sqrt(eta1);
        Mc(8, 6) = -2.f * E * sqrt(g2);
        Mc(8, 14) = 2.f * W * sqrt(eta1);
        Mc(9, 11) = 2.f * E * sqrt(g2);
        Mc(11, 9) = -2.f * E * sqrt(g2);
        Mc(12, 6) = -2.f * W * sqrt(eta1);
        Mc(12, 14) = 2.f * E * sqrt(g2);
        Mc(13, 7) = -2.f * W * sqrt(eta1);
        Mc(14, 8) = -2.f * W * sqrt(eta1);
        Mc(14, 12) = -2.f * E * sqrt(g2);
    }

    void set_Ms()
    {
        Ms(1, 2) = 2.f * W * sqrt(eta1);
        Ms(2, 1) = -2.f * W * sqrt(eta1);
        Ms(4, 5) = -2.f * E * sqrt(g2);
        Ms(5, 4) = 2.f * E * sqrt(g2);
        Ms(7, 8) = -2.f * E * sqrt(g2);
        Ms(8, 7) = 2.f * E * sqrt(g2);
        Ms(9, 12) = 2.f * W * sqrt(eta1);
        Ms(10, 11) = -2.f * E * sqrt(g2);
        Ms(10, 13) = 2.f * W * sqrt(eta1);
        Ms(11, 10) = 2.f * E * sqrt(g2);
        Ms(11, 14) = 2.f * W * sqrt(eta1);
        Ms(12, 9) = -2.f * W * sqrt(eta1);
        Ms(13, 10) = -2.f * W * sqrt(eta1);
        Ms(13, 14) = -2.f * E * sqrt(g2);
        Ms(14, 11) = -2.f * W * sqrt(eta1);
        Ms(14, 13) = 2.f * E * sqrt(g2);
    }
};

class Eq
{
public:
    EqMatrix system;

    Eq(val_t W, val_t E, val_t eta1, val_t g1, val_t g2,
        val_t alpha, val_t domega, val_t eps1, val_t eps2) :
        system(W, E, eta1, g1, g2, alpha, domega, eps1,eps2)
    {
    }

    void operator() (const state_t& x, state_t& dxdt, val_t t)
    {
        system.update_b_and_M(t);
        for (int i = 0; i < N; i++)
        {
            dxdt[i] = system.b[i];
            for (int j = 0; j < N; j++)
            {
                dxdt[i] += system.M(i, j) * x[j];
            }
        }
    }
};