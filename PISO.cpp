#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>

// Solves a tridiagonal system Ax = d using the Thomas algorithm
// a, b, c are the sub-diagonal, main diagonal, and super-diagonal of A
// d is the right-hand side vector 
std::vector<double> solveTridiagonal(const std::vector<double>& a,
                                        const std::vector<double>& b,
                                        const std::vector<double>& c,
                                        const std::vector<double>& d) {
    int n = b.size();
    std::vector<double> c_star(n, 0.0);
    std::vector<double> d_star(n, 0.0);
    std::vector<double> x(n, 0.0);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }

    return x;
}

double T_Na_crit() {
    return 2509.46;
}

double rho_Na_l(double T) {
    double Tc = T_Na_crit();
    return 219.0 + 275.32 * (1.0 - T / Tc) + 511.58 * pow(1.0 - T / Tc, 0.5);
}

double k_Na_l(double T) {
    return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T;
}

double Cp_Na_l(double T) {
    double dT = T - 273.15;
    return 1436.72 - 0.58 * dT + 4.627e-4 * dT * dT;
}

double mu_Na(double T) {
    // Dynamic viscosity of liquid sodium [Pa·s]
    // Correlation: Shpil rain et al. (1985)
    // Valid for 371 K < T < 2500 K
    return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T);
}

int main() {

	// Physical and domain parameters
    const double L = 0.01;              // Length of the domain
	//   const double mu_eff = 0.1 * mu;  // Effective viscosity for DBF wq [Pa s]
    const double K = 1e-6;              // Permeability [m^2]
	const double CF = 0.0;              // Forchheimer coefficient [1/m]

	// Numerical parameters
    const int Nx = 500;                  // Number of nodes (collocated grid)
    const double dx = L / (Nx - 1);      // Distance between nodes
	const double dt = 0.0001;              // Timestep [s]
	const double t_max = 0.5;            // Maximum time [s]
	const int t_iter = t_max / dt;         // Number of timesteps
	const int SIMPLE_iter = 1000;         // Maximum number of SIMPLE iterations
	const double tol = 1e-8;             // Convergence tolerance on the velocity field

    // Initial conditions
    std::vector<double> u(Nx, 0.0), p(Nx, 0.0), T(Nx, 300.0);
    std::vector<double> p_prime(Nx, 0.0);

    const double u_inlet = 0.01;
    const double p_outlet = 0.0;

    const double T_inlet = 1000.0;
    const double T_outlet = 500.0;

    // Boundary conditions
    u[0] = u_inlet;            // Velocity on the left fixed at 0.01
    u[Nx - 1] = u[Nx - 2];  // Zero gradient velocity on the right

    p[Nx - 1] = p_outlet;        // Zero pressure on the right
    p[0] = p[1];            // Zero gradient pressure on the left

    T[0] = T_inlet;           // Temperature on the left fixed at 350
    T[Nx - 1] = T_outlet;      // Temperature on the right fixed at 300

	// Output file
    std::ofstream fout("solution_SIMPLEC_thermal.txt");

    // Rhie and Chow amplification coefficient
    const double rhiechow_coeff = 1.0;

    // Pressure relaxation factor
	const double p_relax = 0.5;

    // Velocity relaxation factor
    const double v_relax = 1.0;

    printf("Threads: %d\n", omp_get_max_threads());

    // Loop on timesteps
    for (double it = 0; it < t_iter; it++) {

        printf("Time: %f / %f, Courant number: %f \n", dt * it, t_max, 0.01 * dt / dx);

		// Loop on SIMPLE iterations
        int iter = 0;
        double maxErr = 1.0;     // This is to enter the loop, then it updates

        while (iter<SIMPLE_iter && maxErr>tol) {

            // 1. Momentum predictor: from u to u_star

            std::vector<double> a(Nx, 0.0), b(Nx, 0.0), c(Nx, 0.0), d(Nx, 0.0), b_SIMPLEC(Nx, 0.0);;

            #pragma omp parallel for
            for (int ix = 1; ix < Nx - 1; ix++) {

				double rho_node = rho_Na_l(T[ix]);
				double mu_node = mu_Na(T[ix]);

                a[ix] = -mu_node / (dx * dx);
                c[ix] = -mu_node / (dx * dx);
                b[ix] = rho_node / dt + 2 * mu_node / (dx * dx) + rho_node * (u[ix] - u[ix - 1]) / dx + mu_node / K + CF * rho_node / sqrt(K) * abs(u[ix]);
                d[ix] = rho_node * u[ix] / dt - (p[ix + 1] - p[ix - 1]) / (2 * dx);
                b_SIMPLEC[ix] = b[ix] - (c[ix] + a[ix]);
            }

            // Velocity fixed at 0.01 on the left
            b[0] = 1;
            c[0] = 0;
            d[0] = u_inlet;
            double b_0_RC = rho_Na_l(T[0]) / dt + 2 * mu_Na(T[0]) / (dx * dx) + mu_Na(T[0]) / K + CF * rho_Na_l(T[0]) / sqrt(K) * abs(u[0]);
            b_SIMPLEC[0] = rho_Na_l(T[0]) / dt + 2 * mu_Na(T[0]) / (dx * dx) + mu_Na(T[0]) / K + CF * rho_Na_l(T[0]) / sqrt(K) * abs(u[0]);

            // Zero gradient velocity on the right
            a[Nx - 1] = -1;
            b[Nx - 1] = 1;
            d[Nx - 1] = 0;
			double b_Nx_1_RC = rho_Na_l(T[Nx - 1]) / dt + 3 * mu_Na(T[Nx - 1]) / (dx * dx) + mu_Na(T[Nx - 1]) / K + CF * rho_Na_l(T[Nx - 1]) / sqrt(K) * abs(u[Nx - 1]);
            b_SIMPLEC[Nx - 1] = rho_Na_l(T[Nx - 1]) / dt + 3 * mu_Na(T[Nx - 1]) / (dx * dx) + mu_Na(T[Nx - 1]) / K + CF * rho_Na_l(T[Nx - 1]) / sqrt(K) * abs(u[Nx - 1]);

            u = solveTridiagonal(a, b, c, d);

			// 2. Pressure correction equation: from u_star to p_prime
            std::vector<double> aP(Nx, 0.0), bP(Nx, 0.0), cP(Nx, 0.0), dP(Nx, 0.0);

            double rhiechow = 0.0;
            double dudx = 0.0;

            #pragma omp parallel for
            for (int ix = 2; ix < Nx - 2; ix++) {

                rhiechow = -1 / (8 * dx) * (1 / b[ix + 1] - 1 / b[ix - 1]) *
                    (-p[ix - 2] + 4 * p[ix - 1] - 6 * p[ix] + 4 * p[ix + 1] - p[ix + 2]);

                dudx = (u[ix + 1] - u[ix - 1]) / (2 * dx)
                    + rhiechow_coeff * rhiechow;

                aP[ix] = 1.0 / dx / dx;
                cP[ix] = 1.0 / dx / dx;
                bP[ix] = -2.0 / dx / dx;
                dP[ix] = rho_Na_l(T[ix]) / dt * dudx;
            }

            // Pressure correction zero on the right side (pressure must remain constant)
            aP[Nx - 1] = 0;
            bP[Nx - 1] = 1;
            dP[Nx - 1] = 0;

            // Pressure correction gradient zero on the left side
            cP[0] = -1;
            bP[0] = 1;
            dP[0] = 0;

            // Pressure correction without Rhie and Chow for the second node
            aP[1] = 1.0 / (dx * dx);
            cP[1] = 1.0 / (dx * dx);
            bP[1] = -2.0 / (dx * dx);

            double rhiechow1 = -1 / (8 * dx) * (1 / b[2] - 1 / b_0_RC) *
                (-3 * p[0] + 4 * p[2] - p[3]);
            dP[1] = rho_Na_l(T[1]) / dt * ((u[2] - u[0]) / (2 * dx) + rhiechow_coeff * rhiechow1);

            // Pressure correction with Rhie and Chow for the second-to-last node
            aP[Nx - 2] = 1.0 / dx / dx;
            cP[Nx - 2] = 1.0 / dx / dx;
            bP[Nx - 2] = -2.0 / dx / dx;

            double rhiechow2 = -1 / (8 * dx) * (1 / b_Nx_1_RC - 1 / b[Nx - 3]) *
                (-p[Nx - 4] + 4 * p[Nx - 3] - 6 * p[Nx - 2]);
            dP[Nx - 2] = rho_Na_l(T[Nx - 2]) / dt * ((u[Nx - 1] - u[Nx - 3]) / (2 * dx) + rhiechow_coeff * rhiechow2);

            p_prime = solveTridiagonal(aP, bP, cP, dP);

			// 3. Pressure correction loop: from p_prime to p
            for (int ix = 0; ix < Nx; ix++) {

                p[ix] += p_relax * p_prime[ix]; // With relaxing factor
            }

            // Error on the velocity field
            maxErr = 0.0;

            // 4. Velocity correction loop
            for (int ix = 1; ix < Nx - 1; ix++) {

                double u_old = u[ix];
                u[ix] = u[ix] - v_relax / b_SIMPLEC[ix] * (p_prime[ix + 1] - p_prime[ix - 1]) / (2 * dx);

                maxErr = std::max(maxErr, std::fabs(u[ix] - u_old));
            }

            // Boundary condition, zero gradient on right side velocity
            u[Nx - 1] = u[Nx - 2];

            iter++;
        }

        // 5. Temperature loop
        std::vector<double> aT(Nx, 0.0), bT(Nx, 0.0), cT(Nx, 0.0), dT(Nx, 0.0);

        #pragma omp parallel for
        for (int ix = 1; ix < Nx - 1; ix++) {

			double alpha_node = k_Na_l(T[ix]) / (rho_Na_l(T[ix]) * Cp_Na_l(T[ix]));

			// Fully implicit discretization 
            aT[ix] = -alpha_node / (dx * dx) - u[ix] / dx;
            cT[ix] = -alpha_node / (dx * dx);
            bT[ix] = 1 / dt + u[ix] / dx + 2 * alpha_node / (dx * dx);
            dT[ix] = T[ix] / dt;

			// Implicit diffusion, explicit convection discretization
            /*aT[ix] = -alpha / (dx * dx);
            cT[ix] = -alpha / (dx * dx);
            bT[ix] = 1 / dt + 2 * alpha / (dx * dx);
            dT[ix] = 1 / dt * T[ix] + u[ix] * (T[ix] - T[ix - 1]) / dx;*/

            // Explicit diffusion, implicit convection discretization
            //aT[ix] = - u[ix] / dx;
            //cT[ix] = 0;
            //bT[ix] = 1 / dt + u[ix] / dx;
            //dT[ix] = T[ix] / dt + alpha / (dx * dx) * (T[ix + 1] - 2 * T[ix] + T[ix - 1]);

            // Fully explicit discretization
            /*aT[ix] = 0;
            cT[ix] = 0;
            bT[ix] = 1;
            dT[ix] = T[ix] + 
                dt * alpha / (dx * dx) * (T[ix + 1] - 2 * T[ix] + T[ix - 1]) - 
                u[ix] * (T[ix] - T[ix - 1]);*/
        }

        // Fixed temperature on the left
        bT[0] = 1;
        cT[0] = 0;
        dT[0] = T_inlet;

        // Fixed temperature on the right
        aT[Nx - 1] = 0;
        bT[Nx - 1] = 1;
        dT[Nx - 1] = T_outlet;

        T = solveTridiagonal(aT, bT, cT, dT);

        if(it == (t_iter - 1)) {

            // Output
            for (int ix = 0; ix < Nx; ix++) {

                // fout << dt*it << " " << ix * dx << " " << u[ix] << " " << p[ix] << " " << T[ix] << "\n";
			    // fout << T[ix] << ", ";
                // fout << u[ix] << ", ";
                fout << p[ix] << ", ";
            }
            fout << "\n";
        }
    }
    fout.close();

    return 0;
}
