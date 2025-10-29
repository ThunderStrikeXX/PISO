#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>

// =======================================================================
//
//                        [SOLVING ALGORITHMS]
//
// =======================================================================

#pragma region solver

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

#pragma endregion

// =======================================================================
//
//                       [MATERIAL PROPERTIES]
//
// =======================================================================

#pragma region liquid_sodium_properties

/**
 * @brief Provides thermophysical properties for Liquid Sodium (Na).
 *
 * This namespace contains constant data and functions to calculate key
 * temperature-dependent properties of liquid sodium, which is commonly used
 * as a coolant in fast breeder reactors.
 * * All functions accept temperature T in **Kelvin [K]** and return values
 * in standard SI units.
 */
namespace liquid_sodium {

    // Critical temperature [K]
    constexpr double Tcrit = 2509.46;

    // Density [kg/m^3]
    double rho(double T) { return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5); }

    // Thermal conductivity [W/(m·K)]
    double k(double T) { return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T; }

    // Specific heat [J/(kg·K)]
    double cp(double T) {
        double dT = T - 273.15;
        return 1436.72 - 0.58 * dT + 4.627e-4 * dT * dT;
    }

    // Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    double mu(double T) { return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T); }
}

#pragma endregion

int main() {

    // =======================================================================
    //
    //                       [MATERIAL PROPERTIES]
    //
    // =======================================================================

    #pragma region constants_variables

	// Geometric parameters
    const double L = 1.0;                // Length of the domain
    const int N = 100;                   // Number of nodes (collocated grid)
    const double dz = L / (N - 1);       // Distance between nodes
    const double D_pipe = 0.1;           // Pipe diameter [m], used only to estimate Reynolds number

    // Physical parameters
    const double K = 1e-6;              // Permeability [m^2]
	const double CF = 0.0;              // Forchheimer coefficient [1/m]
    const double T_init = 200;

	// Time-stepping parameters
	const double dt = 0.001;                                // Timestep [s]
	const double t_max = 1.0;                               // Maximum time [s]
	const int t_iter = (int)std::round(t_max / dt);         // Number of timesteps

    // PISO parameters
	const int tot_iter = 200;            // Inner iterations per step [-]
    const int corr_iter = 2;             // PISO correctors per iteration [-]
	const double tol = 1e-8;             // Tolerance for the inner iterations [-]

    // Initial conditions
    std::vector<double> u(N, 0.001), p(N, 50000.0), T(N, T_init);                   // Collocated grid, values in center-cell
    std::vector<double> p_storage(N + 2, 50000.0);                                  // Storage for ghost nodes at the boundaries
    double* p_padded = &p_storage[1];                                               // Poìnter to work on the storage with the same indes
    std::vector<double> p_prime(N, 0.0);                                            // Pressure correction

    // Boundary conditions (Dirichlet p at outlet, T at both ends, u inlet)
    const double u_inlet = 0.0;             // Inlet velocity [m/s]
    const double u_outlet = 0.0;            // Outlet velocity [m/s]
    const double p_outlet = 50000.0;        // Outlet pressure [Pa]
    const double T_inlet = 300.0;           // Inlet temperature [K] (evaporator)
    const double T_outlet = 100.0;          // Outlet temperature [K] (condenser)

	// Output file
    std::ofstream fout("solution_PISO_liquid.txt");

    // Mass source and sink definitions
    std::vector<double> Sm(N, 0.0);

    const double mass_source_zone = 0.2;
    const double mass_sink_zone = 0.2;

    const double mass_source_nodes = std::floor(N * mass_source_zone);
    const double mass_sink_nodes = std::floor(N * mass_sink_zone);

    for (int i = 1; i < N - 1; ++i) {

        if (i > 0 && i <= mass_source_nodes) Sm[i] = 1.0;
        else if (i >= (N - mass_sink_nodes) && i < (N - 1)) Sm[i] = -1.0;

    }

    // Momentum source
    std::vector<double> Su(N, 0.0);

    // Energy source
    std::vector<double> St(N, 0.0);

    const double energy_source_zone = 0.2;
    const double energy_sink_zone = 0.2;

    const double energy_source_nodes = std::floor(N * energy_source_zone);
    const double energy_sink_nodes = std::floor(N * energy_sink_zone);

    for (int i = 1; i < N - 1; ++i) {

        if (i > 0 && i <= energy_source_nodes) St[i] = 1.0;
        else if (i >= (N - energy_sink_nodes) && i < (N - 1)) St[i] = -1.0;

    }

    // Models
    const int rhie_chow_on_off = 1;  // 0: no RC correction, 1: with RC correction

    // Pressure relaxation factor
	const double p_relax = 1.0;

    // Velocity relaxation factor
    const double u_relax = 1.0;

    // The coefficient bU is needed in momentum predictor loop and pressure correction to estimate the velocities at the faces using the Rhie and Chow correction
    std::vector<double> aU(N, 0.0), bU(N, liquid_sodium::rho(T_init) * dz / dt + 2 * liquid_sodium::mu(T_init) / dz), cU(N, 0.0), dU(N, 0.0);

    #pragma endregion

    // Number of processors aUailable for parallelization
    printf("Threads: %d\n", omp_get_max_threads());

    // Loop on timesteps
    for (double it = 0; it < t_iter; it++) {

        const double max_u = *std::max_element(u.begin(), u.end());
        const double min_T = *std::min_element(T.begin(), T.end());

        std::cout << "Solving! Time elapsed:" << dt * it << "/" << t_max
            << ", max courant number: " << max_u * dt / dz
            << ", max reynolds number: " << max_u * D_pipe * liquid_sodium::rho(min_T) / liquid_sodium::mu(min_T) << "\n";

        // PISO iterations
        int iter = 0;
        double maxErr = 1.0;    

        while (iter<tot_iter && maxErr>tol) {

            // =======================================================================
            //
            //                      [MOMENTUM PREDICTOR]
            //
            // =======================================================================

            #pragma region momentum_predictor

            #pragma omp parallel for
            for (int i = 1; i < N - 1; i++) {

                const double rho_P = liquid_sodium::rho(T[i]);
                const double rho_L = liquid_sodium::rho(T[i - 1]);
                const double rho_R = liquid_sodium::rho(T[i + 1]);

                const double D_l = 0.5 * (rho_P + rho_L) / dz;
                const double D_r = 0.5 * (rho_P + rho_R) / dz;

                const double mu_P = liquid_sodium::mu(T[i]);

                const double rhie_chow_l = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
                const double rhie_chow_r = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

                const double u_l_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
                const double u_r_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;

                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

                const double F_l = rho_l * u_l_face;
                const double F_r = rho_r * u_r_face;

                aU[i] = -std::max(F_l, 0.0) - D_l;
                cU[i] = std::max(-F_r, 0.0) - D_r;
                bU[i] = (std::max(F_r, 0.0) - std::max(-F_l, 0.0)) + rho_P * dz / dt + D_l + D_r + mu_P / K * dz + CF * mu_P * dz / sqrt(K) * abs(u[i]);
                dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho_P * u[i] * dz / dt + Su[i] * dz;
            }

            // Velocity BC: Dirichlet at l, dirichlet at r
            const double D_first = liquid_sodium::mu(T[0]) / dz;
            const double D_last = liquid_sodium::mu(T[N - 1]) / dz;

            bU[0] = (liquid_sodium::rho(T[0]) * dz / dt + 2 * D_first); cU[0] = 0.0; dU[0] = (liquid_sodium::rho(T[0]) * dz / dt + 2 * D_first) * u_inlet;
            aU[N - 1] = 0.0; bU[N - 1] = (liquid_sodium::rho(T[N - 1]) * dz / dt + 2 * D_last); dU[N - 1] = (liquid_sodium::rho(T[N - 1]) * dz / dt + 2 * D_last) * u_outlet;

            u = solveTridiagonal(aU, bU, cU, dU);

            #pragma endregion

            for (int piso = 0; piso < corr_iter; piso++) {

                // =======================================================================
                //
                //                       [PRESSURE CORRECTOR]
                //
                // =======================================================================

                #pragma region pressure_corrector

                std::vector<double> aP(N, 0.0), bP(N, 0.0), cP(N, 0.0), dP(N, 0.0);

                #pragma omp parallel for
                for (int i = 1; i < N - 1; i++) {

                    const double rho_i = liquid_sodium::rho(T[i]);

                    const double rhie_chow_l = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
                    const double rhie_chow_r = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

                    const double dudz = (u[i + 1] - u[i - 1]) / (2 * dz) + rhie_chow_on_off * (rhie_chow_r - rhie_chow_l);

                    const double rho_w = 0.5 * (liquid_sodium::rho(T[i - 1]) + rho_i);
                    const double d_w_face = 0.5 * (1.0 / bU[i - 1] + 1.0 / bU[i]); // 1/Ap average on west face
                    const double E_w = rho_w * d_w_face / (dz * dz);

                    const double rho_e = 0.5 * (rho_i + liquid_sodium::rho(T[i + 1]));
                    const double d_e_face = 0.5 * (1.0 / bU[i] + 1.0 / bU[i + 1]);  // 1/Ap average on east face
                    const double E_e = rho_e * d_e_face / (dz * dz);

                    const double u_w_star = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
                    const double mdot_w_star = (u_w_star > 0.0) ? liquid_sodium::rho(T[i - 1]) * u_w_star : rho_i * u_w_star;

                    const double u_e_star = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;
                    const double mdot_e_star = (u_e_star > 0.0) ? rho_i * u_e_star : liquid_sodium::rho(T[i + 1]) * u_e_star;

                    const double mass_imbalance = (mdot_e_star - mdot_w_star) / dz;

                    aP[i] = -E_w;
                    cP[i] = -E_e;
                    bP[i] = E_w + E_e;
                    dP[i] = Sm[i] - mass_imbalance;
                }

                // Pressure correction zero on the right side (pressure must remain constant)
                aP[N - 1] = 0; bP[N - 1] = 1; dP[N - 1] = 0;

                // Pressure correction gradient zero on the left side
                cP[0] = -1; bP[0] = 1; dP[0] = 0;

                p_prime = solveTridiagonal(aP, bP, cP, dP);
                printf("");

                #pragma endregion

                // =======================================================================
                //
                //                        [PRESSURE UPDATER]
                //
                // =======================================================================

                #pragma region pressure_updater

                for (int i = 0; i < N; i++) {

                    p[i] += p_prime[i];     // Note that PISO does not require an under-relaxation factor
                    p_storage[i + 1] = p[i];
                }

                p_storage[0] = p_storage[1];
                p_storage[N + 1] = p_outlet;

                #pragma endregion

                // =======================================================================
                //
                //                        [VELOCITY UPDATER]
                //
                // =======================================================================

                #pragma region velocity_updater

                maxErr = 0.0;
                for (int i = 1; i < N - 1; i++) {

                    double u_prev = u[i];
                    u[i] = u[i] - (p_prime[i + 1] - p_prime[i - 1]) / (2.0 * dz * bU[i]);

                    maxErr = std::max(maxErr, std::fabs(u[i] - u_prev));
                }

                #pragma endregion

            }

            iter++;
        }

        // =======================================================================
        //
        //                        [TEMPERATURE CALCULATOR]
        //
        // =======================================================================

        #pragma region temperature_calculator

        std::vector<double> aT(N, 0.0), bT(N, 0.0), cT(N, 0.0), dT(N, 0.0);

        #pragma omp parallel for
        for (int i = 1; i < N - 1; i++) {

            double k = liquid_sodium::k(T[i]);
            double cp = liquid_sodium::cp(T[i]);
            double rho_i = liquid_sodium::rho(T[i]);

            double rhoCp_dt = rho_i * cp / dt;

            double D_w = k / (dz * dz);
            double D_e = k / (dz * dz);

            double rhie_chow_l = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
            double rhie_chow_r = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

            double u_l_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
            double u_r_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;

            double rho_w = (u_l_face >= 0) ? liquid_sodium::rho(T[i - 1]) : rho_i;
            double rho_e = (u_r_face >= 0) ? rho_i : liquid_sodium::rho(T[i + 1]);

            double Fw = rho_w * u_l_face;
            double Fe = rho_e * u_r_face;

            double C_w_dx = (Fw * cp) / dz;
            double C_e_dx = (Fe * cp) / dz;

            double A_w = D_w + std::max(C_w_dx, 0.0);
            double A_e = D_e + std::max(-C_e_dx, 0.0);

            aT[i] = -A_w;
            cT[i] = -A_e;
            bT[i] = A_w + A_e + rhoCp_dt;

            dT[i] = rhoCp_dt * T[i] + St[i];
        }

        // Temperature BCs
        bT[0] = 1.0; cT[0] = 0.0; dT[0] = T_inlet;
        aT[N - 1] = 0.0; bT[N - 1] = 1.0; dT[N - 1] = T_outlet;

        T = solveTridiagonal(aT, bT, cT, dT);

        #pragma endregion

        // =======================================================================
        //
        //                                [OUTPUT]
        //
        // =======================================================================

        #pragma region output

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < N; i++) {
                fout << u[i] << ", ";
            }
        } fout << "\n\n";

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < N; i++) {
                fout << p[i] << ", ";
            }
        } fout << "\n\n";

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < N; i++) {
                fout << T[i] << ", ";
            }
        }

        #pragma endregion
    }

    fout.close();

    return 0;
}
