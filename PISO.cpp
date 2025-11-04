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

// Initializes vector with equally spaced values between min and max
std::vector<double> linspace(double T_min, double T_max, int N) {
    std::vector<double> T(N);
    double dT = (T_max - T_min) / (N - 1);
    for (int i = 0; i < N; i++) T[i] = T_min + i * dT;
    return T;
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

    // Solidification temperature, gives warning if below
    constexpr double Tsolid = 370.87;

    // Density [kg/m^3]
    double rho(double T) { 

        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5); 
    }

    // Thermal conductivity [W/(m·K)]
    double k(double T) { 
        
        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T; 
    }

    // Specific heat [J/(kg·K)]
    double cp(double T) {

        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        double dXT = T - 273.15;
        return 1436.72 - 0.58 * dXT + 4.627e-4 * dXT * dXT;
    }

    // Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    double mu(double T) { 
        
        if (T < Tsolid) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T); 
    }
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
    const double T_init = 600;

	// Time-stepping parameters
	const double dt = 0.1;                                // Timestep [s]
	const double t_max = 1000.0;                               // Maximum time [s]
	const int t_iter = (int)std::round(t_max / dt);         // Number of timesteps

    // PISO parameters
	const int tot_iter = 200;            // Inner iterations per step [-]
    const int corr_iter = 2;             // PISO correctors per iteration [-]
	const double tol = 1e-8;             // Tolerance for the inner iterations [-]

    // Initial conditions
    std::vector<double> u(N, -0.001), p(N, 50000.0), T(N, T_init);                   // Collocated grid, values in center-cell
    std::vector<double> p_storage(N + 2, 50000.0);                                  // Storage for ghost nodes at the boundaries
    double* p_padded = &p_storage[1];                                               // Poìnter to work on the storage with the same indes
    std::vector<double> T_old(N, T_init), p_old(N, 50000.0);        // Backup values
    std::vector<double> p_prime(N, 0.0);                                            // Pressure correction

    // Boundary conditions (Dirichlet p at outlet, T at both ends, u inlet)
    const double u_inlet = 0.0;             // Inlet velocity [m/s]
    const double u_outlet = 0.0;            // Outlet velocity [m/s]
    const double p_outlet = 50000.0;        // Outlet pressure [Pa]

	// Output file
    std::ofstream fout("solution_PISO_liquid.txt");

    // Mass source and sink definitions
    std::vector<double> Sm(N, 0.0);

    const double mass_source_zone = 0.2;
    const double mass_sink_zone = 0.2;

    const double mass_source_nodes = std::floor(N * mass_source_zone);
    const double mass_sink_nodes = std::floor(N * mass_sink_zone);

    for (int i = 1; i < N - 1; ++i) {

        if (i > 0 && i <= mass_source_nodes) Sm[i] = -1.0;
        else if (i >= (N - mass_sink_nodes) && i < (N - 1)) Sm[i] = +1.0;

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

        if (i > 0 && i <= energy_source_nodes) St[i] = 1000000.0;
        else if (i >= (N - energy_sink_nodes) && i < (N - 1)) St[i] = -1000000.0;

    }

    // Models
    const int rhie_chow_on_off = 1;  // 0: no RC correction, 1: with RC correction

    // The coefficient bU is needed in momentum predictor loop and pressure correction to estimate the velocities at the faces using the Rhie and Chow correction
    std::vector<double> aU(N, 0.0), bU(N, liquid_sodium::rho(T_init) * dz / dt + 2 * liquid_sodium::mu(T_init) / dz), cU(N, 0.0), dU(N, 0.0);

    #pragma endregion

    // Number of processors aUailable for parallelization
    printf("Threads: %d\n", omp_get_max_threads());

    // Loop on timesteps
    for (double it = 0; it < t_iter; it++) {

        const double max_abs_u =
            std::abs(*std::max_element(u.begin(), u.end(),
                [](double a, double b) { return std::abs(a) < std::abs(b); }
            ));
        const double min_T = *std::min_element(T.begin(), T.end());

        std::cout << "Solving! Time elapsed:" << dt * it << "/" << t_max
            << ", max courant number: " << max_abs_u * dt / dz
            << ", max reynolds number: " << max_abs_u * D_pipe * liquid_sodium::rho(min_T) / liquid_sodium::mu(min_T) << "\n";

        // Backup variables
        T_old = T;
        p_old = p;

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

                const double mu_P = liquid_sodium::mu(T[i]);
                const double mu_L = liquid_sodium::mu(T[i - 1]);
                const double mu_R = liquid_sodium::mu(T[i + 1]);

                const double D_l = 0.5 * (mu_P + mu_L) / dz;
                const double D_r = 0.5 * (mu_P + mu_R) / dz;

                const double rhie_chow_l = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
                const double rhie_chow_r = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

                const double u_l_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
                const double u_r_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;

                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

                const double F_l = rho_l * u_l_face;
                const double F_r = rho_r * u_r_face;

                aU[i] = -std::max(F_l, 0.0) - D_l;
                cU[i] = -std::max(-F_r, 0.0) - D_r;
                bU[i] = (std::max(F_r, 0.0) + std::max(-F_l, 0.0)) + rho_P * dz / dt + D_l + D_r + mu_P / K * dz + CF * mu_P * dz / sqrt(K) * abs(u[i]);
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
                //                       [CONTINUITY SATISFACTOR]
                //
                // =======================================================================

                #pragma region continuity_satisfactor

                std::vector<double> aP(N, 0.0), bP(N, 0.0), cP(N, 0.0), dP(N, 0.0);

                #pragma omp parallel for
                for (int i = 1; i < N - 1; i++) {

                    const double rho_P = liquid_sodium::rho(T[i]);
                    const double rho_L = liquid_sodium::rho(T[i - 1]);
                    const double rho_R = liquid_sodium::rho(T[i + 1]);

                    const double rhie_chow_l = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
                    const double rhie_chow_r = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

                    const double rho_l = 0.5 * (rho_L + rho_P);
                    const double d_l_face = 0.5 * (1.0 / bU[i - 1] + 1.0 / bU[i]); // 1/Ap average on west face
                    const double E_l = rho_l * d_l_face / dz;

                    const double rho_r = 0.5 * (rho_P + rho_R);
                    const double d_r_face = 0.5 * (1.0 / bU[i] + 1.0 / bU[i + 1]);  // 1/Ap average on east face
                    const double E_r = rho_r * d_r_face / dz;

                    const double u_l_star = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
                    const double mdot_l_star = (u_l_star > 0.0) ? rho_L * u_l_star : rho_P * u_l_star;

                    const double u_r_star = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;
                    const double mdot_r_star = (u_r_star > 0.0) ? rho_P * u_r_star : rho_R * u_r_star;

                    const double mass_imbalance = (mdot_r_star - mdot_l_star);

                    aP[i] = -E_l;
                    cP[i] = -E_r;
                    bP[i] = E_l + E_r;          // No compressibility term
                    dP[i] = Sm[i] * dz - mass_imbalance;
                }

                // BCs for p': zero gradient aVT inlet and zero correction aVT outlet
                bP[0] = 1.0; cP[0] = -1.0; dP[0] = 0.0;
                bP[N - 1] = 1.0; aP[N - 1] = 0.0; dP[N - 1] = 0.0;

                p_prime = solveTridiagonal(aP, bP, cP, dP);

                #pragma endregion

                // =======================================================================
                //
                //                        [PRESSURE CORRECTOR]
                //
                // =======================================================================

                #pragma region pressure_corrector

                for (int i = 0; i < N; i++) {

                    p[i] += p_prime[i];     // Note that PISO does not require an under-relaxation factor
                    p_storage[i + 1] = p[i];
                }

                p_storage[0] = p_storage[1];
                p_storage[N + 1] = p_outlet;

                #pragma endregion

                // =======================================================================
                //
                //                        [VELOCITY CORRECTOR]
                //
                // =======================================================================

                #pragma region velocity_corrector

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

        std::vector<double> aXT(N, 0.0), bXT(N, 0.0), cXT(N, 0.0), dXT(N, 0.0);

        #pragma omp parallel for
        for (int i = 1; i < N - 1; i++) {

            const double rho_P = liquid_sodium::rho(T[i]);
            const double rho_L = liquid_sodium::rho(T[i - 1]);
            const double rho_R = liquid_sodium::rho(T[i + 1]);

            const double k_cond_P = liquid_sodium::k(T[i]);
            const double k_cond_L = liquid_sodium::k(T[i - 1]);
            const double k_cond_R = liquid_sodium::k(T[i + 1]);

            const double cp_P = liquid_sodium::cp(T[i]);
            const double cp_L = liquid_sodium::cp(T[i - 1]);
            const double cp_R = liquid_sodium::cp(T[i + 1]);

            const double rhoCp_dzdt = rho_P * cp_P * dz / dt;

            // Linear interpolation diffusion coefficient
            const double D_l = 0.5 * (k_cond_P + k_cond_L) / dz;
            const double D_r = 0.5 * (k_cond_P + k_cond_R) / dz;

            const double rhie_chow_l = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
            const double rhie_chow_r = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

            const double u_l_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
            const double u_r_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;

            // Upwind density
            const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
            const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

            // Upwind specific heat
            const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;
            const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;

            const double Fl = rho_l * u_l_face;
            const double Fr = rho_r * u_r_face;

            const double C_l = (Fl * cp_l);
            const double C_r = (Fr * cp_r);

            aXT[i] = -D_l - std::max(C_l, 0.0);
            cXT[i] = -D_r - std::max(-C_r, 0.0);
            bXT[i] = (std::max(C_r, 0.0) + std::max(-C_l, 0.0)) + D_l + D_r + rhoCp_dzdt;

            dXT[i] = rhoCp_dzdt * T_old[i] + St[i] * dz;
        }

        // Temperature BCs
        bXT[0] = 1.0; cXT[0] = -1.0; dXT[0] = 0.0;
        aXT[N - 1] = -1.0; bXT[N - 1] = 1.0; dXT[N - 1] = 0.0;

        T = solveTridiagonal(aXT, bXT, cXT, dXT);

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
