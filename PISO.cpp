#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <filesystem>
#include <omp.h>

#include "tdma.h"

#pragma region input

namespace fs = std::filesystem;

std::string chooseInputFile(const std::string& inputDir) {
    std::vector<fs::path> files;

    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        throw std::runtime_error("Input directory not found: " + inputDir);
    }

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }

    if (files.empty()) {
        throw std::runtime_error("No input files found in directory: " + inputDir);
    }

    std::cout << "Available input files:\n";
    for (std::size_t i = 0; i < files.size(); ++i) {
        std::cout << "  [" << i << "] "
            << files[i].filename().string() << '\n';
    }

    std::cout << "Select input file by number: ";
    std::size_t choice;
    std::cin >> choice;

    if (choice >= files.size()) {
        throw std::runtime_error("Invalid selection index");
    }

	return files[choice].string(); // Complete path to the selected file
}
          
struct Input {

    int    N = 0;                           // Number of cells [-]
    double L = 0.0;                         // Length of the domain [m]

    double dt_user = 0.0;                   // User-defined time step [s]
    double simulation_time = 0.0;           // Total simulation time [s]

    int    piso_outer_iter = 0;             // PISO outer iterations [-]
    int    piso_inner_iter = 0;             // PISO inner iterations [-]
    double piso_outer_tol = 0.0;            // PISO outer tolerance [-]
    double piso_inner_tol = 0.0;            // PISO inner tolerance [-]
    bool   rhie_chow_on_off_l = true;       // Rhie–Chow on/off [-]

    double rho = 0.0;                       // Density [kg/m3]
    double mu = 0.0;                        // Dynamic viscosity [kg/(m s)]
	double k = 0.0;                         // Thermal conductivity [W/(m K)]
	double cp = 0.0;                        // Specific heat capacity [J/(kg K)]

    double S_m_cell = 0.0;                  // Volumetric mass source [kg/(m3 s)]
    double S_h_cell = 0.0;                  // Volumetric heat source [W/m3]

    double z_evap_start = 0.0;              // Evaporation zone start [m]
    double z_evap_end = 0.0;                // Evaporation zone end [m]
    double z_cond_start = 0.0;              // Condensation zone start [m]
    double z_cond_end = 0.0;                // Condensation zone end [m]

    int    u_inlet_bc = 0;                  // 0 Dirichlet, 1 Neumann
    double u_inlet_value = 0.0;             // [m/s]

    int    u_outlet_bc = 0;                 // 0 Dirichlet, 1 Neumann
    double u_outlet_value = 0.0;            // [m/s]

    int    T_inlet_bc = 0;                  // 0 Dirichlet, 1 Neumann
    double T_inlet_value = 0.0;             // [K]

    int    T_outlet_bc = 0;                 // 0 Dirichlet, 1 Neumann
    double T_outlet_value = 0.0;            // [K]

    int    p_inlet_bc = 0;                  // 0 Dirichlet, 1 Neumann
    double p_inlet_value = 0.0;             // [Pa]

    int    p_outlet_bc = 0;                 // 0 Dirichlet, 1 Neumann
    double p_outlet_value = 0.0;            // [Pa]

    double u_initial = 0.0;                 // [m/s]
    double p_initial = 0.0;                 // [Pa]
    double T_initial = 0.0;                 // [K]

    int number_output = 0;            // Number of outputs [-]

    std::string velocity_file = "";
    std::string pressure_file = "";
    std::string temperature_file = "";
};

Input readInput(const std::string& filename) {

    std::ifstream file(filename);
    std::string line, key, eq, value;

    std::unordered_map<std::string, std::string> dict;

    while (std::getline(file, line)) {

        // Removes comments
        auto comment = line.find('#');
        if (comment != std::string::npos)
            line = line.substr(0, comment);

		// Removes empty lines
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;

        // Finds '='
        auto pos = line.find('=');
        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // Trim key
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);

        // Trim value
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        dict[key] = value;
    }

    Input in;

    in.N = std::stoi(dict["N"]);
    in.L = std::stod(dict["L"]);

    in.dt_user = std::stod(dict["dt_user"]);
    in.simulation_time = std::stod(dict["simulation_time"]);

    in.piso_outer_iter = std::stoi(dict["piso_outer_iter"]);
    in.piso_inner_iter = std::stoi(dict["piso_inner_iter"]);
    in.piso_outer_tol = std::stod(dict["piso_outer_tol"]);
    in.piso_inner_tol = std::stod(dict["piso_inner_tol"]);
    in.rhie_chow_on_off_l = std::stoi(dict["rhie_chow"]);

    in.rho = std::stod(dict["rho"]);
    in.mu = std::stod(dict["mu"]);
    in.k = std::stod(dict["k"]);
    in.cp = std::stod(dict["cp"]);

    in.S_m_cell = std::stod(dict["S_m_cell"]);
    in.S_h_cell = std::stod(dict["S_h_cell"]);

    in.z_evap_start = std::stod(dict["z_evap_start"]);
    in.z_evap_end = std::stod(dict["z_evap_end"]);
    in.z_cond_start = std::stod(dict["z_cond_start"]);
    in.z_cond_end = std::stod(dict["z_cond_end"]);

    in.u_inlet_bc = std::stoi(dict["u_inlet_bc"]);
    in.u_inlet_value = std::stod(dict["u_inlet_value"]);

    in.u_outlet_bc = std::stoi(dict["u_outlet_bc"]);
    in.u_outlet_value = std::stod(dict["u_outlet_value"]);

    in.T_inlet_bc = std::stoi(dict["T_inlet_bc"]);
    in.T_inlet_value = std::stod(dict["T_inlet_value"]);

    in.T_outlet_bc = std::stoi(dict["T_outlet_bc"]);
    in.T_outlet_value = std::stod(dict["T_outlet_value"]);

    in.p_inlet_bc = std::stoi(dict["p_inlet_bc"]);
    in.p_inlet_value = std::stod(dict["p_inlet_value"]);

    in.p_outlet_bc = std::stoi(dict["p_outlet_bc"]);
    in.p_outlet_value = std::stod(dict["p_outlet_value"]);

	in.u_initial = std::stod(dict["u_initial"]);
	in.T_initial = std::stod(dict["T_initial"]);
    in.p_initial = std::stod(dict["p_initial"]);

    in.number_output = std::stoi(dict["number_output"]);
    in.velocity_file = dict["velocity_file"];
    in.pressure_file = dict["pressure_file"];
    in.temperature_file = dict["temperature_file"];

    return in;
}

#pragma endregion

// =======================================================================
//                                MAIN
// =======================================================================

int main() {

    std::string inputFile = chooseInputFile("input");
    std::cout << "Using input file: " << inputFile << std::endl;

    Input in = readInput(inputFile);

	const int    N = in.N;                                              // Number of cells [-]
    const double L = in.L;                                              // Length of the domain [m]
	const double dz = L / N;                                            // Cell size [m]

	double dt_user = in.dt_user;                                        // User-defined time step [s]
	const double simulation_time = in.simulation_time;                  // Total simulation time [s]
	const int time_steps = static_cast<int>(simulation_time / dt_user); // Number of time steps [-]
        
	const int number_output = in.number_output;                         // Number of outputs [-]
	const int print_every = time_steps / number_output;                 // Print output every n time steps [-]

	double time_total = 0.0;                                            // Total simulation time [s]
    double dt = dt_user;                                                // Time step [s]

	const int tot_outer_l = in.piso_outer_iter;                         // PISO outer iterations [-]
	const int tot_inner_l = in.piso_inner_iter;                         // PISO inner iterations [-]
	const double outer_tol_l = in.piso_outer_tol;                       // PISO outer tolerance [-]
	const double inner_tol_l = in.piso_inner_tol;                       // PISO inner tolerance [-]
	const bool rhie_chow_on_off_l = in.rhie_chow_on_off_l;              // Rhie–Chow interpolation on/off (1/0) [-]

	const double rho_l = in.rho;                                        // Density [kg/m3]
	const double mu = in.mu;                                            // Dynamic viscosity [kg/(m s)]
	const double k = in.k;                                              // Thermal diffusivity [m2/s]
    const double cp = in.cp;                                            // Specific heat capacity at constant pressure [J/kgK]

	std::vector<double> u_l(N, in.u_initial);                           // Velocity field [m/s]
	std::vector<double> T_l(N, in.T_initial);                           // Temperature field [K]
	std::vector<double> p_l(N, in.p_initial);                           // Pressure field [Pa]

	std::vector<double> u_l_old = u_l;                                  // Previous time step velocity [m/s]
	std::vector<double> T_l_old = T_l;                                  // Previous time step temperature [K]
	std::vector<double> p_l_old = p_l;                                  // Previous time step pressure [Pa]

	std::vector<double> T_l_iter(N, 0.0);                               // Temperature field for Picard iteration [K]

	std::vector<double> p_prime_l(N, 0.0);                              // Pressure correction [Pa]
	std::vector<double> p_storage_l(N + 2);                             // Padded pressure storage for Rhie–Chow [Pa]
	double* p_padded_l = &p_storage_l[1];                               // Pointer to the real nodes of the padded pressure storage [Pa]

	// p_storage_l initialization
    for (int i = 0; i < N; ++i)
        p_storage_l[i + 1] = p_l[i];

    p_storage_l[0] = p_l[0];
    p_storage_l[N + 1] = p_l[N - 1];

	std::vector<double> u_prev(N, 0.0);                                 // Previous iteration velocity for convergence check [m/s]
	std::vector<double> p_prev(N, 0.0);                                 // Previous iteration pressure for convergence check [Pa]
	std::vector<double> T_prev(N, 0.0);                                 // Previous iteration temperature for convergence check [K]

	std::vector<double> S_m(N, 0.0);                                    // Volumetric mass source [kg/(m3 s)]
	std::vector<double> S_h(N, 0.0);                                    // Volumetric heat source [W/m3]

	// Source vectors definition
    for (int i = 0; i < N; ++i) {

        const double z = (i + 0.5) * dz;

        if (z >= in.z_evap_start && z <= in.z_evap_end) {
            S_m[i] = in.S_m_cell;
            S_h[i] = in.S_h_cell;
        }
        else if (z >= in.z_cond_start && z <= in.z_cond_end) {
            S_m[i] = -in.S_m_cell;
            S_h[i] = -in.S_h_cell;
        }
    }

	const double u_inlet_value = in.u_inlet_value;          // Inlet velocity [m/s]
	const double u_outlet_value = in.u_outlet_value;        // Outlet velocity [m/s]
	const bool u_inlet_bc = in.u_inlet_bc;                  // Inlet velocity BC type (Dirichlet: 0.0, Neumann: 1.0) [-]
	const bool u_outlet_bc = in.u_outlet_bc;                // Outlet velocity BC type (Dirichlet: 0.0, Neumann: 1.0) [-]

	const double T_inlet_value = in.T_inlet_value;          // Inlet temperature [K]
	const double T_outlet_value = in.T_outlet_value;        // Outlet temperature [K]
	const bool T_inlet_bc = in.T_inlet_bc;                  // Inlet temperature BC type (Dirichlet: 0.0, Neumann: 1.0) [-]
	const bool T_outlet_bc = in.T_outlet_bc;                // Outlet temperature BC type (Dirichlet: 0.0, Neumann: 1.0) [-]

	const double p_inlet_value = in.p_inlet_value;          // Inlet pressure [Pa]
	const double p_outlet_value = in.p_outlet_value;        // Outlet pressure [Pa]
	const bool p_inlet_bc = in.p_inlet_bc;                  // Inlet pressure BC type (Dirichlet: 0.0, Neumann: 1.0) [-]
	const bool p_outlet_bc = in.p_outlet_bc;                // Outlet pressure BC type (Dirichlet: 0.0, Neumann: 1.0) [-]

	const double z_evap_start = in.z_evap_start;                        // Evaporation zone start and end [m]
	const double z_evap_end = in.z_evap_end;                            // Evaporation zone start and end [m]

	const double z_cond_start = in.z_cond_start;                        // Evaporation zone start and end [m]
	const double z_cond_end = in.z_cond_end;                            // Condensation zone start and end [m]

	const double L_evap = z_evap_end - z_evap_start;                    // Length of the evaporation zone [m]
	const double L_cond = z_cond_end - z_cond_start;                    // Length of the condensation zone [m]

    std::vector<double> aLU(N, 0.0);                                    // Lower tridiagonal coefficient for velocity
    std::vector<double> bLU(N, rho_l * dz / dt_user + 2 * mu / dz);     // Central tridiagonal coefficient for velocity
    std::vector<double> cLU(N, 0.0);                                    // Upper tridiagonal coefficient for velocity
    std::vector<double> dLU(N, 0.0);                                    // Known vector coefficient for velocity

	std::vector<double> aLP(N, 0.0);                                    // Lower tridiagonal coefficient for pressure
	std::vector<double> bLP(N, 0.0);                                    // Central tridiagonal coefficient for pressure
	std::vector<double> cLP(N, 0.0);                                    // Upper tridiagonal coefficient for pressure
	std::vector<double> dLP(N, 0.0);                                    // Known vector coefficient for pressure

	std::vector<double> aLT(N, 0.0);                                    // Lower tridiagonal coefficient for temperature
	std::vector<double> bLT(N, 0.0);                                    // Central tridiagonal coefficient for temperature
	std::vector<double> cLT(N, 0.0);                                    // Upper tridiagonal coefficient for temperature
	std::vector<double> dLT(N, 0.0);                                    // Known vector coefficient for temperature

    const double K = 1e-8;
	const double CF = 1e-4;

    fs::path inputPath(inputFile);
    std::string caseName = inputPath.filename().string();
    fs::path outputDir = fs::path("output") / caseName;
    fs::create_directories(outputDir);

    std::ofstream v_out(outputDir / in.velocity_file);              // Velocity output file
    std::ofstream p_out(outputDir / in.pressure_file);              // Pressure output file
    std::ofstream T_out(outputDir / in.temperature_file);           // Temperature output file

    // Convergence metrics
	double continuity_residual = 1.0;
    double momentum_residual = 1.0;
	double energy_residual = 1.0;

    double u_error_l = 1.0;
    int outer_l = 0;

    double p_error_l = 1.0;
    double rho_error_l = 1.0;
    int inner_l = 0;

    double start = omp_get_wtime();

	// Time-stepping loop
    for (int n = 0; n <= time_steps; ++n) {

        // Saving old variables
        u_l_old = u_l;
        T_l_old = T_l;
        p_l_old = p_l;

        u_error_l = 1.0;
        outer_l = 0;

        momentum_residual = 1.0;
        energy_residual = 1.0;

        while (outer_l < tot_outer_l && (momentum_residual > outer_tol_l || energy_residual > outer_tol_l)) {

            // ===========================================================
            // MOMENTUM PREDICTOR
            // ===========================================================

            for (int i = 1; i < N - 1; ++i) {

                const double D_l = mu / dz;       // [kg/(m2s)]
                const double D_r = mu / dz;       // [kg/(m2s)]

                const double avgInvbLU_L = 0.5 * (1.0 / bLU[i - 1] + 1.0 / bLU[i]); // [m2s/kg]
                const double avgInvbLU_R = 0.5 * (1.0 / bLU[i + 1] + 1.0 / bLU[i]); // [m2s/kg]

                // Rhie–Chow corrections for face velocities
                const double rc_l = -avgInvbLU_L / 4.0 *
                    (p_padded_l[i - 2] - 3.0 * p_padded_l[i - 1] + 3.0 * p_padded_l[i] - p_padded_l[i + 1]); // [m/s]
                const double rc_r = -avgInvbLU_R / 4.0 *
                    (p_padded_l[i - 1] - 3.0 * p_padded_l[i] + 3.0 * p_padded_l[i + 1] - p_padded_l[i + 2]); // [m/s]

                // Face velocities (avg + RC)
                const double u_l_face = 0.5 * (u_l[i - 1] + u_l[i]) + rhie_chow_on_off_l * rc_l;    // [m/s]
                const double u_r_face = 0.5 * (u_l[i] + u_l[i + 1]) + rhie_chow_on_off_l * rc_r;    // [m/s]

                const double F_l = rho_l * u_l_face; // [kg/(m2s)]
                const double F_r = rho_l * u_r_face; // [kg/(m2s)]

                const double f = +mu / K * dz + rho_l * CF * std::abs(u_l[i]) / std::sqrt(K) * dz;

                aLU[i] =
                    - std::max(F_l, 0.0)
                    - D_l;                                  // [kg/(m2s)]
                cLU[i] =
                    - std::max(-F_r, 0.0)
                    - D_r;                                  // [kg/(m2s)]
                bLU[i] =
                    + std::max(F_r, 0.0)
                    + std::max(-F_l, 0.0)
                    + rho_l * dz / dt
                    + D_l + D_r
                    // + f
                    ;                            // [kg/(m2s)]
                dLU[i] =
                    - 0.5 * (p_l[i + 1] - p_l[i - 1])
                    + rho_l * u_l_old[i] * dz / dt;         // [kg/(ms2)]
            }

            /// Diffusion coefficients for the first and last node to define BCs
            const double D_first = mu / dz;
            const double D_last = mu / dz;

            /// Velocity BCs needed variables for the first node
            const double u_r_face_first = 0.5 * (u_l[1]);
            const double F_r_first = rho_l * u_r_face_first;

            /// Velocity BCs needed variables for the last node
            const double u_l_face_last = 0.5 * (u_l[N - 2]);
            const double F_l_last = rho_l * u_l_face_last;

			if (u_inlet_bc == 0) {                               // Dirichlet BC
                aLU[0] = 0.0;
                bLU[0] = rho_l * dz / dt + 2 * D_first + F_r_first;
                cLU[0] = 0.0;
                dLU[0] = bLU[0] * u_inlet_value;
			}
			else if (u_inlet_bc == 1) {                          // Neumann BC
                aLU[0] = 0.0;
                bLU[0] = + (rho_l * dz / dt + 2 * D_first + F_r_first);
                cLU[0] = - (rho_l * dz / dt + 2 * D_first + F_r_first);
                dLU[0] = 0.0;
			}

			if (u_outlet_bc == 0) {                              // Dirichlet BC
                aLU[N - 1] = 0.0;
                bLU[N - 1] = + (rho_l * dz / dt + 2 * D_last - F_l_last);
                cLU[N - 1] = 0.0;
                dLU[N - 1] = bLU[N - 1] * u_outlet_value;
            }
			else if (u_outlet_bc == 1) {                          // Neumann BC
                aLU[N - 1] = - (rho_l * dz / dt + 2 * D_last - F_l_last);
                bLU[N - 1] = + (rho_l * dz / dt + 2 * D_last - F_l_last);
                cLU[N - 1] = 0.0;
                dLU[N - 1] = 0.0;
            }

            u_l = tdma::solve(aLU, bLU, cLU, dLU);

            // ===============================================================
            // TEMPERATURE SOLVER
            // ===============================================================

            // Energy equation for T (implicit), upwind convection, central diffusion
            for (int i = 1; i < N - 1; i++) {

                const double D_l = k / (rho_l * cp * dz);      /// [W/(m2 K)]
                const double D_r = k / (rho_l * cp * dz);      /// [W/(m2 K)]

                const double avgInvbLU_L = 0.5 * (1.0 / bLU[i - 1] + 1.0 / bLU[i]);     // [m2s/kg]
                const double avgInvbLU_R = 0.5 * (1.0 / bLU[i + 1] + 1.0 / bLU[i]);     // [m2s/kg]

                const double rc_l = -avgInvbLU_L / 4.0 *
                    (p_padded_l[i - 2] - 3.0 * p_padded_l[i - 1] + 3.0 * p_padded_l[i] - p_padded_l[i + 1]);    // [m/s]
                const double rc_r = -avgInvbLU_R / 4.0 *
                    (p_padded_l[i - 1] - 3.0 * p_padded_l[i] + 3.0 * p_padded_l[i + 1] - p_padded_l[i + 2]);    // [m/s]

                const double u_l_face = 0.5 * (u_l[i - 1] + u_l[i]) + rhie_chow_on_off_l * rc_l;         // [m/s]
                const double u_r_face = 0.5 * (u_l[i] + u_l[i + 1]) + rhie_chow_on_off_l * rc_r;         // [m/s]

                aLT[i] =
                    - D_l
                    - std::max(u_l_face, 0.0)
                    ;              /// [W/(m2 K)]

                cLT[i] =
                    - D_r
                    - std::max(-u_r_face, 0.0)
                    ;            /// [W/(m2 K)]

                bLT[i] =
                    + std::max(u_r_face, 0.0)
                    + std::max(-u_l_face, 0.0)
                    + D_l + D_r
                    + dz / dt
                    ;                              /// [W/(m2 K)]

                dLT[i] =
                    + dz / dt * T_l_old[i]
                    + S_h[i] * dz
					+ S_m[i] * T_l_old[i] * dz / rho_l
                    ;                          /// [W/m2]
            }

            // BCs on temperature
            if (T_inlet_bc == 0) {                          // Dirichlet BC

                aLT[0] = 0.0;
                bLT[0] = 1.0;
                cLT[0] = 0.0;
                dLT[0] = T_inlet_value;
            }
            else if (T_inlet_bc == 1) {                     // Neumann BC

                aLT[0] = 0.0;
                bLT[0] = 1.0;
                cLT[0] = -1.0;
                dLT[0] = 0.0;
            }

            if (T_outlet_bc == 0) {                         // Dirichlet BC

                aLT[N - 1] = 0.0;
                bLT[N - 1] = 1.0;
                cLT[N - 1] = 0.0;
                dLT[N - 1] = T_outlet_value;
            }
            else if (T_outlet_bc == 1) {                    // Neumann BC

                aLT[N - 1] = -1.0;
                bLT[N - 1] = 1.0;
                cLT[N - 1] = 0.0;
                dLT[N - 1] = 0.0;
            }

            T_prev = T_l;
            T_l = tdma::solve(aLT, bLT, cLT, dLT);

            rho_error_l = 1.0;
            p_error_l = 1.0;
            inner_l = 0;

            continuity_residual = 1.0;

            while (inner_l < tot_inner_l && continuity_residual > inner_tol_l) {

                // -------------------------------------------------------
                // CONTINUITY SATISFACTOR: assemble pressure correction
                // -------------------------------------------------------

                for (int i = 1; i < N - 1; ++i) {

                    const double avgInvbLU_L = 0.5 * (1.0 / bLU[i - 1] + 1.0 / bLU[i]);     // [m2s/kg]
                    const double avgInvbLU_R = 0.5 * (1.0 / bLU[i + 1] + 1.0 / bLU[i]);     // [m2s/kg]

                    const double rc_l = -avgInvbLU_L / 4.0 *
                        (p_padded_l[i - 2] - 3.0 * p_padded_l[i - 1] + 3.0 * p_padded_l[i] - p_padded_l[i + 1]);    // [m/s]
                    const double rc_r = -avgInvbLU_R / 4.0 *
                        (p_padded_l[i - 1] - 3.0 * p_padded_l[i] + 3.0 * p_padded_l[i + 1] - p_padded_l[i + 2]);    // [m/s]

                    const double u_l_star = 0.5 * (u_l[i - 1] + u_l[i]) + rhie_chow_on_off_l * rc_l;    // [m/s]
                    const double u_r_star = 0.5 * (u_l[i] + u_l[i + 1]) + rhie_chow_on_off_l * rc_r;    // [m/s]

                    const double phi_l = rho_l * u_l_star;   // [kg/(m2s)]
                    const double phi_r = rho_l * u_r_star;   // [kg/(m2s)]

                    const double mass_imbalance = (phi_r - phi_l);  // [kg/(m2s)]

                    const double mass_flux = S_m[i] * dz;         // [kg/(m2s)]

                    const double E_l = rho_l * avgInvbLU_L / dz; // [s/m]
                    const double E_r = rho_l * avgInvbLU_R / dz; // [s/m]

                    aLP[i] =
                        - E_l
                        ;               /// [s/m]

                    cLP[i] =
                        - E_r
                        ;               /// [s/m]

                    bLP[i] =
                        + E_l 
                        + E_r
                        ;               /// [s/m]

                    dLP[i] = + mass_flux - mass_imbalance;  /// [kg/(m2s)]
                }

                // BCs on p_prime
                if (p_inlet_bc == 0) {                               // Dirichlet BC
                    aLP[0] = 0.0;
                    bLP[0] = 1.0;
                    cLP[0] = 0.0;
                    dLP[0] = 0.0;
                }
                else if (p_inlet_bc == 1) {                          // Neumann BC
                    aLP[0] = 0.0;
                    bLP[0] = 1.0;
                    cLP[0] = -1.0;
                    dLP[0] = 0.0;
                }

                if (p_outlet_bc == 0) {                              // Dirichlet BC
                    aLP[N - 1] = 0.0;
                    bLP[N - 1] = 1.0;
                    cLP[N - 1] = 0.0;
                    dLP[N - 1] = 0.0;
                }
                else if (p_outlet_bc == 1) {                          // Neumann BC
                    aLP[N - 1] = -1.0;
                    bLP[N - 1] = 1.0;
                    cLP[N - 1] = 0.0;
                    dLP[N - 1] = 0.0;
                }

                p_prime_l = tdma::solve(aLP, bLP, cLP, dLP);

                // -------------------------------------------------------
                // PRESSURE CORRECTOR
                // -------------------------------------------------------

                p_error_l = 0.0;

                for (int i = 0; i < N; ++i) {

                    p_prev[i] = p_l[i];
                    p_l[i] += p_prime_l[i];

                    p_storage_l[i + 1] = p_l[i];
                    p_error_l = std::max(p_error_l, std::fabs(p_l[i] - p_prev[i]));
                }

                // BCs on pressure
                if (p_inlet_bc == 0) {                              // Dirichlet BC

					p_l[0] = p_inlet_value;
                    p_storage_l[N + 1] = p_inlet_value;
                }
                else if (p_inlet_bc == 1) {                         // Neumann BC

					p_l[0] = p_l[1];
                    p_storage_l[0] = p_storage_l[1];
                }

                if (p_outlet_bc == 0) {                              // Dirichlet BC

					p_l[N - 1] = p_outlet_value;
                    p_storage_l[N + 1] = p_outlet_value;
                }
                else if (p_outlet_bc == 1) {                         // Neumann BC

					p_l[N - 1] = p_l[N - 2];
                    p_storage_l[N + 1] = p_storage_l[N];
                }

                // -------------------------------------------------------
                // VELOCITY CORRECTOR
                // -------------------------------------------------------

                u_error_l = 0.0;

                for (int i = 1; i < N - 1; ++i) {
                    u_prev[i] = u_l[i];
                    u_l[i] -= (p_prime_l[i + 1] - p_prime_l[i - 1]) / (2.0 * bLU[i]);
                    u_error_l = std::max(u_error_l, std::fabs(u_l[i] - u_prev[i]));
                }

                // -------------------------------------------------------
				// CONTINUITY RESIDUAL CALCULATION
                // -------------------------------------------------------

                double phi_ref = 0.0;
                double Sm_ref = 0.0;

                for (int i = 1; i < N - 1; ++i) {

                    const double u_l_face = 0.5 * (u_l[i - 1] + u_l[i]);
                    const double u_r_face = 0.5 * (u_l[i] + u_l[i + 1]);

                    phi_ref = std::max(phi_ref, rho_l * std::abs(u_l_face));
                    phi_ref = std::max(phi_ref, rho_l * std::abs(u_r_face));

                    Sm_ref = std::max(Sm_ref, std::abs(S_m[i] * dz));
                }

                const double cont_ref = std::max({ phi_ref, Sm_ref, 1e-30 });

                continuity_residual = 0.0;

                for (int i = 1; i < N - 1; ++i) {

                    const double avgInvbLU_L = 0.5 * (1.0 / bLU[i - 1] + 1.0 / bLU[i]);     // [m2s/kg]
                    const double avgInvbLU_R = 0.5 * (1.0 / bLU[i + 1] + 1.0 / bLU[i]);     // [m2s/kg]

                    const double rc_l = -avgInvbLU_L / 4.0 *
                        (p_padded_l[i - 2] - 3.0 * p_padded_l[i - 1] + 3.0 * p_padded_l[i] - p_padded_l[i + 1]);    // [m/s]
                    const double rc_r = -avgInvbLU_R / 4.0 *
                        (p_padded_l[i - 1] - 3.0 * p_padded_l[i] + 3.0 * p_padded_l[i + 1] - p_padded_l[i + 2]);    // [m/s]

                    const double u_l_star = 0.5 * (u_l[i - 1] + u_l[i]) + rhie_chow_on_off_l * rc_l;    // [m/s]
                    const double u_r_star = 0.5 * (u_l[i] + u_l[i + 1]) + rhie_chow_on_off_l * rc_r;    // [m/s]

                    const double phi_l = rho_l * u_l_star;          // [kg/(m2s)]
                    const double phi_r = rho_l * u_r_star;          // [kg/(m2s)]

                    const double mass_imbalance = (phi_r - phi_l);  // [kg/(m2s)]

                    const double mass_flux = S_m[i] * dz;           // [kg/(m2s)]

                    continuity_residual =
                        std::max(continuity_residual,
                            std::abs(mass_flux - mass_imbalance) / cont_ref);
                }

                inner_l++;
            }

            // -------------------------------------------------------
            // MOMENTUM RESIDUAL CALCULATION
            // -------------------------------------------------------

            double U_ref = 0.0;
			double F_ref = 0.0;

            for (int i = 0; i < N; ++i) {

                U_ref = std::max(U_ref, std::abs(u_l[i]));
            }

            for (int i = 0; i < N; ++i) {
                const double F_inertia = rho_l * U_ref * U_ref;
                const double F_unsteady = rho_l * U_ref * dz / dt;
                const double F_viscous = mu * U_ref / dz;

                F_ref = std::max({ F_inertia, F_unsteady, F_viscous, 1e-30 });

            }

            momentum_residual = 0.0;

            for (int i = 1; i < N - 1; ++i) {

                const double avgInvbLU_L = 0.5 * (1.0 / bLU[i - 1] + 1.0 / bLU[i]);     // [m2s/kg]
                const double avgInvbLU_R = 0.5 * (1.0 / bLU[i + 1] + 1.0 / bLU[i]);     // [m2s/kg]

                const double rc_l = -avgInvbLU_L / 4.0 *
                    (p_padded_l[i - 2] - 3.0 * p_padded_l[i - 1] + 3.0 * p_padded_l[i] - p_padded_l[i + 1]);    // [m/s]
                const double rc_r = -avgInvbLU_R / 4.0 *
                    (p_padded_l[i - 1] - 3.0 * p_padded_l[i] + 3.0 * p_padded_l[i + 1] - p_padded_l[i + 2]);    // [m/s]

                const double D_l = mu / dz;
                const double D_r = mu / dz;

                const double u_l_face =
                    0.5 * (u_l[i - 1] + u_l[i]) + rc_l * rhie_chow_on_off_l;
                const double u_r_face =
                    0.5 * (u_l[i] + u_l[i + 1]) + rc_r * rhie_chow_on_off_l;

                const double F_l = rho_l * u_l_face;
                const double F_r = rho_l * u_r_face;

                const double accum =
                    rho_l * dz / dt * (u_l[i] - u_l_old[i]);

                const double conv =
                    F_r * u_r_face - F_l * u_l_face;

                const double diff =
                    D_r * (u_l[i + 1] - u_l[i])
                    - D_l * (u_l[i] - u_l[i - 1]);

                const double press =
                    0.5 * (p_l[i + 1] - p_l[i - 1]);

                const double R =
                    accum + conv - diff + press;

                momentum_residual =
                    std::max(momentum_residual, std::abs(R) / F_ref);
            }

            // -------------------------------
            // TEMPERATURE RESIDUAL
            // -------------------------------
            energy_residual = 0.0;

            for (int i = 0; i < N; ++i) {

                energy_residual = std::max(
                    energy_residual,
                    std::abs(T_l[i] - T_prev[i])
                );
            }

            outer_l++;
        }

        // ===============================================================
        // OUTPUT
        // ===============================================================

        if (n % print_every == 0) {
            for (int i = 0; i < N; ++i) {

                v_out << u_l[i] << ", ";
                p_out << p_l[i] << ", ";
                T_out << T_l[i] << ", ";
            }

            v_out << "\n";
            p_out << "\n";
            T_out << "\n";
        }
    }

    v_out.flush();
    p_out.flush();
    T_out.flush();

    v_out.close();
    p_out.close();
    T_out.close();

    double end = omp_get_wtime();
    printf("Execution time: %.6f s\n", end - start);

    return 0;
}