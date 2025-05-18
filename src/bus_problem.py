import numpy as np
from matplotlib import pyplot as plt

from helper_functions import convert_to_standard_form, initial_point_standard_heuristic
from mehrotra import mpc_solver
from long_step import pd_long_step_solver

# --- Utility function for variable indexing (ensure this is available) ---
def get_busing_var_idx(area_idx, school_idx, grade_idx, num_schools, num_grades):
    # area_idx: 0-5, school_idx: 0-2, grade_idx: 0-2 (6th,7th,8th)
    return area_idx * (num_schools * num_grades) + \
        school_idx * num_grades + \
        grade_idx

def get_symbolic_var_name(flat_idx, num_areas, num_schools, num_grades):
    k_idx = flat_idx % num_grades
    temp_val = flat_idx // num_grades
    j_idx = temp_val % num_schools
    i_idx = temp_val // num_schools
    grade_prefix_map = {0: 'x', 1: 's', 2: 'e'}
    prefix = grade_prefix_map.get(k_idx, f"vK{k_idx}")
    return f"{prefix}{i_idx+1}{j_idx+1}"

def setup_busing_lp():
    print("<<<< Using setup_busing_lp >>>>")
    num_areas = 6
    num_schools = 3
    num_grades = 3

    Students = np.array([
        [144, 171, 135], [222, 168, 210], [165, 176, 209],
        [98, 140, 112], [195, 170, 135], [153, 126, 171]
    ])

    Cost_data = np.array([
        [300, 0, 700],  # Area 1
        [np.nan, 400, 500],  # Area 2 (A2S1 is infeasible)
        [600, 300, 200],  # Area 3
        [200, 500, np.nan],  # Area 4 (A4S3 is infeasible, A4S1 IS $200)
        [0, np.nan, 400],  # Area 5 (A5S2 is infeasible)
        [500, 300, 0]  # Area 6
    ])

    infeasible_routes_list = [
        (1, 0), (4, 1), (3, 2)
    ]

    # NO school_capacities array needed

    M_dummy = 1.0e7

    num_total_potential_vars = num_areas * num_schools * num_grades
    c_input = np.zeros(num_total_potential_vars)
    var_is_decision_variable = np.ones(num_total_potential_vars, dtype=bool)

    for i in range(num_areas):
        for j in range(num_schools):
            is_infeasible_route = (i, j) in infeasible_routes_list
            for k_idx in range(num_grades):
                var_idx = get_busing_var_idx(i, j, k_idx, num_schools, num_grades)
                if is_infeasible_route:
                    c_input[var_idx] = 0
                    var_is_decision_variable[var_idx] = False
                else:
                    if np.isnan(Cost_data[i, j]):
                        print(f"ERROR: Cost is NaN for feasible route ({i},{j}). Var_idx {var_idx}.")
                        c_input[var_idx] = 0
                        var_is_decision_variable[var_idx] = False
                    else:
                        c_input[var_idx] = Cost_data[i, j]

    constraints_list = []
    count_supply, count_percentage, count_fixed_vars = 0, 0, 0

    # 1. Supply Constraints (18 equalities)
    for i_area in range(num_areas):
        for k_grade_idx in range(num_grades):
            coeffs = {}
            active_vars_in_sum = False
            for j_school in range(num_schools):
                var_idx = get_busing_var_idx(i_area, j_school, k_grade_idx, num_schools, num_grades)
                if var_is_decision_variable[var_idx]:
                    coeffs[var_idx] = 1.0
                    active_vars_in_sum = True
            if active_vars_in_sum or Students[i_area, k_grade_idx] == 0:
                constraints_list.append({'coeffs': coeffs, 'sense': 'E', 'rhs': Students[i_area, k_grade_idx],
                                         'type': f'Sup_A{i_area + 1}G{k_grade_idx}'})
                count_supply += 1
            elif Students[i_area, k_grade_idx] != 0 and not active_vars_in_sum:
                print(f"WARNING (No Capacity Setup): Infeasible Supply for Area {i_area + 1}, Grade_idx {k_grade_idx}.")

    # 2. School Capacity Constraints -- REMOVED --
    count_capacity = 0  # Explicitly zero

    # 3. Percentage Constraints (18 individual inequalities)
    grade_map_print = {0: "6th", 1: "7th", 2: "8th"}
    for j_school in range(num_schools):
        for k_target_grade_idx in range(num_grades):
            other_grade_indices = [idx for idx in range(num_grades) if idx != k_target_grade_idx]
            coeffs_upper, coeffs_lower = {}, {}
            school_is_active_for_perc = any(
                var_is_decision_variable[get_busing_var_idx(ia, j_school, kg, num_schools, num_grades)]
                for ia in range(num_areas) for kg in range(num_grades))
            if not school_is_active_for_perc: continue

            for i_area in range(num_areas):
                var_idx_target = get_busing_var_idx(i_area, j_school, k_target_grade_idx, num_schools, num_grades)
                if var_is_decision_variable[var_idx_target]:
                    coeffs_upper[var_idx_target] = coeffs_upper.get(var_idx_target, 0) + 0.64
                    coeffs_lower[var_idx_target] = coeffs_lower.get(var_idx_target, 0) + 0.70
                var_idx_other1 = get_busing_var_idx(i_area, j_school, other_grade_indices[0], num_schools, num_grades)
                if var_is_decision_variable[var_idx_other1]:
                    coeffs_upper[var_idx_other1] = coeffs_upper.get(var_idx_other1, 0) - 0.36
                    coeffs_lower[var_idx_other1] = coeffs_lower.get(var_idx_other1, 0) - 0.30
                var_idx_other2 = get_busing_var_idx(i_area, j_school, other_grade_indices[1], num_schools, num_grades)
                if var_is_decision_variable[var_idx_other2]:
                    coeffs_upper[var_idx_other2] = coeffs_upper.get(var_idx_other2, 0) - 0.36
                    coeffs_lower[var_idx_other2] = coeffs_lower.get(var_idx_other2, 0) - 0.30

            target_g_name = grade_map_print.get(k_target_grade_idx)
            if coeffs_upper:
                constraints_list.append({'coeffs': coeffs_upper, 'sense': 'L', 'rhs': 0.0,
                                         'type': f'PercUpp_S{j_school + 1}_T{target_g_name}'})
                count_percentage += 1
            if coeffs_lower:
                constraints_list.append({'coeffs': coeffs_lower, 'sense': 'G', 'rhs': 0.0,
                                         'type': f'PercLow_S{j_school + 1}_T{target_g_name}'})
                count_percentage += 1

    # 4. Add constraints to fix non-decision variables to 0 (9 constraints)
    for var_idx in range(num_total_potential_vars):
        if not var_is_decision_variable[var_idx]:
            constraints_list.append({'coeffs': {var_idx: 1.0}, 'sense': 'E', 'rhs': 0.0,
                                     'type': f'Fix_{get_symbolic_var_name(var_idx, num_areas, num_schools, num_grades)}'})
            count_fixed_vars += 1

    # --- PRINT CONSTRUCTED CONSTRAINTS (Optional, but useful for verification) ---
    # print("\n--- Constructed Constraints (Readable Format - NO CAPACITY) ---")
    # for i, constr_data in enumerate(constraints_list):
    #     # ... (same printing logic as before) ...
    # print("--- End of Constructed Constraints ---")

    num_total_constraints_generated = len(constraints_list)
    A_input = np.zeros((num_total_constraints_generated, num_total_potential_vars))
    b_input = np.zeros(num_total_constraints_generated)
    constraint_types_input = [''] * num_total_constraints_generated

    for r_idx, constr in enumerate(constraints_list):
        for var_idx_key, coeff_val in constr['coeffs'].items(): A_input[r_idx, var_idx_key] = coeff_val
        b_input[r_idx] = constr['rhs']
        constraint_types_input[r_idx] = constr['sense']

    print(f"\nSetup NO CAPACITY:")
    print(
        f"  num_vars_total_potential={num_total_potential_vars}, num_decision_vars={np.sum(var_is_decision_variable)}")
    print(f"  num_constraints_A_input={A_input.shape[0]} (Generated: {num_total_constraints_generated})")
    print(
        f"  Counts: Supply={count_supply}, Capacity={count_capacity}, Percentage={count_percentage}, FixedToZero={count_fixed_vars}")

    grade_list_for_naming = [6, 7, 8]
    return c_input, A_input, b_input, constraint_types_input, num_total_potential_vars, \
        (num_areas, num_schools, num_grades, grade_list_for_naming), M_dummy

def generate_mps_file(filename, c_input, A_input, b_input, constraint_types_input, problem_dims, problem_name="BUSINGLP"):
    """
    Generates an MPS file, more flexible with constraint naming.

    Args:
        filename (str): The name of the MPS file to create.
        c_input (np.array): Cost vector (num_orig_vars,).
        A_input (np.array): Constraint matrix (num_constraints, num_orig_vars).
        b_input (np.array): RHS vector (num_constraints,).
        constraint_types_input (list): List of strings ('E', 'L', 'G') for constraint types.
        problem_dims (tuple): Contains (num_areas, num_schools, num_grades, grade_list_actual).
                               grade_list_actual might be empty if using PDF's x,s,e var names implicitly.
        problem_name (str): Name for the problem in the MPS file.
    """
    num_areas, num_schools, num_grades, grade_list_actual_from_dims = problem_dims
    num_total_constraints = A_input.shape[0]
    num_total_potential_vars = A_input.shape[1]  # Should match c_input.size

    # Fallback if grade_list_actual_from_dims is empty (e.g. from PDF accurate setup)
    # This is important for get_var_mps_name
    if not grade_list_actual_from_dims and num_grades > 0:
        # Assuming grades are 0, 1, 2 if not specified (e.g. 6th, 7th, 8th)
        # Or derive based on PDF's x,s,e if that was the intent
        # For simple naming, let's just use 0,1,2 if list is empty
        grade_values_for_naming = list(range(num_grades))
        print(
            "Warning (MPS Gen): grade_list_actual was empty in problem_dims. Using 0-indexed grade values for var naming.")
    elif not grade_list_actual_from_dims and num_grades == 0:
        grade_values_for_naming = [0]  # Dummy if no grades
        print("Warning (MPS Gen): num_grades is 0. Using a dummy grade value for var naming.")
    else:
        grade_values_for_naming = grade_list_actual_from_dims

    def get_var_mps_name(area_idx, school_idx, grade_k_val_for_name):
        # Tries to use actual grade value if available, otherwise 0-indexed k
        # Ensure grade_val is simple for MPS name
        g_char = str(grade_k_val_for_name)
        # Example: XA1S1G6 or XA1S1K0 (if grade_val is 0-indexed k)
        return f"XA{area_idx + 1}S{school_idx + 1}K{g_char}"

    with open(filename, 'w') as f:
        f.write(f"NAME          {problem_name}\n")
        f.write("ROWS\n")
        f.write(" N  OBJ\n")

        row_names_mps = []
        # Generic row naming based on index if structure is not the expected 36/39
        # This part needs to align with how your setup_... function orders constraints
        # For the 51-constraint version from setup_busing_lp_PDF_accurate_54vars:
        # 1. Supply (18 E, based on your print for PDF accurate setup)
        # 2. Capacity (3 L)
        # 3. Balance/Percentage (36: 18 L, 18 G)
        # 4. FixedToZero (e.g., 12 E) - but your print showed 51 total before these.
        # Your print: (Supply: 15, Capacity: 3, Balance: 18, FixedToZero: 12) -> Total 48
        # This implies your `constraint_list` in `setup_busing_lp_PDF_accurate_54vars`
        # might have a slightly different count/order than what A_input.shape[0] becomes.
        # Let's use generic names if the count is not 36 or 39 (from your Big M versions)
        # or 48/51 (from PDF accurate).

        # To make this robust, the setup function should ideally return symbolic names or types
        # for each constraint row. For now, let's make a more generic naming based on order.

        # For the 51 constraint version (18 Supply + 3 Cap + 18 Bal_L + 18 Bal_G + (0 if FixedToZero handled differently))
        # Or (18 Supply_E + 3 Cap_L + 18 Bal_L/G + 12 Fixed_E) = 51

        # Simplest robust naming:
        for r_idx in range(num_total_constraints):
            row_name = f"R{r_idx + 1:03}"  # e.g., R001, R002 ... R051
            row_names_mps.append(row_name)
            f.write(f" {constraint_types_input[r_idx]}  {row_name}\n")

        f.write("COLUMNS\n")
        # Loop through all *potential* 54 variables for indexing consistency
        var_idx_counter = 0
        for area_idx in range(num_areas):
            for school_idx in range(num_schools):
                for k_actual_idx in range(num_grades):  # k_actual_idx is 0,1,2
                    # Use grade_values_for_naming for consistent naming
                    grade_val_for_name = grade_values_for_naming[k_actual_idx]

                    var_mps_name = get_var_mps_name(area_idx, school_idx, grade_val_for_name)
                    # var_flat_idx = get_busing_var_idx(area_idx, school_idx, k_actual_idx, num_schools, num_grades)
                    # The var_idx_counter IS the flat_idx if looping consistently
                    var_flat_idx = var_idx_counter

                    cost_val = c_input[var_flat_idx]
                    # Only include cost if non-zero OR if it's part of the objective for a fixed-to-zero var (unlikely but possible)
                    # Standard MPS often omits zero costs, but explicit is fine.
                    # Let's include it if cost_val !=0 OR if it's a fixed var (cost should be 0 then anyway)
                    if abs(cost_val) > 1e-9 or (
                            A_input[:, var_flat_idx] != 0).any():  # If variable has cost or appears in constraints
                        f.write(f"    {var_mps_name:<8}  OBJ       {cost_val:>12.5f}\n")

                    for r_idx in range(num_total_constraints):
                        coeff = A_input[r_idx, var_flat_idx]
                        if abs(coeff) > 1e-9:  # Only write non-zero coefficients
                            f.write(f"    {var_mps_name:<8}  {row_names_mps[r_idx]:<8}{coeff:>12.5f}\n")
                    var_idx_counter += 1

        if var_idx_counter != num_total_potential_vars:
            print(
                f"Error (MPS Gen): Variable counter {var_idx_counter} does not match num_total_potential_vars {num_total_potential_vars}")

        f.write("RHS\n")
        rhs_name = "RHS1"
        for r_idx in range(num_total_constraints):
            rhs_val = b_input[r_idx]
            if abs(rhs_val) > 1e-9:  # Only list non-zero RHS values
                f.write(f"    {rhs_name:<8}  {row_names_mps[r_idx]:<8}{rhs_val:>12.5f}\n")

        # BOUNDS section if you are using setup_busing_lp_PDF_accurate_54vars
        # where some variables are fixed to 0 via equality constraints.
        # If those equality constraints are `1.0 * x_ijk = 0`, they effectively fix the var.
        # Alternatively, one could use the BOUNDS section.
        # If c_input for these vars is 0 and they are fixed by `x_ijk=0` constraint,
        # they don't strictly need explicit bounds here beyond default non-negativity.
        # If using Big M, then M is in c_input and no special bounds needed.

        # Example for explicit fixing using BOUNDS (if not done by equality constraints):
        # f.write("BOUNDS\n")
        # for var_idx in range(num_total_potential_vars):
        #     if not var_is_decision_variable[var_idx]: # Assuming you pass this info
        #         # Reconstruct area, school, grade from var_idx to get var_mps_name
        #         # This is more complex if not looping through i,j,k directly
        #         # var_mps_name = ...
        #         # f.write(f" FX BND1      {var_mps_name:<8}  0.0\n") # FX for fixed value
        #         pass

        f.write("ENDATA\n")
    print(f"MPS file '{filename}' generated successfully with {num_total_constraints} rows.")


def plot_selected_variable_trajectories(x_trajectory_pd, x_trajectory_mpc, selected_var_indices, optimal_x_values, problem_dims, pd_label="Path-Following", mpc_label="Mehrotra PC"):
    num_areas, num_schools, num_grades, _ = problem_dims
    x_traj_pd_np = np.array(x_trajectory_pd);
    x_traj_mpc_np = np.array(x_trajectory_mpc)
    num_plots = len(selected_var_indices)
    if num_plots == 0: print("No variables to plot."); return
    print(f"\nPlotting trajectories for {num_plots} selected variables...")
    cols = min(num_plots, 3);
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5 + 1), squeeze=False)  # Adjusted fig size
    axes_flat = axes.flatten()
    for i, var_idx in enumerate(selected_var_indices):
        ax = axes_flat[i]
        var_label = get_symbolic_var_name(var_idx, num_areas, num_schools, num_grades)
        optimal_val_for_var = optimal_x_values[var_idx] if optimal_x_values is not None and var_idx < len(
            optimal_x_values) else np.nan

        pd_iters = np.arange(x_traj_pd_np.shape[0])
        ax.plot(pd_iters, x_traj_pd_np[:, var_idx], label=pd_label, marker='o', markersize=3, alpha=0.7, linewidth=1)
        mpc_iters = np.arange(x_traj_mpc_np.shape[0])
        ax.plot(mpc_iters, x_traj_mpc_np[:, var_idx], label=mpc_label, marker='x', markersize=4, alpha=0.7,
                linestyle='--', linewidth=1)
        if not np.isnan(optimal_val_for_var):
            ax.axhline(y=optimal_val_for_var, color='r', linestyle=':', linewidth=1.2,
                       label=f'Optimal ({optimal_val_for_var:.1f})')
        ax.set_title(f'{var_label} (idx {var_idx})', fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8);
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(fontsize=7);
        ax.grid(True, linestyle=':', alpha=0.6)
    for j in range(num_plots, len(axes_flat)): fig.delaxes(axes_flat[j])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]);
    fig.suptitle("Convergence Trajectories of Selected Variables", fontsize=14);
    plt.show()


# --- 3D Plotting Function ---
def plot_3d_variable_trajectories(x_trajectory_pd, x_trajectory_mpc, var_indices_for_3d, optimal_x_values, problem_dims, pd_label="Path-Following", mpc_label="Mehrotra PC"):
    """
    Plots the 3D convergence trajectories of three selected variables for PD and MPC.
    """
    if len(var_indices_for_3d) != 3:
        print("Error: Exactly 3 variable indices must be provided for a 3D plot.")
        return

    num_areas, num_schools, num_grades, _ = problem_dims

    x_traj_pd_np = np.array(x_trajectory_pd)
    x_traj_mpc_np = np.array(x_trajectory_mpc)

    # Extract data for the three selected variables
    var_idx_x, var_idx_y, var_idx_z = var_indices_for_3d

    pd_x_data = x_traj_pd_np[:, var_idx_x]
    pd_y_data = x_traj_pd_np[:, var_idx_y]
    pd_z_data = x_traj_pd_np[:, var_idx_z]

    mpc_x_data = x_traj_mpc_np[:, var_idx_x]
    mpc_y_data = x_traj_mpc_np[:, var_idx_y]
    mpc_z_data = x_traj_mpc_np[:, var_idx_z]

    # Get symbolic names for axis labels
    label_x = get_symbolic_var_name(var_idx_x, num_areas, num_schools, num_grades)
    label_y = get_symbolic_var_name(var_idx_y, num_areas, num_schools, num_grades)
    label_z = get_symbolic_var_name(var_idx_z, num_areas, num_schools, num_grades)

    # Optimal point coordinates for the selected variables
    optimal_point = optimal_x_values[var_indices_for_3d]

    # Initial point coordinates (assuming both trajectories start from the same initial x)
    initial_point = x_traj_pd_np[0, var_indices_for_3d]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Path-Following trajectory
    ax.plot(pd_x_data, pd_y_data, pd_z_data, label=f'{pd_label} Trajectory', marker='o', markersize=3, alpha=0.7,
            linestyle='-')

    # Plot Mehrotra PC trajectory
    ax.plot(mpc_x_data, mpc_y_data, mpc_z_data, label=f'{mpc_label} Trajectory', marker='x', markersize=4, alpha=0.7,
            linestyle='--')

    # Mark Initial Point
    ax.scatter(initial_point[0], initial_point[1], initial_point[2], color='blue', s=100, label='Initial Point',
               depthshade=True, marker='^')

    # Mark Optimal Point (HiGHS solution)
    ax.scatter(optimal_point[0], optimal_point[1], optimal_point[2], color='red', s=150,
               label=f'Optimal Point ({optimal_point[0]:.1f}, {optimal_point[1]:.1f}, {optimal_point[2]:.1f})',
               depthshade=True, marker='*')

    ax.set_xlabel(f'Value of {label_x} (idx {var_idx_x})')
    ax.set_ylabel(f'Value of {label_y} (idx {var_idx_y})')
    ax.set_zlabel(f'Value of {label_z} (idx {var_idx_z})')
    ax.set_title('3D Convergence Trajectories of Selected Variables')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.show()

# --- Example of how to run and interpret results (assuming solvers are defined) ---
def main():
    c, A, b, con_types, n_vars, dims, M_cost = setup_busing_lp()
    c_std, A_std, b_std = convert_to_standard_form(c, A, b, con_types[:])
    # Print some info to verify
    #print(f"Busing LP Input for MPS: c shape {c.shape}, A shape {A.shape}, b shape {b.shape}")
    #print(f"Number of variables: {n_vars}")
    #print(f"Number of constraints: {A.shape[0]}")
    #print(f"Problem dimensions (areas, schools, grades, grade_list): {dims}")
    #print(f"M cost used for infeasible: {M_cost}")

    # 2. Generate the MPS file once
    mps_filename = "../busing_problem.mps"
    #generate_mps_file(mps_filename, c, A, b, con_types, dims)

    pd_params_for_pdf_accurate_run = {
    'maxN': 200, # Give it more room
    'eps_mu': 1e-7,
    'eps_feas': 1e-7,
    'sigma': 0.1,      # Try more aggressive centering
    'eta': 0.99,       # Slightly more conservative step
    'reg_param': 1e-7, # Start with a bit higher regularization
    'verbose': 2,
    'min_step_size_stall': 1e-8
    }

    x_pd_sol, obj_pd_solver, it_pd, time_pd, status_pd, x_trajectory_pd = pd_long_step_solver(
        A_std, b_std, c_std, pd_params_for_pdf_accurate_run, name="Busing_PD"
    )
    print(f"\n--- PD Long-Step Results (Busing) ---")
    #print(f"Status: {status_pd}")
    # print x
    #print(f"PD Solution (x): {x_pd_sol}")
    print(f"Objective: {obj_pd_solver}") # This is a minimization problem
    print(f"Iterations: {it_pd}, Time: {time_pd}s")
    print(f"Time per iteration: {time_pd / it_pd:.4f}s")
    '''if status_pd == "Optimal":
        print("\nPD Assignments (Area, School, Grade, Count):")
        for area_idx in range(num_areas_bus):
            for school_idx in range(num_schools_bus):
                for grade_idx, grade_val in enumerate(grade_list_bus):
                    v_idx = get_busing_var_idx(area_idx, school_idx, grade_idx, num_schools_bus, num_grades_bus)
                    if x_pd_sol[v_idx] > 0.01: # Print if non-negligible
                        print(f"  Area {area_idx+1} -> School {school_idx+1}, Grade {grade_val}: {x_pd_sol[v_idx]:.2f}")'''

    # do the same for mpc_solver
    # the return was return final_objective, x.flatten(), y.flatten(), s.flatten(), iterations, total_time, status

    mpc_params_current_run = {
        'maxN': 100,  # Or 200 if needed
        'eps_mu': 1e-7,
        'eps_feas': 1e-7,
        'theta': 0.9995,
        'verbose': 2,
        'min_step_size': 1e-8,
        'reg_param_pred': 1e-7,  # From your successful S2_Cap=1100 run
        'reg_param_corr': 1e-4,  # From your successful S2_Cap=1100 run
        'ds_magnitude_limit': 1e8,
        'dx_ds_clip_val_affine': 1e6,
        'target_debug_iteration': -1,  # Disable iter 9 specific debug
        'debug_print_enabled': False,
        'initial_point_verbose_debug': True  # See initial mu
    }
    obj_mpc_solver, x_mpc_sol, _, _, it_mpc, time_mpc, status_mpc, x_trajectory_mpc = mpc_solver(
        A_std, b_std, c_std,mpc_params_current_run, name="Busing_MPC"
    )
    print(f"\n--- Mehrotra Results (Busing) ---")
    #print(f"Status: {status_mpc}")
    # print x
    #print(f"MPC Solution (x): {x_mpc_sol}")
    print(f"Objective: {obj_mpc_solver}") # This is a minimization problem
    print(f"Iterations: {it_mpc}, Time: {time_mpc}s")
    print(f"Time per iteration: {time_mpc / it_mpc:.4f}s")

    print()
    import highspy
    h = highspy.Highs()
    h.readModel(mps_filename)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("output_flag", False)
    import time
    t0 = time.time()
    h.run()
    t1 = time.time() - t0
    info = h.getInfo()
    solution = h.getSolution()
    col_value = list(solution.col_value)
    highs_solution = np.array(col_value)
    #x, _, _ = initial_point_standard_heuristic(A_std, b_std, c_std)
    #print(f"Initial point: {x}")
    #print(f"Solution (col_value): {col_value}")
    #objective
    print(f"\n--- HiGHS Results (Busing) ---")
    print(f"Objective: {info.objective_function_value}")
    print(f"Iterations: {info.simplex_iteration_count}, Time: {t1}")
    # time per iter
    print(f"Time per iteration: {t1 / info.simplex_iteration_count:.4f}s")

    # 8. Plotting
    if status_pd == "Optimal" and status_mpc == "Optimal":
        # Variables: x12 (idx 3), x22 (idx 12), x41 (idx 27)
        selected_indices_to_plot = [3, 12, 27]

        # Ensure trajectories are not empty and optimal_x_from_highs_orig is valid
        if x_trajectory_pd and x_trajectory_mpc and highs_solution.size == n_vars:
            optimal_values_for_selected_vars = highs_solution[selected_indices_to_plot]

            plot_selected_variable_trajectories(
                x_trajectory_pd,
                x_trajectory_mpc,
                selected_indices_to_plot,
                highs_solution,  # Pass the full 54-var optimal vector
                dims,  # dims should be (num_areas, num_schools, num_grades, grade_list_for_naming)
                pd_label=f"Path-Following ({it_pd} iters)",
                mpc_label=f"Mehrotra PC ({it_mpc} iters)"
            )
            plot_3d_variable_trajectories(
                x_trajectory_pd,
                x_trajectory_mpc,
                selected_indices_to_plot,
                highs_solution,  # Pass the full 54-var optimal vector
                dims,  # dims should be (num_areas, num_schools, num_grades, grade_list_for_naming)
                pd_label=f"Path-Following ({it_pd} iters)",
                mpc_label=f"Mehrotra PC ({it_mpc} iters)"
            )



