
function Deterministic_Model(Data)

    Time = Data["Time"] #Hours in a day
    
    SOC_0 = Data["SOC_0"] # Initial State of charge
    SOC_max = Data["SOC_max"] # State of charge Max
    eta_dis = Data["eta_dis"] # Discharge efficiency
    eta_ch = Data["eta_ch"] # Charge efficiency
    p_dis_max = Data["p_dis_max"] # Charge/discharge caps
    p_ch_max = Data["p_ch_max"]
    Cost_per_cycle = Data["Cost_per_cycle"]

   #####################               FORECASTED PARAMETERS              ######################
 
    f_DA_t     = Data["f_DA_t"] # Forecast for Day-ahead
    f_FD1_up_t = Data["f_FD1_up_t"]  # Forecast for FCR-D D-1 Up
    f_FD2_up_t = Data["f_FD2_up_t"] # Forecast for FCR-D D-2 Up
    f_FD1_dn_t = Data["f_FD1_dn_t"] # Forecast for FCR-D D-1 Down
    f_FD2_dn_t = Data["f_FD2_dn_t"] # Forecast for FCR-D D-2 Down

    f_a_up_t   = Data["f_a_up_t"] # Forecast for Activation in upregulation
    f_a_dn_t   = Data["f_a_dn_t"] # Forecast for Activation in downregulation

    f_FD1_y_up_t = Data["f_FD1_y_up_t"] # Forecast for acceptance in FCR-D D-1 up
    f_FD2_y_up_t = Data["f_FD2_y_up_t"] # Forecast for acceptance in FCR-D D-2 up 
    f_FD1_y_dn_t = Data["f_FD1_y_dn_t"] # Forecast for acceptance in FCR-D D-1 dn
    f_FD2_y_dn_t = Data["f_FD2_y_dn_t"] # Forecast for acceptance in FCR-D D-2 dn

   ############          Model          ###############
    Model_Deterministic = Model(Gurobi.Optimizer) 

   ############          Sets           ###############
    T = Time # Number of time instances
    T2 = T[2:end]
    t_ini = T[1]

   ############        Variable         ###############
    @variable(Model_Deterministic , p_DA_up[T] >= 0)               # Bid Upregulation/Discharging in Day-ahead
    @variable(Model_Deterministic , p_DA_dn[T] >= 0)               # Bid Downregulation/Charging in Day-ahead
    @variable(Model_Deterministic , p_FD1_up[T] >= 0)              # Accepted Bid FCR-D D-1 Upregulation/Discharging (SLACK)
    @variable(Model_Deterministic , p_FD1_dn[T] >= 0)              # Accepted Bid FCR-D D-1 Downregulation/Charging (SLACK)
    @variable(Model_Deterministic , p_FD2_up[T] >= 0)              # Accepted Bid FCR-D D-2 Upregulation/Discharging (SLACK)
    @variable(Model_Deterministic , p_FD2_dn[T] >= 0)              # Accepted Bid FCR-D D-2 Downregulation/Charging (SLACK)

    @variable(Model_Deterministic , b_FD1_up[T] >= 0)              # Provided Bid FCR-D D-1 Upregulation/Discharging
    @variable(Model_Deterministic , b_FD2_up[T] >= 0)              # Provided Bid FCR-D D-2 Upregulation/Discharging
    @variable(Model_Deterministic , b_FD1_dn[T] >= 0)              # Provided Bid FCR-D D-1 Downregulation/Charging
    @variable(Model_Deterministic , b_FD2_dn[T] >= 0)              # Provided Bid FCR-D D-2 Downregulation/Charging
    @variable(Model_Deterministic , p_all_dn[T] >= 0)              # Aggregation of all charge in time t
    @variable(Model_Deterministic , p_all_up[T] >= 0)              # Aggregation of all discharge in time t

    @variable(Model_Deterministic , SOC[T] >= 0)                   # State of charge variable 

    @variable(Model_Deterministic , G_DA[T])                          # Gain/Profit from DA
    @variable(Model_Deterministic , G_FD1[T])                          # Gain/Profit from FCR-D1
    @variable(Model_Deterministic , G_FD2[T])                          # Gain/Profit from FCR-D1
    @variable(Model_Deterministic , C_Deg[T])                         # Cost from degradation

    ############        Objective        ###############
    @objective(Model_Deterministic, Max, sum(G_DA[t] + G_FD2[t] + G_FD1[t] - C_Deg[t] for t in T))

    @constraint(Model_Deterministic, [t in T], G_DA[t] == f_DA_t[t]*(p_DA_up[t] - p_DA_dn[t])) # Constraint to set G_DA
    @constraint(Model_Deterministic, [t in T], G_FD2[t] == f_FD2_up_t[t]*p_FD2_up[t] + f_FD2_dn_t[t]*p_FD2_dn[t]) # Constraint to set G_FD2
    @constraint(Model_Deterministic, [t in T], G_FD1[t] == f_FD1_up_t[t]*p_FD1_up[t] + f_FD1_dn_t[t]*p_FD1_dn[t]) # Constraint to set G_FD1
    @constraint(Model_Deterministic, [t in T], C_Deg[t] == (p_all_dn[t] + p_all_up[t])/(2*SOC_max) * Cost_per_cycle) # Constraint to set G_Bal

    ############   Battery Constraints   ###############

    # All charging:
    @constraint(Model_Deterministic, p_all_dn_con[t in T], p_all_dn[t] == p_DA_dn[t] +  f_a_dn_t[t]*(p_FD1_dn[t] + p_FD2_dn[t])) # Constraint to keep track of the summation of all the charge
    @constraint(Model_Deterministic, p_all_up_con[t in T], p_all_up[t] == p_DA_up[t] +  f_a_up_t[t]*(p_FD1_up[t] + p_FD2_up[t])) # Constraint to keep track of the summation of all the charge

    # STATE OF CHARGE CONSTRAINT
    @constraint(Model_Deterministic, SOC_con[t in T2], SOC[t] == SOC[t-1] 
                                        + eta_ch * p_all_dn[t]
                                        - eta_dis * p_all_up[t] ) # Constraint State of charge

    @constraint(Model_Deterministic, SOC[t_ini] == SOC_0 
                                        + eta_ch * p_all_dn[t_ini]
                                        - eta_dis * p_all_up[t_ini] ) # Constraint State of charge

    @constraint(Model_Deterministic, SOC_cap_con[t in T], SOC[t] >= (p_DA_up[t] + b_FD1_up[t] + b_FD2_up[t]) ) # To ensure that enough energy in battery for upregulation/discharging. The SOC need to be bigger or equal to all the bids combined for that hour
    @constraint(Model_Deterministic, SOC_cap_con2[t in T], SOC[t] <= SOC_max - (p_DA_dn[t] + b_FD1_dn[t] + b_FD2_dn[t]) ) # To ensure that enough energy can be downregulated/charged to the battery. The SOC need to be smaller or equal to the max SOC minus all the downregulated bids combined for that hour

    # CHARGING AND DISCHARGING CONSTRAINT
    @constraint(Model_Deterministic, Charging_con[t in T], p_DA_dn[t] + b_FD1_dn[t] + b_FD2_dn[t] <= p_ch_max ) # Constraint State of charge
    @constraint(Model_Deterministic, Discharging_con[t in T], p_DA_up[t] + b_FD1_up[t] + b_FD2_up[t] <= p_dis_max ) # Constraint State of charge

    ############  ACCEPTANCE CONSTRAINTS  ##############
    @constraint(Model_Deterministic, FD1_up_acc_con[t in T], p_FD1_up[t] == f_FD1_y_up_t[t]*b_FD1_up[t] ) # The true power after corrected for acceptance of the bid
    @constraint(Model_Deterministic, FD2_up_acc_con[t in T], p_FD2_up[t] == f_FD2_y_up_t[t]*b_FD2_up[t] ) # The true power after corrected for acceptance of the bid
    @constraint(Model_Deterministic, FD1_dn_acc_con[t in T], p_FD1_dn[t] == f_FD1_y_dn_t[t]*b_FD1_dn[t] ) # The true power after corrected for acceptance of the bid
    @constraint(Model_Deterministic, FD2_dn_acc_con[t in T], p_FD2_dn[t] == f_FD2_y_dn_t[t]*b_FD2_dn[t] ) # The true power after corrected for acceptance of the bid

    ############         Solving         ###############
    optimize!(Model_Deterministic)

    ############         Results         ###############
    Results_det = Dict("obj_t" => [value.(G_DA[t])+value.(G_FD2[t])+value.(G_FD1[t])-value.(C_Deg[t]) for t in T],
                        "b_FD1_up" => [value.(b_FD1_up[t]) for t in T],
                        "b_FD2_up" => [value.(b_FD2_up[t]) for t in T],
                        "b_FD1_dn" => [value.(b_FD1_dn[t]) for t in T],
                        "b_FD2_dn" => [value.(b_FD2_dn[t]) for t in T],
                        "b_DA_up" => [value.(p_DA_up[t]) for t in T], 
                        "b_DA_dn" => [value.(p_DA_dn[t]) for t in T],
                        "p_FD1_up" => [value.(p_FD1_up[t]) for t in T],
                        "p_FD1_dn" => [value.(p_FD1_dn[t]) for t in T],
                        "p_FD2_up" => [value.(p_FD2_up[t]) for t in T],
                        "p_FD2_dn" => [value.(p_FD2_dn[t]) for t in T],
                        "p_all_dn" => [value.(p_all_dn[t]) for t in T],
                        "p_all_up" => [value.(p_all_up[t]) for t in T],
                        "SOC" => [value.(SOC[t]) for t in T],
                        "G_DA" => [value.(G_DA[t]) for t in T],
                        "G_FD1" => [value.(G_FD1[t]) for t in T],
                        "G_FD2" => [value.(G_FD2[t]) for t in T],
                        "C_Deg" => [value.(C_Deg[t]) for t in T],
                        "f_lambda_FD1_up" => f_FD1_up_t,
                        "f_lambda_FD2_up" => f_FD2_up_t,
                        "f_lambda_FD1_dn" => f_FD1_dn_t,
                        "f_lambda_FD2_dn" => f_FD2_dn_t,
                        "f_DA_t" => f_DA_t,
                        "f_a_up_t" => f_a_up_t,
                        "f_a_dn_t" => f_a_dn_t,
                        "f_FD1_y_up_t" => f_FD1_y_up_t,
                        "f_FD2_y_up_t" => f_FD2_y_up_t,
                        "f_FD1_y_dn_t" => f_FD1_y_dn_t,
                        "f_FD2_y_dn_t" => f_FD2_y_dn_t
                        )    

    @info("New deterministic solution saved!")
    return Results_det
end

