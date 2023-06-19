
function Training_Learning_Model(Data, Data_index, Architecture)

    #
    # Data: All the data required for running the 
    # Data_index: Provide the index for the sample data, test index and forecast indexes
    # Architecture: The architecture of the coefficient. Either GA= coefficient for each feature. Or HA=coefficient for each feature and hour
    # Model_configuration: "Without forecast", "With forecast" or "Forecast minimization"



    # Pick the indexes
    idx = Data_index["N_train"]
    D = Data_index["D_train"]
    H = Data_index["H"]

    if Architecture == "GA"
        HF = [1]
    elseif Architecture == "HA"
        HF = H
    end

    # Collect FEATURES SPECIFIC INFORMATION
    X = Data["X"] # All the features
    X_train_f = Data["X_train_f"] # All the forecast features to use in the training 
    X_f = Data["X_f"] # Forecasted feature for the bidding day D

    ## Concatenate X and X_f if forecast is used.
    #if Model_configuration == "With forecast"
    #    # Concatenate such that it go from: X:(24, 5, 5), X_f(24, 5) -> X(24,6,5)
    #    println("hej")
    #    println(size(X),size(X_f))
    #    X_f = reshape(X_f,(24,5,1))
    #    X = cat(X, X_f, dims=2)
    #    println(size(X))
    #end

    F = Data["n_features"] # Total number of features


    # Collect BATTERY SPECIFIC INFORMATION
    SOC_0 = Data["SOC_0"] # Initial State of charge
    SOC_max = Data["SOC_max"] # State of charge Max
    eta_dis = Data["eta_dis"] # Discharge efficiency
    eta_ch = Data["eta_ch"] # Charge efficiency
    p_dis_max = Data["p_dis_max"] # Charge/discharge caps
    p_ch_max = Data["p_ch_max"]
    Cost_per_cycle = Data["Cost_per_cycle"]
    
    # Collect FORECASTS
    #### THIS WE CAN KNOW WITH HISTORICAL DATA
    lambda_DA = Data["lambda_DA"]
    
    #### HERE WE DO NOT KNOW MUCH!!
    # THIS IS THE TRUE VOLUME WEIGHTED AVERAGE AND IS THEREFORE ALWAYS ACCEPTED
    lambda_FD2_up = Data["lambda_FD2_up"] # X[:,:,"FD2_up"] 
    lambda_FD2_dn = Data["lambda_FD2_dn"] # X[:,:,"FD2_down"]
    lambda_FD1_up = Data["lambda_FD1_up"] # X[:,:,"FD1_up"]
    lambda_FD1_dn = Data["lambda_FD1_dn"] #  X[:,:,"FD1_down"]

    #### THIS WE CAN KNOW WITH HISTORICAL DATA
    a_up_t   = Data["a_up_t"] # 
    a_dn_t   = Data["a_dn_t"] # 

    #### HOW CAN WE KNOW WHETHER WE GOT ACCEPTED???? 
    y_FD2_up =  Data["f_FD2_y_up_t"]
    y_FD2_dn =  Data["f_FD2_y_dn_t"]
    y_FD1_up =  Data["f_FD1_y_up_t"]
    y_FD1_dn =  Data["f_FD1_y_dn_t"]

    
    ####################################################
    ############          Model          ###############
    ####################################################

    Model_Learning = Model(Gurobi.Optimizer) # Initialize solver

    ####################################################
    ############          Sets           ###############
    ####################################################


    H2 = H[2:end]
    H_ini = H[1]
    D_ini = D[1]

    ####################################################
    ############        Variable         ###############
    ####################################################

    
    # VALUES TO OPTIMIZE
    @variable(Model_Learning, q_FD2_up[HF,1:(F+1)])                # Coefficient for FCR-D D-2 up. Offset + all features
    @variable(Model_Learning, q_FD2_dn[HF,1:(F+1)])                # Coefficient for FCR-D D-2 dn. Offset + all features
    @variable(Model_Learning, q_DA_up[HF,1:(F+1)])                 # Coefficient for DA up. Offset + all features
    @variable(Model_Learning, q_DA_dn[HF,1:(F+1)])                 # Coefficient for DA dn. Offset + all features
    @variable(Model_Learning, q_FD1_up[HF,1:(F+1)])                # Coefficient for FCR-D D-1 up. Offset + all features
    @variable(Model_Learning, q_FD1_dn[HF,1:(F+1)])                # Coefficient for FCR-D D-1 dn. Offset + all features


    @variable(Model_Learning , SOC[H,D] >= 0)                   # State of charge variable 

    # Helper variables
    @variable(Model_Learning , p_FD2_up[H,D] >= 0)              # Accepted Bid FCR-D D-2 Upregulation/Discharging (SLACK)
    @variable(Model_Learning , p_FD2_dn[H,D] >= 0)              # Accepted Bid FCR-D D-2 Downregulation/Charging (SLACK)
    @variable(Model_Learning , p_DA_up[H,D] >= 0)               # Accepted Bid Upregulation/Discharging in Day-ahead (SLACK)
    @variable(Model_Learning , p_DA_dn[H,D] >= 0)               # Accepted Bid Downregulation/Charging in Day-ahead (SLACK)
    @variable(Model_Learning , p_FD1_up[H,D] >= 0)              # Accepted Bid FCR-D D-1 Upregulation/Discharging (SLACK)
    @variable(Model_Learning , p_FD1_dn[H,D] >= 0)              # Accepted Bid FCR-D D-1 Downregulation/Charging (SLACK)
    @variable(Model_Learning , p_all_dn[H,D] >= 0)              # Aggregation of all charge in time t
    @variable(Model_Learning , p_all_up[H,D] >= 0)              # Aggregation of all discharge in time t

    @variable(Model_Learning , b_FD2_up[H,D] >= 0)              # Provided Bid FCR-D D-2 Upregulation/Discharging
    @variable(Model_Learning , b_FD2_dn[H,D] >= 0)              # Provided Bid FCR-D D-2 Downregulation/Charging
    @variable(Model_Learning , b_DA_up[H,D] >= 0)               # Provided Bid Upregulation/Discharging in Day-ahead 
    @variable(Model_Learning , b_DA_dn[H,D] >= 0)               # Provided Bid Downregulation/Charging in Day-ahead 
    @variable(Model_Learning , b_FD1_up[H,D] >= 0)              # Provided Bid FCR-D D-2 Upregulation/Discharging
    @variable(Model_Learning , b_FD1_dn[H,D] >= 0)              # Provided Bid FCR-D D-2 Downregulation/Charging
    
    @variable(Model_Learning , G_FD1[H,D])                       # Gain/Profit from FCR-D
    @variable(Model_Learning , G_FD2[H,D])                       # Gain/Profit from FCR-D
    @variable(Model_Learning , G_DA[H,D])                       # Gain/Profit from Day-ahead
    @variable(Model_Learning , C_Deg[H,D])                      # Cost from degradation


    ###################################################
    #                     NOTE                        #

    # All variables and parameters will be indexed: [h,d]

    #                                                 #
    ###################################################

    ####################################################
    ############        Objective        ###############
    ####################################################

    @objective(Model_Learning, Max, sum((G_FD1[h,d] + G_FD2[h,d] + G_DA[h,d] - C_Deg[h,d]) for h in H, d in D))

    @constraint(Model_Learning, FD1_slack_obj[h in H, d in D], G_FD1[h,d] == lambda_FD1_up[ h,d ]*p_FD1_up[h,d] + lambda_FD1_dn[ h,d ]*p_FD1_dn[h,d]) # Constraint to set G_FD
    @constraint(Model_Learning, FD2_slack_obj[h in H, d in D], G_FD2[h,d] == lambda_FD2_up[ h,d ]*p_FD2_up[h,d] + lambda_FD2_dn[ h,d ]*p_FD2_dn[h,d]) # Constraint to set G_FD

    @constraint(Model_Learning, DA_slack_obj[h in H, d in D], G_DA[h,d] == lambda_DA[ h,d ]*(p_DA_up[h,d]-p_DA_dn[h,d])) # Constraint to set G_DA

    @constraint(Model_Learning, Deg_slack_obj[h in H, d in D], C_Deg[h,d] == (p_all_dn[h,d] + p_all_up[h,d])/(2*SOC_max) * Cost_per_cycle ) # Constraint to set G_Bal

    ####################################################
    ############   Battery Constraints   ###############
    ####################################################

    # All charging:
    @constraint(Model_Learning, p_all_dn_con[h in H, d in D], p_all_dn[h,d] == p_DA_dn[h,d] + a_dn_t[ h,d ]*(p_FD2_dn[h,d] + p_FD1_dn[h,d])) # Constraint to keep track of the summation of all the charge
    @constraint(Model_Learning, p_all_up_con[h in H, d in D], p_all_up[h,d] == p_DA_up[h,d]+ a_up_t[ h,d ]*(p_FD2_up[h,d] + p_FD1_up[h,d])) # Constraint to keep track of the summation of all the charge


    # STATE OF CHARGE CONSTRAINT

    @constraint(Model_Learning, SOC_con[h in H2, d in D], SOC[h,d] == SOC[h-1,d] + eta_ch*p_all_dn[h,d] - eta_dis*p_all_up[h,d])

    @constraint(Model_Learning, SOC_D2[d in D], SOC[H_ini,d] == SOC_0 + eta_ch*p_all_dn[H_ini,d]- eta_dis*p_all_up[H_ini,d])

    @constraint(Model_Learning, SOC_cap_con[h in H, d in D], SOC[h,d] >= (b_FD2_up[h,d] + p_DA_up[h,d] + b_FD1_up[h,d]) ) # To ensure that enough energy in battery for upregulation/discharging. The SOC need to be bigger or equal to all the bids combined for that hour
    @constraint(Model_Learning, SOC_cap_con2[h in H, d in D], SOC[h,d] <= SOC_max - (b_FD2_dn[h,d] + p_DA_dn[h,d] + b_FD1_dn[h,d]) ) # To ensure that enough energy can be downregulated/charged to the battery. The SOC need to be smaller or equal to the max SOC minus all the downregulated bids combined for that hour

    # CHARGING AND DISCHARGING CONSTRAINT
    @constraint(Model_Learning, Charging_con[h in H, d in D], b_FD2_dn[h,d] + p_DA_dn[h,d] + b_FD2_dn[h,d] <= p_ch_max ) # Constraint State of charge
    @constraint(Model_Learning, Discharging_con[h in H, d in D], b_FD2_up[h,d] + p_DA_up[h,d] + b_FD2_up[h,d] <= p_dis_max ) # Constraint State of charge

    ####################################################
    ############  ACCEPTANCE CONSTRAINTS  ##############
    ####################################################

    @constraint(Model_Learning, FD2_up_acc_con[h in H, d in D], p_FD2_up[h,d] == y_FD2_up[ h,d ]*b_FD2_up[h,d] ) # The true power after corrected for acceptance of the bid
    @constraint(Model_Learning, FD2_dn_acc_con[h in H, d in D], p_FD2_dn[h,d] == y_FD2_dn[ h,d ]*b_FD2_dn[h,d] ) # The true power after corrected for acceptance of the bid
    @constraint(Model_Learning, DA_up_acc_con[h in H, d in D], p_DA_up[h,d] == 1*b_DA_up[h,d] ) # assuming 100 % acceptance in DA
    @constraint(Model_Learning, DA_dn_acc_con[h in H, d in D], p_DA_dn[h,d] == 1*b_DA_dn[h,d] ) # assuming 100 % acceptance in DA
    @constraint(Model_Learning, FD1_up_acc_con[h in H, d in D], p_FD1_up[h,d] == y_FD1_up[ h,d ]*b_FD1_up[h,d] ) # The true power after corrected for acceptance of the bid
    @constraint(Model_Learning, FD1_dn_acc_con[h in H, d in D], p_FD1_dn[h,d] == y_FD1_dn[ h,d ]*b_FD1_dn[h,d] ) # The true power after corrected for acceptance of the bid
    
    # Here you are setting the constraint for the bids for each days within the whole training period
    # As we have features for a long time period, we divide the features in days (24 hours) and great day/i constraints
    

    if Architecture == "GA"
        @constraint(Model_Learning, FD2_Coef_con_up[h in H, d in D], b_FD2_up[h,d] == sum(q_FD2_up[1,f] * X[ h,d, f] for f in 1:F) + q_FD2_up[1,F+1])
        @constraint(Model_Learning, FD2_Coef_con_dn[h in H, d in D], b_FD2_dn[h,d] == sum(q_FD2_dn[1,f] * X[ h,d, f] for f in 1:F) + q_FD2_dn[1,F+1])
        @constraint(Model_Learning, DA_Coef_con_up[h in H,  d in D], b_DA_up[h,d] == sum(q_DA_up[1,f] * X[ h,d, f] for f in 1:F) + q_DA_up[1,F+1])
        @constraint(Model_Learning, DA_Coef_con_dn[h in H,  d in D], b_DA_dn[h,d] == sum(q_DA_dn[1,f] * X[ h,d, f] for f in 1:F) + q_DA_dn[1,F+1])
        @constraint(Model_Learning, FD1_Coef_con_up[h in H,  d in D], b_FD1_up[h,d] == sum(q_FD1_up[1,f] * X[ h,d, f] for f in 1:F) + q_FD1_up[1,F+1])
        @constraint(Model_Learning, FD1_Coef_con_dn[h in H,  d in D], b_FD1_dn[h,d] == sum(q_FD1_dn[1,f] * X[ h,d, f] for f in 1:F) + q_FD1_dn[1,F+1])

    elseif Architecture == "HA"
        @constraint(Model_Learning, FD2_Coef_con_up[h in H, d in D], b_FD2_up[h,d] == sum(q_FD2_up[h,f] * X[ h,d, f] for f in 1:F) + q_FD2_up[h,F+1])
        @constraint(Model_Learning, FD2_Coef_con_dn[h in H, d in D], b_FD2_dn[h,d] == sum(q_FD2_dn[h,f] * X[ h,d, f] for f in 1:F) + q_FD2_dn[h,F+1])
        @constraint(Model_Learning, DA_Coef_con_up[h in H,  d in D], b_DA_up[h,d] == sum(q_DA_up[h,f] * X[ h,d, f] for f in 1:F) + q_DA_up[h,F+1])
        @constraint(Model_Learning, DA_Coef_con_dn[h in H,  d in D], b_DA_dn[h,d] == sum(q_DA_dn[h,f] * X[ h,d, f] for f in 1:F) + q_DA_dn[h,F+1])
        @constraint(Model_Learning, FD1_Coef_con_up[h in H,  d in D], b_FD1_up[h,d] == sum(q_FD1_up[h,f] * X[ h,d, f] for f in 1:F) + q_FD1_up[h,F+1])
        @constraint(Model_Learning, FD1_Coef_con_dn[h in H,  d in D], b_FD1_dn[h,d] == sum(q_FD1_dn[h,f] * X[ h,d, f] for f in 1:F) + q_FD1_dn[h,F+1])
    end


    ####################################################
    ############         Solving         ###############
    ####################################################
    optimize!(Model_Learning)


    ####################################################
    ############         Results         ###############
    ####################################################
    learn_solution = Dict("b_FD2_up" => [value.(b_FD2_up[h,d]) for h in H, d in D],
                        "b_FD2_dn" => [value.(b_FD2_dn[h,d]) for h in H, d in D],
                        "b_DA_up" => [value.(b_DA_up[h,d]) for h in H, d in D],
                        "b_DA_dn" => [value.(b_DA_dn[h,d]) for h in H, d in D],
                        "b_FD1_up" => [value.(b_FD1_up[h,d]) for h in H, d in D],
                        "b_FD1_dn" => [value.(b_FD1_dn[h,d]) for h in H, d in D],   
                        "p_FD2_up" => [value.(p_FD2_up[h,d]) for h in H, d in D],
                        "p_FD2_dn" => [value.(p_FD2_dn[h,d]) for h in H, d in D],
                        "p_DA_up" => [value.(p_DA_up[h,d]) for h in H, d in D],
                        "p_DA_dn" => [value.(p_DA_dn[h,d]) for h in H, d in D],
                        "p_FD1_up" => [value.(p_FD1_up[h,d]) for h in H, d in D],
                        "p_FD1_dn" => [value.(p_FD1_dn[h,d]) for h in H, d in D],
                        "SOC" => [value.(SOC[h,d]) for h in H, d in D],
                        "q_FD2_up" => [value.(q_FD2_up[hf,f]) for hf in HF, f in 1:(F+1)], # Added (F+1) because we want to save the offset as well
                        "q_FD2_dn" => [value.(q_FD2_dn[hf,f]) for hf in HF, f in 1:(F+1)],
                        "q_DA_up" => [value.(q_DA_up[hf,f]) for hf in HF, f in 1:(F+1)],
                        "q_DA_dn" => [value.(q_DA_dn[hf,f]) for hf in HF, f in 1:(F+1)],
                        "q_FD1_up" => [value.(q_FD1_up[hf,f]) for hf in HF, f in 1:(F+1)],
                        "q_FD1_dn" => [value.(q_FD1_dn[hf,f]) for hf in HF, f in 1:(F+1)],
                        "X" => X)

    @info("Learning model solved!")
    return learn_solution
end



function Create_bid_Learning(Data, Results_from_training)

    #=
    Data. (Dict). Data from Data_import_Julia. Should consist of all the features
    Data_index. (Dict). All the indexes used for training
    Results_from_training (the Coefficient from the training model)

    This model takes the coefficients and the bidded prices found from the training and provide the accepted bids with both volume and price

    =#

    # Forecasted values of the features to get the bids

    # Forecasted prices which is equivalent to what we will bid

    # The Calculated coefficients from the model

    ############## The coefficients: ##############
    q_FD2_up = Results_from_training["q_FD2_up"]
    q_FD2_dn = Results_from_training["q_FD2_dn"]
    q_DA_up  = Results_from_training["q_DA_up"]
    q_DA_dn  = Results_from_training["q_DA_dn"]
    q_FD1_up = Results_from_training["q_FD1_up"]
    q_FD1_dn = Results_from_training["q_FD1_dn"]
    
    ############## The training bids: ##############
    b_FD2_up_train = Results_from_training["b_FD2_up"]
    b_FD2_dn_train = Results_from_training["b_FD2_dn"]
    b_DA_up_train  = Results_from_training["b_DA_up"]
    b_DA_dn_train  = Results_from_training["b_DA_dn"]
    b_FD1_up_train = Results_from_training["b_FD1_up"]
    b_FD1_dn_train = Results_from_training["b_FD1_dn"]

    ################ The Forecasted prices i.e. the bid prices ################

    f_DA = Data["f_DA_t"]
    f_lambda_FD2_up = Data["f_FD2_up_t"]
    f_lambda_FD2_dn = Data["f_FD2_dn_t"]
    f_lambda_FD1_up = Data["f_FD1_up_t"]
    f_lambda_FD1_dn = Data["f_FD1_dn_t"]

    features = Data["X_f"]
    columnLength = length(features[1,1,:]) # all the columns in features (not accounting the offset) (F,)
    offset = ones(length(features[:,1,1])) # (24,)

    ############### Compute the bids ##################
    b_FD2_up = sum(q_FD2_up[f] * features[:,1, f] for f in 1:columnLength) + offset * q_FD2_up[columnLength+1]
    b_FD2_dn = sum(q_FD2_dn[f] * features[:,1, f] for f in 1:columnLength) + offset * q_FD2_dn[columnLength+1]
    b_DA_up  = sum(q_DA_up[f] * features[:,1, f] for f in 1:columnLength) + offset * q_DA_up[columnLength+1]
    b_DA_dn  = sum(q_DA_dn[f] * features[:,1, f] for f in 1:columnLength) + offset * q_DA_dn[columnLength+1]
    b_FD1_up = sum(q_FD1_up[f] * features[:,1, f] for f in 1:columnLength) + offset * q_FD1_up[columnLength+1]
    b_FD1_dn = sum(q_FD1_dn[f] * features[:,1, f] for f in 1:columnLength) + offset * q_FD1_dn[columnLength+1]

    # println(size(b_FD2_up))
    # println(size(f_lambda_FD2_up))

    y_FD2_up =  mean(Data["f_FD2_y_up_t"], dims=2)
    y_FD2_dn =  mean(Data["f_FD2_y_dn_t"], dims=2)
    y_FD1_up =  mean(Data["f_FD1_y_up_t"], dims=2)
    y_FD1_dn =  mean(Data["f_FD1_y_dn_t"], dims=2)

    #Calculate expected revenue
    G_DA_t = f_DA.*(b_DA_up .- b_DA_dn)
    G_FD2_t = f_lambda_FD2_up.*b_FD2_up.*y_FD2_up.+f_lambda_FD2_dn.*b_FD2_dn.*y_FD2_dn #Forecast price, trained acceptance, bid
    G_FD1_t = f_lambda_FD1_up.*b_FD1_up.*y_FD1_up.+f_lambda_FD1_dn.*b_FD1_dn.*y_FD1_dn #Forecast price, trained acceptance, bid
    C_Deg_t = ((b_DA_up .- b_DA_dn)) ./ (2*Data["SOC_max"]) .* Data["Cost_per_cycle"] #Degradation only from DA contribution
    obj_t = G_DA_t + G_FD2_t + G_FD1_t - C_Deg_t

    soc = transpose(mean(Results_from_training["SOC"], dims=2))
    X = Results_from_training["X"]

    Bid_Results = Dict("obj_t" => obj_t,
                       "b_FD2_up" => b_FD2_up, "b_FD2_dn" => b_FD2_dn,
                       "b_DA_up"  => b_DA_up , "b_DA_dn"  => b_DA_dn,
                       "b_FD1_up" => b_FD1_up, "b_FD1_dn" => b_FD1_dn,
                       "b_FD2_up_train" => b_FD2_up_train, "b_FD2_dn_train" => b_FD2_dn_train,
                       "b_DA_up_train" => b_DA_up_train, "b_DA_dn_train" => b_DA_dn_train,
                       "b_FD1_up_train" => b_FD1_up_train, "b_FD1_dn_train" => b_FD1_dn_train,
                       "f_lambda_FD2_up" => f_lambda_FD2_up, "f_lambda_FD2_dn" => f_lambda_FD2_dn,
                       "f_lambda_FD1_up" => f_lambda_FD1_up, "f_lambda_FD1_dn" => f_lambda_FD1_dn,
                       "q_FD2_up" => q_FD2_up,
                       "q_FD2_dn" => q_FD2_dn,
                       "q_DA_up" => q_DA_up,
                       "q_DA_dn" => q_DA_dn,
                       "q_FD1_up" => q_FD1_up,
                       "q_FD1_dn" => q_FD1_dn,
                       "SOC" => soc,
                       "f_FD2_y_up_t" => y_FD2_up,
                       "f_FD2_y_dn_t" => y_FD2_dn,
                       "f_FD1_y_up_t" => y_FD1_up,
                       "f_FD1_y_dn_t" => y_FD1_dn,
                       "X" => X
                       )
    
    @info("New learning solution saved!")
    return Bid_Results
end
