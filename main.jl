#Main functions to run the models 

using DataFrames, CSV, Statistics
using JuMP, Gurobi

#Data handling functions
include("Import_data.jl")
include("Import_data_rule.jl")
include("Import_data_det.jl")
include("Import_data_oracle.jl")
include("Import_data_sto.jl")
include("Import_data_learn.jl")
include("Import_data_real.jl")

#Model functions
include("Model_Rule_based.jl")
include("Model_Deterministic.jl")
include("Model_Stochastic.jl")
include("Model_Learning.jl")
include("Model_Realtime.jl")

#Functions to run models
function run_rule(processed_data, forecast_data, d_train_set, moving_day, Threshold_Max_coef, Threshold_Min_coef, test_day_2023)
    
    #Rule-based model
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day) #default d=0, AuctionType="D-2"
    data_rule = data_import_Rule(forecast_data,Data_index)

    Bid_Results_rule = baseline_model_DA_and_FCR_D(data_rule, Threshold_Max_coef, Threshold_Min_coef)

    #Test rule-based model real-time
    data_real_rule = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_rule)
    RT_results_rule = RT_operation(data_real_rule)

    result_rule = Dict("Bid" => Bid_Results_rule, "RT" => RT_results_rule)
    
    return result_rule
end

function run_det(processed_data, forecast_data, d_train_set, moving_day, test_day_2023)

    #Deterministic Model
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day) 
    data_det = data_import_Deterministic(forecast_data, Data_index)
    Bid_Results_det  = Deterministic_Model(data_det)
    
    #Test deterministic model real-time
    data_real_det = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_det)
    
    RT_results_det = RT_operation(data_real_det)

    result_det = Dict("Bid" => Bid_Results_det, "RT" => RT_results_det)

    return result_det
end

function run_oracle(processed_data, d_train_set, moving_day, test_day_2023)

    #Deterministic Model
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day) #default d=0, AuctionType="D-2"
    data_oracle = data_import_Oracle(processed_data, Data_index,test_day_2023)
    
    Bid_Results_oracle  = Deterministic_Model(data_oracle)
    
    #Test deterministic model real-time
    data_real_oracle = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_oracle)
    
    RT_results_oracle = RT_operation(data_real_oracle)

    result_oracle = Dict("Bid" => Bid_Results_oracle, "RT" => RT_results_oracle)

    return result_oracle
end

function run_sto(processed_data, forecast_data, d_train_set, moving_day, size_W1, size_W2, size_W3, test_day_2023)
    #Stochastic Model
    forecast_day_2023 = moving_day + 1 
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day) #default d=0, AuctionType="D-2"

    data_sto = data_import_stochastic(processed_data, forecast_data, Data_index, size_W1, size_W2, size_W3, forecast_day_2023)
    sto_solution = stochastic_model(data_sto)
    #print(data_sto["f_FD1_up_t"])

    Bid_Results_sto = create_bid_stochastic(data_sto, sto_solution)
    #print(Bid_Results_sto["f_lambda_FD1_up"])
    #Test stochastic model real-time
    data_real_sto = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_sto)
    #print(data_real_sto["FD1_up_price"])
    RT_results_sto = RT_operation(data_real_sto)

    result_sto = Dict("Bid" => Bid_Results_sto, "RT" => RT_results_sto)

    return result_sto
end

function run_learn(processed_data, forecast_data, forgettingFactor_data, d_train_set, moving_day, test_day_2023, scaling)
    #Learning Model
    forecast_day_2023 = moving_day + 1 

    Data_index = Define_Training_and_Test_index(d_train_set, moving_day) #default d=0, AuctionType="D-2"

    # Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up","FD_act_down","FD_act_up"]
    Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up"]
    #Feature_Selection = ["Spot","FD1_down","FD2_down","FD1_up","FD2_up","Spot^2","Spot FD1_down","Spot FD2_down","Spot FD1_up","Spot FD2_up","FD1_down^2","FD1_down FD2_down","FD1_down FD1_up","FD1_down FD2_up","FD2_down^2","FD2_down FD1_up","FD2_down FD2_up","FD1_up^2","FD1_up FD2_up","FD2_up^2"]
    
    data_learn = data_import_Learning(processed_data, forecast_data, forgettingFactor_data, Data_index, Feature_Selection, scaling)

    Architecture = "GA" # General or Hourly architecture of the coefficients
    learn_solution = Training_Learning_Model(data_learn, Data_index, Architecture)
    print(data_learn["f_FD1_up_t"])
    Bid_Results_learn = Create_bid_Learning(data_learn, learn_solution, forecast_day_2023)

    #Test learning model real-time
    data_real_learn = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_learn)
    print(data_real_learn["FD1_up_price"])
    RT_results_learn = RT_operation(data_real_learn)

    result_learn = Dict("Bid" => Bid_Results_learn, "RT" => RT_results_learn)

    return result_learn
end

function run_all(Models_range, d_train_set_range, moving_day_range,forecast_range, out_of_sample, scaling, save_all)
    #Fixed parameters
    Threshold_Max_coef = 0.9
    Threshold_Min_coef = 1.1 

    processed_data = load_data("processed")
    forgettingFactor_data = load_data("forgettingFactor")

    RT_revenue = Dict()
    Exp_revenue = Dict()
    

    #Run models
    for (f,forecast) in enumerate(forecast_range)

        forecast_data = load_data(forecast) # Loop over different forecast accuracies

        for d_train_set in d_train_set_range
            size_W1 = d_train_set 
            size_W2 = d_train_set
            size_W3 = d_train_set
            for moving_day in moving_day_range
                if out_of_sample == false
                    test_day_2023_range = 1
                elseif out_of_sample == true
                    test_day_2023_range = 2:(88-moving_day)
                    print(test_day_2023_range)
                end
                for test_day_2023 in test_day_2023_range
                    id = "f$(f)_d$(d_train_set)_upd$(moving_day)_t$(test_day_2023+moving_day)"
                    @info("Started running id: $(id)")
                    

                    if issubset(["rule"],Models_range)  == true # Check if rule need to be runned
                        result_rule = run_rule(processed_data, forecast_data, d_train_set, moving_day, Threshold_Max_coef, Threshold_Min_coef, test_day_2023)
                        RT_rule_revenue = result_rule["RT"]["revenue"]
                        Exp_rule_revenue = sum(result_rule["Bid"]["obj_t"])
                        if save_all == true
                            save_dict(result_rule, "rule_$(id)")
                        end
                    else
                        RT_rule_revenue = 0
                        Exp_rule_revenue = 0
                    end

                    if issubset(["det"],Models_range)  == true
                        result_det = run_det(processed_data, forecast_data,d_train_set, moving_day, test_day_2023)
                        RT_det_revenue = result_det["RT"]["revenue"]
                        Exp_det_revenue = sum(result_det["Bid"]["obj_t"])
                        if save_all == true
                            save_dict(result_det, "det_$(id)")
                        end
                    else
                        RT_det_revenue = 0
                        Exp_det_revenue = 0
                    end

                    if issubset(["oracle"],Models_range)  == true
                        result_oracle = run_oracle(processed_data, d_train_set, moving_day, test_day_2023)
                        RT_oracle_revenue = result_oracle["RT"]["revenue"]
                        Exp_oracle_revenue = sum(result_oracle["Bid"]["obj_t"])
                        if save_all == true
                            save_dict(result_oracle, "oracle_$(id)")
                        end
                    else
                        RT_oracle_revenue = 0
                        Exp_oracle_revenue = 0
                    end
                    
                    if issubset(["sto"],Models_range) == true
                        result_sto = run_sto(processed_data, forecast_data,d_train_set, moving_day, size_W1, size_W2, size_W3, test_day_2023)
                        RT_sto_revenue = result_sto["RT"]["revenue"]
                        Exp_sto_revenue = sum(result_sto["Bid"]["obj_t"])
                        if save_all == true
                            save_dict(result_sto, "sto_$(id)")
                        end
                    else
                        RT_sto_revenue = 0
                        Exp_sto_revenue = 0
                    end

                    if issubset(["learn"],Models_range)  == true
                        result_learn = run_learn(processed_data, forecast_data, forgettingFactor_data , d_train_set, moving_day, test_day_2023, scaling)
                        RT_learn_revenue = result_learn["RT"]["revenue"]
                        Exp_learn_revenue = sum(result_learn["Bid"]["obj_t"])
                        if save_all == true
                            save_dict(result_learn, "learn_$(id)")
                        end     
                    else
                        RT_learn_revenue = 0
                        Exp_learn_revenue = 0                
                    end

                    
                    #Store RT results for all models
                    RT_revenue[id] = Dict("rule" => RT_rule_revenue,
                                        "det" => RT_det_revenue,
                                        "oracle" => RT_oracle_revenue,
                                        "sto" => RT_sto_revenue,
                                        "learn" => RT_learn_revenue)
                    Exp_revenue[id] = Dict("rule" => Exp_rule_revenue,
                                        "det" => Exp_det_revenue,
                                        "oracle" => Exp_oracle_revenue,
                                        "sto" => Exp_sto_revenue,
                                        "learn" => Exp_learn_revenue)

                    @info("Finished running id: $(id)")
                end
            end
        end
    end

    #Save RT revenue and expected revenue results for all models
    if out_of_sample == true
        save_dict(RT_revenue, "RT_revenue_OoS")
        save_dict(Exp_revenue, "Exp_revenue_OoS")
    elseif out_of_sample == false
        save_dict(RT_revenue, "RT_revenue")
        save_dict(Exp_revenue, "Exp_revenue")
    end

    return RT_revenue, Exp_revenue
end

Models_range = ["learn"]
#Models_range = ["rule","det","oracle","sto","learn"]

#Default parameters for 'run_all' function
d_train_set_range = [5]
#d_train_set_range = [2,4,5,7,9,11]
#d_train_set_range = 1:10 #Set one value for one test case 
moving_day_range = 1 #(within range 0:87)
forecast_range = ["forecast_all1"]
#forecast_range = ["forecast1", "forecast2", "forecast3", "forecast4", "forecast5", "forecast6"]
out_of_sample = false #true/false (if true, moving day cannot be more than 86) !FIX m_set_range and moving_day when running out-of-sample!
scaling = true #true/false (for learning)
save_all = true #true/false (for saving individual results)

RT_revenue, Exp_revenue = run_all(Models_range,d_train_set_range, moving_day_range,forecast_range, out_of_sample, scaling, save_all)

