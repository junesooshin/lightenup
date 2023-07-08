#Main functions to run the models 

using DataFrames, CSV, Statistics
using JuMP, Gurobi#, BenchmarkTools

#Data handling functions
include("Import_data.jl")
include("Import_data_rule.jl")
include("Import_data_det.jl")
include("Import_data_oracle.jl")
include("Import_data_sto.jl")
include("Import_data_feature.jl")
include("Import_data_real.jl")

#Model functions
include("Model_Rule_based.jl")
include("Model_Deterministic.jl")
include("Model_Stochastic.jl")
include("Model_Feature.jl")
include("Model_Realtime.jl")

#Functions to run models
function run_rule(processed_data, forecast_data, d_train_set, moving_day, Threshold_Max_coef, Threshold_Min_coef, test_day_2023,gamma)
    
    #Rule-based model
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day)
    data_rule = data_import_Rule(forecast_data,Data_index,gamma)

    Bid_Results_rule = baseline_model_DA_and_FCR_D(data_rule, Threshold_Max_coef, Threshold_Min_coef)

    #Test rule-based model real-time
    data_real_rule = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_rule)
    RT_results_rule = RT_operation(data_real_rule)

    result_rule = Dict("Bid" => Bid_Results_rule, "RT" => RT_results_rule)
    
    return result_rule
end

function run_det(processed_data, forecast_data, d_train_set, moving_day, test_day_2023,gamma)

    #Deterministic Model
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day) 
    data_det = data_import_Deterministic(forecast_data, Data_index,gamma)
    Bid_Results_det  = Deterministic_Model(data_det)
    
    #Test deterministic model real-time
    data_real_det = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_det)
    
    RT_results_det = RT_operation(data_real_det)

    result_det = Dict("Bid" => Bid_Results_det, "RT" => RT_results_det)

    return result_det
end

function run_oracle(processed_data, d_train_set, moving_day, test_day_2023)

    #Deterministic Model
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day)
    data_oracle = data_import_Oracle(processed_data, Data_index,test_day_2023)
    
    Bid_Results_oracle  = Deterministic_Model(data_oracle)
    
    #Test deterministic model real-time
    data_real_oracle = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_oracle)
    
    RT_results_oracle = RT_operation(data_real_oracle)

    result_oracle = Dict("Bid" => Bid_Results_oracle, "RT" => RT_results_oracle)

    return result_oracle
end

function run_sto(processed_data, forecast_data, d_train_set, moving_day,gamma, size_W1, size_W2, size_W3, test_day_2023,Acceptance)

    #Stochastic Model
    Data_index = Define_Training_and_Test_index(d_train_set, moving_day)
    data_sto = data_import_stochastic(processed_data, forecast_data, Data_index,gamma, size_W1, size_W2, size_W3,false,false,Acceptance,"With forecast in input")
    #data_sto = data_import_stochastic(processed_data, forecast_data, Data_index, size_W1, size_W2, size_W3,"Without forecast in input")

    sto_solution = stochastic_model(data_sto)
    Bid_Results_sto = create_bid_stochastic(data_sto, sto_solution)
    data_real_sto = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_sto)
    RT_results_sto = RT_operation(data_real_sto)

    result_sto = Dict("Bid" => Bid_Results_sto, "RT" => RT_results_sto)

    return result_sto
end

function run_feature(processed_data, forecast_data, Architecture, d_train_set, moving_day,gamma, test_day_2023, scaling,Acceptance)
    #Feature Model

    Data_index = Define_Training_and_Test_index(d_train_set, moving_day)

    #Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up","FD_act_down","FD_act_up"]
    Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up"]
    #Feature_Selection = ["Spot", "FD1_down", "FD2_down", "FD1_up", "FD2_up", "Spot^2", "Spot_FD1_down", "Spot_FD2_down", "Spot_FD1_up", "Spot_FD2_up", "FD1_down^2", "FD1_down_FD2_down", "FD1_down_FD1_up", "FD1_down_FD2_up", "FD2_down^2", "FD2_down_FD1_up", "FD2_down_FD2_up", "FD1_up^2", "FD1_up_FD2_up", "FD2_up^2"]
    
    data_feature = data_import_Feature(processed_data, forecast_data, Data_index,gamma, Feature_Selection, scaling,false,Acceptance,"With forecast in input")

    feature_solution = Feature_Model(data_feature, Architecture)
    
    Bid_Results_feature = Create_bid_Feature(data_feature, feature_solution, Architecture)

    #Test Feature model real-time
    data_real_feature = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_feature)
    
    RT_results_feature = RT_operation(data_real_feature)

    result_feature = Dict("Bid" => Bid_Results_feature, "RT" => RT_results_feature)

    return result_feature
end

function run_all(Models_range, d_train_set_range, moving_day_range,forecast_range,gamma_range_range,out_of_sample, scaling,Acceptance, save_all)
    
    
    #Fixed parameters
    Threshold_Max_coef = 0.9
    Threshold_Min_coef = 1.1 

    processed_data = load_data("real")
    #processed_data = load_data("features")

    RT_profit = Dict()
    Exp_profit = Dict()
    

    #Run models
    
    for (f,forecast) in enumerate(forecast_range) # Specify which forecast is used

        forecast_data = load_data(forecast) # Loop over different forecast accuracies

        for gamma in gamma_range # Specify which gamma is used to multiply on the forecast
            
            for d_train_set in d_train_set_range # Specify the number of days used for the samples to stochastic and feature
                size_W1 = d_train_set 
                size_W2 = d_train_set
                size_W3 = d_train_set

                for moving_day in moving_day_range # Specify the moving day. Which day is in the test
                    if out_of_sample == false
                        test_day_2023_range = 1
                    elseif out_of_sample == true
                        test_day_2023_range = 2:(88-moving_day)
                        print(test_day_2023_range)
                    end

                    for test_day_2023 in test_day_2023_range # Take the test day

                        
                        if length(gamma_range) > 1
                            id = "acc_$(gamma)_f$(f)_d$(d_train_set)_upd$(moving_day)_t$(test_day_2023+moving_day)"
                        else
                            id = "f$(f)_d$(d_train_set)_upd$(moving_day)_t$(test_day_2023+moving_day)"
                        end

                        @info("Started running id: $(id)")
                        

                        if issubset(["rule"],Models_range)  == true # Check if rule need to be runned
                            result_rule = run_rule(processed_data, forecast_data, d_train_set, moving_day, Threshold_Max_coef, Threshold_Min_coef, test_day_2023,gamma)
                            RT_rule_profit = result_rule["RT"]["profit"]
                            Exp_rule_profit = sum(result_rule["Bid"]["obj_t"])
                            if save_all == true
                                save_dict(result_rule, "rule_$(id)")
                            end
                        else
                            RT_rule_profit = 0
                            Exp_rule_profit = 0
                        end

                        if issubset(["det"],Models_range)  == true
                            result_det = run_det(processed_data, forecast_data, d_train_set, moving_day, test_day_2023,gamma)
                            RT_det_profit = result_det["RT"]["profit"]
                            Exp_det_profit = sum(result_det["Bid"]["obj_t"])
                            if save_all == true
                                save_dict(result_det, "det_$(id)")
                            end
                        else
                            RT_det_profit = 0
                            Exp_det_profit = 0
                        end

                        if issubset(["oracle"],Models_range)  == true
                            result_oracle = run_oracle(processed_data, d_train_set, moving_day, test_day_2023)
                            RT_oracle_profit = result_oracle["RT"]["profit"]
                            Exp_oracle_profit = sum(result_oracle["Bid"]["obj_t"])
                            if save_all == true
                                save_dict(result_oracle, "oracle_$(id)")
                            end
                        else
                            RT_oracle_profit = 0
                            Exp_oracle_profit = 0
                        end
                        
                        if issubset(["sto"],Models_range) == true
                            result_sto = run_sto(processed_data, forecast_data,d_train_set, moving_day,gamma, size_W1, size_W2, size_W3, test_day_2023,Acceptance)
                            RT_sto_profit = result_sto["RT"]["profit"]
                            Exp_sto_profit = sum(result_sto["Bid"]["obj_t"])
                            if save_all == true
                                save_dict(result_sto, "sto_$(id)")
                            end
                        else
                            RT_sto_profit = 0
                            Exp_sto_profit = 0
                        end

                        if issubset(["feature"],Models_range)  == true
                            Architectures = ["GA"] # General or Hourly architecture of the coefficients
                            for Architecture in Architectures
                                
                                result_feature = run_feature(processed_data, forecast_data, Architecture , d_train_set, moving_day,gamma, test_day_2023, scaling,Acceptance)
                                RT_feature_profit = result_feature["RT"]["profit"]
                                Exp_feature_profit = sum(result_feature["Bid"]["obj_t"])
                                if save_all == true
                                    save_dict(result_feature, "feature_$(id)")
                                end 
                            end    
                        else
                            RT_feature_profit = 0
                            Exp_feature_profit = 0                
                        end

                        
                        #Store RT results for all models
                        RT_profit[id] = Dict("rule" => RT_rule_profit,
                                            "det" => RT_det_profit,
                                            "oracle" => RT_oracle_profit,
                                            "sto" => RT_sto_profit,
                                            "feature" => RT_feature_profit)
                        Exp_profit[id] = Dict("rule" => Exp_rule_profit,
                                            "det" => Exp_det_profit,
                                            "oracle" => Exp_oracle_profit,
                                            "sto" => Exp_sto_profit,
                                            "feature" => Exp_feature_profit)

                        @info("Finished running id: $(id)")
                    
                    end
                end
            end
        end
    end

    #Save RT profit and expected profit results for all models
    if out_of_sample == true
        save_dict(RT_profit, "RT_profit_OoS")
        save_dict(Exp_profit, "Exp_profit_OoS")
    elseif out_of_sample == false
        save_dict(RT_profit, "RT_profit")
        save_dict(Exp_profit, "Exp_profit")
    end

    return RT_profit, Exp_profit
end

#gamma_range = [0.8,0.83,0.85,0.87,0.9,0.93,0.95,0.97, 1.0,1.05,1.1]
gamma_range = [0.85]

#Models_range = ["feature"]
Models_range = ["rule","det","oracle","sto","feature"]

#Default parameters for 'run_all' function
d_train_set_range = [5]
#d_train_set_range = [2,5,10,20,40,80,160,320,365]
#d_train_set_range = [2,4,5,7,9,11]

#moving_day_range = 0   #(within range 0:87)
moving_day_range = 0:87 #(within range 0:87)

forecast_range = ["forecast_all1"]
#forecast_range = ["forecast_all2"]
#forecast_range = ["forecast_allfeature"]

Acceptance = ""
# Settings for different forecast and acceptance:
#forecast_range = ["forecast_all0","forecast_all2"] # load_data("real")
#forecast_range = ["forecast_all0_acc","forecast_all2_acc"] # load_data("real_acc")
#forecast_range = ["forecast_all0_acc","forecast_all2_acc_perf"] # load_data("real_acc")

out_of_sample = false #true/false (if true, moving day cannot be more than 86) !FIX m_set_range and moving_day when running out-of-sample!
scaling = true #true/false (for Feature)
save_all = false #true/false (for saving individual results)

RT_profit, Exp_profit = run_all(Models_range,d_train_set_range, moving_day_range,forecast_range,gamma_range, out_of_sample, scaling,Acceptance, save_all)


