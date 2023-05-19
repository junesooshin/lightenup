#Main functions to run the models 

using DataFrames, CSV, Statistics
using JuMP, Gurobi

#Data handling functions
include("Import_data.jl")
include("Import_data_rule.jl")
include("Import_data_det.jl")
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
function run_rule(m_set, moving_day, Threshold_Max_coef, Threshold_Min_coef, test_day_2023)
    #Rule-based model
    forecast_day_2023 = moving_day + 1 
    processed_data = load_data("processed")
    forecast_data = load_data("forecast")
    Data_index = Define_Training_and_Test_period(m_set, moving_day) #default d=0, AuctionType="D-2"
    
    data_rule = data_import_Rule(forecast_data)
    Bid_Results_rule = baseline_model_DA_and_FCR_D(data_rule, forecast_day_2023, Threshold_Max_coef, Threshold_Min_coef)

    #Test rule-based model real-time
    data_real_rule = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_rule)
    RT_results_rule = RT_operation(data_real_rule)

    result_rule = Dict("Bid" => Bid_Results_rule, "RT" => RT_results_rule)
    
    return result_rule
end

function run_det(m_set, moving_day, test_day_2023)
    #Deterministic Model
    forecast_day_2023 = moving_day + 1 
    processed_data = load_data("processed")
    forecast_data = load_data("forecast")
    Data_index = Define_Training_and_Test_period(m_set, moving_day) #default d=0, AuctionType="D-2"
    data_det = data_import_Deterministic(forecast_data, forecast_day_2023)
    Bid_Results_det  = Deterministic_Model(data_det)

    #Test deterministic model real-time
    data_real_det = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_det)
    RT_results_det = RT_operation(data_real_det)

    result_det = Dict("Bid" => Bid_Results_det, "RT" => RT_results_det)

    return result_det
end

function run_sto(m_set, moving_day, size_W1, size_W2, size_W3, test_day_2023)
    #Stochastic Model
    forecast_day_2023 = moving_day + 1 
    processed_data = load_data("processed")
    forecast_data = load_data("forecast")
    Data_index = Define_Training_and_Test_period(m_set, moving_day) #default d=0, AuctionType="D-2"

    data_sto = data_import_stochastic(processed_data, forecast_data, Data_index, size_W1, size_W2, size_W3, forecast_day_2023)
    sto_solution = stochastic_model(data_sto)
    Bid_Results_sto = create_bid_stochastic(data_sto, sto_solution)

    #Test stochastic model real-time
    data_real_sto = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_sto)
    RT_results_sto = RT_operation(data_real_sto)

    result_sto = Dict("Bid" => Bid_Results_sto, "RT" => RT_results_sto)

    return result_sto
end

function run_learn(m_set, moving_day, test_day_2023, scaling)
    #Learning Model
    forecast_day_2023 = moving_day + 1 
    processed_data = load_data("processed")
    forecast_data = load_data("forecast")
    Data_index = Define_Training_and_Test_period(m_set, moving_day) #default d=0, AuctionType="D-2"
    # Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up","FD_act_down","FD_act_up"]
    Feature_Selection = ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up"]
    data_learn = data_import_Learning(processed_data, forecast_data, Data_index, Feature_Selection, scaling)

    learn_solution = Training_Learning_Model(data_learn, Data_index)
    Bid_Results_learn = Create_bid_Learning(data_learn, learn_solution, forecast_day_2023)

    #Test learning model real-time
    data_real_learn = data_import_real(processed_data, Data_index, test_day_2023, Bid_Results_learn)
    RT_results_learn = RT_operation(data_real_learn)

    result_learn = Dict("Bid" => Bid_Results_learn, "RT" => RT_results_learn)

    return result_learn
end

function run_all(m_set_range, moving_day_range, out_of_sample, scaling, save_all)
    #Fixed parameters
    Threshold_Max_coef = 0.9
    Threshold_Min_coef = 1.1
    best_scenario_combinations = [2  2  3  3  3  4  5  5  5  5  4  5;      
                                  3  5  5  5  5  5  6  6  6  6  8  8;      
                                  5  6  6  8 10  9  7  8  9 10 10  9] 

    RT_revenue = Dict()
    Exp_revenue = Dict()
    #Run models
    for m_set in m_set_range
        size_W1 = best_scenario_combinations[1,m_set] 
        size_W2 = best_scenario_combinations[2,m_set] 
        size_W3 = best_scenario_combinations[3,m_set] 
        for moving_day in moving_day_range
            if out_of_sample == false
                test_day_2023_range = 1
            elseif out_of_sample == true
                test_day_2023_range = 2:(88-moving_day)
                print(test_day_2023_range)
            end
            for test_day_2023 in test_day_2023_range
                id = "m$(m_set)_d$(moving_day)_t$(test_day_2023+moving_day)"
                @info("Started running id: $(id)")

                result_rule = run_rule(m_set, moving_day, Threshold_Max_coef, Threshold_Min_coef, test_day_2023)
                result_det = run_det(m_set, moving_day, test_day_2023)
                result_sto = run_sto(m_set, moving_day, size_W1, size_W2, size_W3, test_day_2023)
                result_learn = run_learn(m_set, moving_day, test_day_2023, scaling)
                
                #Store RT results for all models
                RT_revenue[id] = Dict("rule" => result_rule["RT"]["revenue"],
                                      "det" => result_det["RT"]["revenue"],
                                      "sto" => result_sto["RT"]["revenue"],
                                      "learn" => result_learn["RT"]["revenue"])
                Exp_revenue[id] = Dict("rule" => result_rule["Bid"]["obj"],
                                       "det" => result_det["Bid"]["obj"],
                                       "sto" => result_sto["Bid"]["obj"],
                                       "learn" => result_learn["Bid"]["obj"])
                #Save all solutions
                if save_all == true
                    save_dict(result_rule, "rule_$(id)")
                    save_dict(result_det, "det_$(id)")
                    save_dict(result_sto, "sto_$(id)")
                    save_dict(result_learn, "learn_$(id)")
                end
                @info("Finished running id: $(id)")
            end
        end
    end
    #Save RT revenue and expected revenue results for all models
    save_dict(RT_revenue, "RT_revenue")
    save_dict(Exp_revenue, "Exp_revenue")

    return RT_revenue, Exp_revenue
end

#Default parameters for 'run_all' function
m_set_range = 1:12 #Set one value for one test case (within range 1:12)
moving_day_range = 0 #(within range 0:87)
out_of_sample = false #true/false (if true, moving day cannot be more than 86)
scaling = true #true/false (for learning)
save_all = false #true/false (for saving individual results)
RT_revenue, Exp_revenue = run_all(m_set_range, moving_day_range, out_of_sample, scaling, save_all)

