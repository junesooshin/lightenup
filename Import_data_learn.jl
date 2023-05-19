#data import functions for learning model

function data_import_Learning(Data_all, forecast_data, Data_index, Feature_Selection, scaling)

    N_train_flat = Data_index["N_train_flat"]
    D_train = Data_index["D_train"]
    H  = Data_index["H"]

    #Features
    F = length(Feature_Selection)
    # Training data:
    Data_train = Data_all[N_train_flat, :]
    if scaling == true
        train_df, scaled_train_parameters = min_max_scaler(Data_train, "train", false)
    elseif scaling == false
        train_df = Data_train
    end
    X = reshape(Matrix(train_df[:,Feature_Selection]), (length(H), length(D_train), F))

    # Forecast data
    D_forecast = floor(Int, size(forecast_data)[1]/24) #number of days for forecasts
    if scaling == true
        forecast_df, scaled_forecast_parameters = min_max_scaler(forecast_data, "test", scaled_train_parameters)
    elseif scaling == false
        forecast_df = forecast_data
    end
    X_f = reshape(Matrix(forecast_df[:, Feature_Selection]), (length(H), D_forecast, F))

    n_features = size(X)[3]   # Number of features
    n_train_days = size(X)[2] # Total number of training days
    n_hours = size(X)[1]      # Total number of hours

    # The acceptance in the learning model is just the difference in volume between D-1 and D-2 for up and down
    # Not scaled since they are already percentages
    f_FD1_y_up_t = reshape(Data_all[N_train_flat,"FD1_up_percentage"], (n_hours, n_train_days) )   
    f_FD2_y_up_t = reshape(Data_all[N_train_flat,"FD2_up_percentage"], (n_hours, n_train_days) )    
    f_FD1_y_dn_t = reshape(Data_all[N_train_flat,"FD1_down_percentage"], (n_hours, n_train_days) )  
    f_FD2_y_dn_t = reshape(Data_all[N_train_flat,"FD2_down_percentage"], (n_hours, n_train_days) ) 

    #Activation not scaled since they are already in percentages
    a_up_t   = reshape(Data_all[N_train_flat,"FD_act_up"], (n_hours, n_train_days) ) 
    a_dn_t   = reshape(Data_all[N_train_flat,"FD_act_down"], (n_hours, n_train_days) ) 

    lambda_DA = reshape(Data_train[:,"Spot"], (n_hours, n_train_days))
    lambda_FD2_up = reshape(Data_train[:,"FD2_up"], (n_hours, n_train_days)) 
    lambda_FD2_dn = reshape(Data_train[:,"FD2_down"], (n_hours, n_train_days)) 
    lambda_FD1_up = reshape(Data_train[:,"FD1_up"], (n_hours, n_train_days)) 
    lambda_FD1_dn = reshape(Data_train[:,"FD1_down"], (n_hours, n_train_days)) 

    #########################            FORECASTED PRICES           ############################

    # ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up"]
    f_FD2_up = reshape(forecast_data[:,"FD2_up"], (n_hours, D_forecast) ) # 
    f_FD2_dn = reshape(forecast_data[:,"FD2_down"], (n_hours, D_forecast) ) # 
    f_FD1_up = reshape(forecast_data[:,"FD1_up"], (n_hours, D_forecast) ) # 
    f_FD1_dn = reshape(forecast_data[:,"FD1_down"], (n_hours, D_forecast) ) # 
    f_DA = reshape(forecast_data[:,"Spot"], (n_hours, D_forecast) ) # 

    ##########################               PARAMETERS              ############################
 
    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")

    SOC_0 = Data_Battery["SOC_0"]
    SOC_max = Data_Battery["SOC_max"]
    eta_dis = Data_Battery["eta_dis"]
    eta_ch = Data_Battery["eta_ch"]
    p_dis_max = Data_Battery["p_dis_max"]
    p_ch_max = Data_Battery["p_ch_max"]
    if scaling == true
        Cost_per_cycle = (Data_Battery["Cost_per_cycle"]- scaled_train_parameters["min_spot"])/(scaled_train_parameters["max_spot"]-scaled_train_parameters["min_spot"])
    elseif scaling == false
        Cost_per_cycle = Data_Battery["Cost_per_cycle"]
    end

    Data = Dict(
    "n_features" => n_features, "X" => X, "X_f" => X_f,
    "f_FD1_y_up_t" => f_FD1_y_up_t, "f_FD1_y_dn_t" => f_FD1_y_dn_t, 
    "f_FD2_y_up_t" => f_FD2_y_up_t, "f_FD2_y_dn_t" => f_FD2_y_dn_t, 
    "lambda_FD2_up" => lambda_FD2_up, "lambda_FD2_dn" => lambda_FD2_dn,
    "lambda_FD1_up" => lambda_FD1_up, "lambda_FD1_dn" => lambda_FD1_dn,
    "lambda_DA" => lambda_DA,
    "f_FD2_up" => f_FD2_up, "f_FD2_dn" => f_FD2_dn,
    "f_FD1_up" => f_FD1_up, "f_FD1_dn" => f_FD1_dn,
    "f_DA" => f_DA,
    "a_up_t" => a_up_t, "a_dn_t" => a_dn_t, 
    "SOC_0" => SOC_0, "SOC_max" => SOC_max, "eta_dis" => eta_dis,"eta_ch" => eta_ch,
    "p_dis_max" => p_dis_max,"p_ch_max" => p_ch_max, "Cost_per_cycle" => Cost_per_cycle,
    scaling == true ? "scaled_train_parameters" => scaled_train_parameters : "scaled_train_parameters" => NaN,
    scaling == true ? "scaled_forecast_parameters" => scaled_forecast_parameters : "scaled_forecast_parameters" => NaN
    )
    
    @info("Data import for learning model complete!")
    return Data
end

function min_max_scaler(df, dataset, training_parameters)
    #Add features if they need to be scaled!

    if dataset == "train" 
        max_spot = maximum(df[:,"Spot"])
        min_spot = minimum(df[:,"Spot"])
        max_FD1_down = maximum(df[:,"FD1_down"])
        min_FD1_down = minimum(df[:,"FD1_down"])
        max_FD2_down = maximum(df[:,"FD2_down"])
        min_FD2_down = minimum(df[:,"FD2_down"])
        max_FD1_up = maximum(df[:,"FD1_up"])
        min_FD1_up = minimum(df[:,"FD1_up"])
        max_FD2_up = maximum(df[:,"FD2_up"])
        min_FD2_up = minimum(df[:,"FD2_up"])
    elseif dataset == "test" #Pass transformation from train data
        max_spot = training_parameters["max_spot"]
        min_spot = training_parameters["min_spot"]
        max_FD1_down = training_parameters["max_FD1_down"]
        min_FD1_down = training_parameters["min_FD1_down"]
        max_FD2_down = training_parameters["max_FD2_down"]
        min_FD2_down = training_parameters["min_FD2_down"]
        max_FD1_up = training_parameters["max_FD1_up"]
        min_FD1_up = training_parameters["min_FD1_up"]
        max_FD2_up = training_parameters["max_FD2_up"]
        min_FD2_up = training_parameters["min_FD2_up"]
    end
    
    spot_scaled = (df[:,"Spot"] .- min_spot) ./ (max_spot-min_spot)
    FD1_down_scaled = (df[:,"FD1_down"] .- min_FD1_down) ./ (max_FD1_down-min_FD1_down)
    FD2_down_scaled = (df[:,"FD2_down"] .- min_FD2_down) ./ (max_FD2_down-min_FD2_down)
    FD1_up_scaled = (df[:,"FD1_up"] .- min_FD1_up) ./ (max_FD1_up-min_FD1_up)
    FD2_up_scaled = (df[:,"FD2_up"] .- min_FD2_up) ./ (max_FD2_up-min_FD2_up)

    scaled = Dict("Spot" => spot_scaled,
                  "FD1_down" => FD1_down_scaled,
                  "FD2_down" => FD2_down_scaled,
                  "FD1_up" => FD1_up_scaled,
                  "FD2_up" => FD2_up_scaled)

    df_scaled = DataFrame(scaled)
    
    scaled_parameters = Dict("max_spot" => max_spot,
                            "min_spot" => min_spot,
                            "max_FD1_down" => max_FD1_down,
                            "min_FD1_down" => min_FD1_down,
                            "max_FD2_down" => max_FD2_down,
                            "min_FD2_down" => min_FD2_down,
                            "max_FD1_up" => max_FD1_up,
                            "min_FD1_up" => min_FD1_up,
                            "max_FD2_up" => max_FD2_up,
                            "min_FD2_up" => min_FD2_up) 
    
    return df_scaled, scaled_parameters
end
