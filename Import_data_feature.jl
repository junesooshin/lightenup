#data import functions for Feature model

function data_import_Feature(Data_all, forecast_data, forgettingFactor_data, Data_index, Feature_Selection, scaling, Model_configuration = "Without forecast in input")

    N_train_flat = Data_index["N_train_flat"]
    D_train = Data_index["D_train"]

    D_forecast = Data_index["D_forecast"]
    N_forecast_flat = Data_index["N_forecast_flat"]
    
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
    X_train_f = reshape(Matrix(forecast_data[N_train_flat,Feature_Selection]), (length(H), length(D_train), F))

    n_features = size(X)[3]   # Number of features
    n_train_days = size(X)[2] # Total number of training days
    n_hours = size(X)[1]      # Total number of hours

    # Forgetting factor (D)
    # Column names [1.0,0.9,0.96,0.98,0.985,0.99,0.995,0.999]
    forgetting = forgettingFactor_data[D_train,"1.0"]

    # Forecast data
    if scaling == true
        forecast_df, scaled_forecast_parameters = min_max_scaler(forecast_data, "test", scaled_train_parameters)
    elseif scaling == false
        forecast_df = forecast_data
    end
    
    X_f = reshape(Matrix(forecast_df[N_forecast_flat, Feature_Selection]),(length(H),1,F))
    



    # The acceptance in the Feature model is just the difference in volume between D-1 and D-2 for up and down
    # Not scaled since they are already percentages
    f_FD1_y_up_t = reshape(Data_all[N_train_flat,"FD1_up_percentage"], (length(H), n_train_days) )   
    f_FD2_y_up_t = reshape(Data_all[N_train_flat,"FD2_up_percentage"], (length(H), n_train_days) )    
    f_FD1_y_dn_t = reshape(Data_all[N_train_flat,"FD1_down_percentage"], (length(H), n_train_days) )  
    f_FD2_y_dn_t = reshape(Data_all[N_train_flat,"FD2_down_percentage"], (length(H), n_train_days) ) 

    #Activation not scaled since they are already in percentages
    a_up_t   = reshape(Data_all[N_train_flat,"FD_act_up"], (length(H), n_train_days) ) 
    a_dn_t   = reshape(Data_all[N_train_flat,"FD_act_down"], (length(H), n_train_days) ) 

    lambda_DA = reshape(Data_train[:,"Spot"], (length(H), n_train_days))
    lambda_FD2_up = reshape(Data_train[:,"FD2_up"], (length(H), n_train_days)) 
    lambda_FD2_dn = reshape(Data_train[:,"FD2_down"], (length(H), n_train_days)) 
    lambda_FD1_up = reshape(Data_train[:,"FD1_up"], (length(H), n_train_days)) 
    lambda_FD1_dn = reshape(Data_train[:,"FD1_down"], (length(H), n_train_days)) 

    #########################            FORECASTED PRICES           ############################

    # ["Spot", "FD1_down","FD2_down","FD1_up","FD2_up"]
    f_FD2_up_t = reshape(forecast_data[N_forecast_flat,"FD2_up"], (length(H)) ) 
    f_FD2_dn_t = reshape(forecast_data[N_forecast_flat,"FD2_down"], (length(H)) ) 
    f_FD1_up_t = reshape(forecast_data[N_forecast_flat,"FD1_up"], (length(H)) ) 
    f_FD1_dn_t = reshape(forecast_data[N_forecast_flat,"FD1_down"], (length(H)) ) 
    f_DA_t     = reshape(forecast_data[N_forecast_flat,"Spot"], (length(H)) ) 
    f_a_up_t   = reshape(forecast_data[N_forecast_flat,"FD_act_up"], (length(H)) ) 
    f_a_dn_t   = reshape(forecast_data[N_forecast_flat,"FD_act_down"], (length(H)) ) 
    f_FD1_down_percentage = reshape(forecast_data[N_forecast_flat,"FD1_up_percentage"], (length(H)) )   
    f_FD2_up_percentage = reshape(forecast_data[N_forecast_flat,"FD2_up_percentage"], (length(H)) )    
    f_FD1_up_percentage = reshape(forecast_data[N_forecast_flat,"FD1_down_percentage"], (length(H)) )  
    f_FD2_down_percentage = reshape(forecast_data[N_forecast_flat,"FD2_down_percentage"], (length(H)) ) 



    # Modify training if forecast is included in training for Feature models
    if Model_configuration == "With forecast in input"

        n_train_days = n_train_days + 1

        X_with_forecast = cat(X, X_f,dims=(2)) # Add the forecast to the training data
        X = X_with_forecast

        # Collect the sample data and the forecasted price
        lambda_DA = cat(lambda_DA, f_DA_t,dims=(2)) # Collect the spot
        lambda_FD1_dn = cat(lambda_FD1_dn, f_FD1_dn_t,dims=(2))
        lambda_FD2_dn = cat(lambda_FD2_dn, f_FD2_dn_t,dims=(2))
        lambda_FD1_up = cat(lambda_FD1_up, f_FD1_up_t,dims=(2))
        lambda_FD2_up = cat(lambda_FD2_up, f_FD2_up_t,dims=(2))
        a_up_t = cat(a_up_t, f_a_up_t,dims=(2))
        a_dn_t = cat(a_dn_t, f_a_dn_t,dims=(2))

        f_FD1_y_dn_t = cat(f_FD1_y_dn_t, f_FD1_down_percentage,dims=(2))
        f_FD2_y_dn_t = cat(f_FD2_y_dn_t, f_FD2_down_percentage,dims=(2))
        f_FD1_y_up_t = cat(f_FD1_y_up_t, f_FD1_up_percentage,dims=(2))
        f_FD2_y_up_t = cat(f_FD2_y_up_t, f_FD2_up_percentage,dims=(2))
        
        @info("Forecast added to input for optimization of the Feature model!")

    end


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
    "n_features" => n_features, "X" => X, "X_f" => X_f, "X_train_f" => X_train_f,
    "f_FD1_y_up_t" => f_FD1_y_up_t, "f_FD1_y_dn_t" => f_FD1_y_dn_t, 
    "f_FD2_y_up_t" => f_FD2_y_up_t, "f_FD2_y_dn_t" => f_FD2_y_dn_t, 
    "lambda_FD2_up" => lambda_FD2_up, "lambda_FD2_dn" => lambda_FD2_dn,
    "lambda_FD1_up" => lambda_FD1_up, "lambda_FD1_dn" => lambda_FD1_dn,
    "lambda_DA" => lambda_DA,
    "f_FD2_up_t" => f_FD2_up_t, "f_FD2_dn_t" => f_FD2_dn_t,
    "f_FD1_up_t" => f_FD1_up_t, "f_FD1_dn_t" => f_FD1_dn_t,
    "f_DA_t" => f_DA_t,
    "a_up_t" => a_up_t, "a_dn_t" => a_dn_t, 
    "SOC_0" => SOC_0, "SOC_max" => SOC_max, "eta_dis" => eta_dis,"eta_ch" => eta_ch,
    "p_dis_max" => p_dis_max,"p_ch_max" => p_ch_max, "Cost_per_cycle" => Cost_per_cycle,
    scaling == true ? "scaled_train_parameters" => scaled_train_parameters : "scaled_train_parameters" => NaN,
    scaling == true ? "scaled_forecast_parameters" => scaled_forecast_parameters : "scaled_forecast_parameters" => NaN,
    "forgettingFactor" => forgetting 
    )
    
    @info("Data import for feature model complete!")
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
