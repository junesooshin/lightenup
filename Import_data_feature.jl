#data import functions for Feature model

function data_import_Feature(Data_all, forecast_data, Data_index, Feature_Selection, scaling,temporal = false, Model_configuration = "Without forecast in input")

    N_train_flat = Data_index["N_train_flat"]
    D_train = Data_index["D_train"]

    D_forecast = Data_index["D_forecast"]
    N_forecast_flat = Data_index["N_forecast_flat"]
    
    H  = Data_index["H"]

    #Features
    F = length(Feature_Selection)
    
    # Training data:
    Data_train = Data_all[N_train_flat, :]

    #println(Data_train[5:7,2])
    #println(Data_train[(5+24):(7+24),2])
    #=
    if temporal == true
        temporal_relation = [0.8, 0.4, 0.3, 0.2, 0.1, 0.8]
        temporal_relation = temporal_relation / sum(temporal_relation)
        block_size = 24
        num_blocks = size(Data_train, 1) ÷ block_size
        
        num_features = size(Data_train, 2)

        for i in 1:num_blocks
            start_index = (i - 1) * block_size + 1
            end_index = i * block_size
            
            for f in 1:num_features
                #println(temporal_relation[i])
                #println(Data_train[start_index:end_index, :] .* temporal_relation[i])
                #map!(x -> x * temporal_relation[i], Data_train[start_index:end_index, :] )
                #Data_train[start_index:end_index, f] = 
                println(Data_train[5:7, f] .* temporal_relation[i])
                Data_train[5:7, f] = Data_train[5:7, f] .* temporal_relation[i]
                println(Data_train[5:7, f])
            end
        end
        #println(Data_train[5:7,2])
        #println(Data_train[(5+24):(7+24),2])
    end=#
    if temporal
        temporal_relation = [0.8, 0.4, 0.3, 0.2, 0.1, 0.8]
        temporal_relation = temporal_relation / sum(temporal_relation)
        block_size = 24
        num_blocks = size(Data_train, 1) ÷ block_size
        num_features = size(Data_train, 2)
    
        # Convert temporal_relation to a 2D array
        #temporal_relation = repeat(temporal_relation, outer=(block_size, 1))
        data_w_relation = zeros(size(Data_train))
        for i in 1:num_blocks
            start_index = (i - 1) * block_size + 1
            end_index = i * block_size
            # for f in 1:num_features
            #     #temp_array = similar(Data_train[start_index:end_index, f])  # Create a temporary array
            #     #@views temp_array .= Data_train[start_index:end_index, f] .* temporal_relation[i]  # Perform element-wise multiplication
            #     #Data_train[start_index:end_index, f] .= temp_array  # Assign the result back to Data_train
                
            #     for h in start_index:end_index
            #         # temp_array = similar(Data_train[h, f])  # Create a temporary array
            #         # @views temp_array = Data_train[h, f] * temporal_relation[i]  # Perform element-wise multiplication
            #         Data_train[h, f] = Data_train[h, f] .* temporal_relation[i]  # Assign the result back to Data_train
            #     end
            # end
            data_w_relation[start_index:end_index, :] .= Data_train[start_index:end_index, :] .* temporal_relation[i]
        end
        Data_train = DataFrame(data_w_relation, names(Data_train))
    end

    if scaling == true
        train_df, Min_train, Max_train = min_max_scaler(Data_train, "train", false,false)
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
    #forgetting = forgettingFactor_data[D_train,"1.0"]

    # Forecast data
    if scaling == true
        forecast_df, Min_forecast,Max_forecast = min_max_scaler(forecast_data, "test", Max_train,Min_train)
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
        D_train = collect(1:n_train_days)

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
        Cost_per_cycle = (Data_Battery["Cost_per_cycle"]- Min_train["Spot"])/(Max_train["Spot"]-Min_train["Spot"])
    elseif scaling == false
        Cost_per_cycle = Data_Battery["Cost_per_cycle"]
    end

    Data = Dict(
    "n_features" => n_features, "D_train" => D_train, "H" => H,
     "X" => X, "X_f" => X_f, "X_train_f" => X_train_f,
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
    "p_dis_max" => p_dis_max,"p_ch_max" => p_ch_max, "Cost_per_cycle" => Cost_per_cycle
    #scaling == true ? "scaled_train_parameters" => scaled_train_parameters : "scaled_train_parameters" => NaN,
    #scaling == true ? "scaled_forecast_parameters" => scaled_forecast_parameters : "scaled_forecast_parameters" => NaN,
    #"forgettingFactor" => forgetting 
    )
    
    @info("Data import for feature model complete!")
    return Data
end


function min_max_scaler(df, dataset, training_Max_parameters,training_Min_parameters)

    featurenames = names(df)
    Scaled_features = zeros(( size(df, 1),length(featurenames)))
    Scaled_features = DataFrame(Scaled_features, Symbol.(featurenames))

    if dataset == "train" 

        min_values = minimum.(eachcol(df))
        Min = Dict(zip(featurenames, min_values))

        max_values = maximum.(eachcol(df))
        Max = Dict(zip(featurenames, max_values))
        
    elseif dataset == "test" #Pass transformation from train data

        # Use min and max from training
        Min = training_Min_parameters
        Max = training_Max_parameters

    end

    for feature in featurenames
        Scaled_features[:,feature] = (df[:,feature] .- Min[feature]) ./ (Max[feature]-Min[feature])
    end
    
    return Scaled_features, Min, Max
end
