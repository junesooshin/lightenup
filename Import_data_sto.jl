#data import function for stochastic model

function data_import_stochastic(data_import, forecast_data, Data_index, W1, W2, W3, forecast_day_2023)
    # Import parameters for stochastic model
    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")

    #Get training data from original df
    first_h_train = Data_index["N_train"][1,1]
    last_h_train = Data_index["N_train"][end,end]
    df_train = data_import[first_h_train:last_h_train, :]
    no_days = floor(Int, size(df_train)[1]/24)

    #Extract each column and reshape
    spot_train = reshape(df_train[:,"Spot"], (24, no_days))
    FD1_down_train = reshape(df_train[:,"FD1_down"], (24, no_days))
    FD2_down_train = reshape(df_train[:,"FD2_down"], (24, no_days))
    FD1_up_train = reshape(df_train[:,"FD1_up"], (24, no_days))
    FD2_up_train = reshape(df_train[:,"FD2_up"], (24, no_days))
    FD_act_down_train = reshape(df_train[:,"FD_act_down"], (24, no_days))
    FD_act_up_train = reshape(df_train[:,"FD_act_up"], (24, no_days))
    FD1_down_percentage = reshape(df_train[:,"FD1_down_percentage"], (24, no_days))
    FD2_down_percentage = reshape(df_train[:,"FD2_down_percentage"], (24, no_days))
    FD1_up_percentage = reshape(df_train[:,"FD1_up_percentage"], (24, no_days))
    FD2_up_percentage = reshape(df_train[:,"FD2_up_percentage"], (24, no_days))

    # #Get forecast data and reshape
    # D_forecast = floor(Int, size(forecast_data)[1]/24) #number of days for forecasts
    # f_lambda_FD1_dn = reshape(forecast_data[:, "FD1_down"], (24, D_forecast))
    # f_lambda_FD2_dn = reshape(forecast_data[:, "FD2_down"], (24, D_forecast))
    # f_lambda_FD1_up   = reshape(forecast_data[:, "FD1_up"], (24, D_forecast))
    # f_lambda_FD2_up   = reshape(forecast_data[:, "FD2_up"], (24, D_forecast))
    # f_DA = reshape(forecast_data[:, "Spot"], (24, D_forecast))
    
    # #Extract volume weighted average values of the test day
    # vol_avg_price_FD1_down = f_lambda_FD1_dn[:, forecast_day_2023]
    # vol_avg_price_FD2_down = f_lambda_FD2_dn[:, forecast_day_2023]
    # vol_avg_price_FD1_up = f_lambda_FD1_up[:, forecast_day_2023]
    # vol_avg_price_FD2_up = f_lambda_FD2_up[:, forecast_day_2023]
    # f_Spot_price = f_DA[:, forecast_day_2023]

    # Forecast price comes from the mean of the scenarios
    vol_avg_price_FD1_down = mean(FD1_down_train, dims=2)
    vol_avg_price_FD2_down = mean(FD2_down_train, dims=2)
    vol_avg_price_FD1_up = mean(FD1_up_train, dims=2)
    vol_avg_price_FD2_up = mean(FD2_up_train, dims=2)
    f_Spot_price = mean(spot_train, dims=2)

    FD1_down_accept_price = construct_acceptance(FD1_down_train, vol_avg_price_FD1_down)
    FD2_down_accept_price = construct_acceptance(FD2_down_train, vol_avg_price_FD2_down)
    FD1_up_accept_price = construct_acceptance(FD1_up_train, vol_avg_price_FD1_up)
    FD2_up_accept_price = construct_acceptance(FD2_up_train, vol_avg_price_FD2_up)

    #Incorporate both price acceptance and volume acceptance
    # FD1_down_accept = FD1_down_accept_price.*FD1_down_percentage
    # FD2_down_accept = FD2_down_accept_price.*FD2_down_percentage
    # FD1_up_accept = FD1_up_accept_price.*FD1_up_percentage
    # FD2_up_accept = FD2_up_accept_price.*FD2_up_percentage

    #Incorporate only volume acceptance
    FD1_down_accept = FD1_down_percentage
    FD2_down_accept = FD2_down_percentage
    FD1_up_accept = FD1_up_percentage
    FD2_up_accept = FD2_up_percentage

    Data = Dict("T" => [i for i in 1:24],
                "size_W1" => W1,
                "size_W2" => W2,
                "size_W3" => W3, 

                # Saving all the accepted mean values (not the best way)
        
                # All the training data
                # "FD2_up_train_all" => FD2_up_train,  
                # "FD2_up_accept_all" => FD2_up_accept,
                # "FD2_dn_train_all" => FD2_down_train, 
                # "FD2_dn_accept_all" => FD2_down_accept,
                # "spot_train_all" => spot_train,   
                # "FD1_up_train_all" => FD1_up_train,
                # "FD1_up_accept_all" => FD1_up_accept,
                # "FD1_dn_train_all" => FD1_down_train,
                # "FD1_dn_accept_all" => FD1_down_accept,

                
                # "FD2_up_train" => FD2_up_train[:,1:W1], # To be plotted
                # "FD2_dn_train" => FD2_down_train[:,1:W1],
                # "FD2_up_accept" => FD2_up_accept[:,1:W1],
                # "FD2_dn_accept" => FD2_down_accept[:,1:W1],
                # "spot_train" => spot_train[:,1:W1*W2],
                # "FD1_up_train" => FD1_up_train[:,1:W1*W2*W3],
                # "FD1_dn_train" => FD1_down_train[:,1:W1*W2*W3],
                # "FD1_up_accept" => FD1_up_accept[:,1:W1*W2*W3],
                # "FD1_dn_accept" => FD1_down_accept[:,1:W1*W2*W3],
                # "FD_act_up_train" => FD_act_up_train[:,1:W1*W2*W3],
                # "FD_act_dn_train" => FD_act_down_train[:,1:W1*W2*W3],

                # Forecasted prices 
                "f_FD1_dn" => vol_avg_price_FD1_down,
                "f_FD2_dn" => vol_avg_price_FD2_down,
                "f_FD1_up" => vol_avg_price_FD1_up,
                "f_FD2_up" => vol_avg_price_FD2_up,
                "f_Spot_price" => f_Spot_price,

                "f_DA_t" =>f_Spot_price,
                "f_FD1_up_t" => vol_avg_price_FD1_up, 
                "f_FD2_up_t" => vol_avg_price_FD2_up, 
                "f_FD1_dn_t" => vol_avg_price_FD1_down,
                "f_FD2_dn_t" => vol_avg_price_FD2_down,

                # The training data which is being provided to the model
                "f_FD2_up_tw" => FD2_up_train, 
                "f_FD2_dn_tw" => FD2_down_train,
                "f_FD2_y_up_tw" => FD2_up_accept, 
                "f_FD2_y_dn_tw" => FD2_down_accept,
                "f_DA_tw" => reshape(repeat(spot_train, no_days), (24,W1,W2)),
                "f_FD1_up_tw" => reshape(repeat(FD1_up_train, no_days*no_days), (24,W1,W2,W3)), 
                "f_FD1_dn_tw" => reshape(repeat(FD1_down_train, no_days*no_days), (24,W1,W2,W3)),
                "f_a_up_tw" => reshape(repeat(FD_act_up_train, no_days*no_days), (24,W1,W2,W3)), 
                "f_a_dn_tw" => reshape(repeat(FD_act_down_train, no_days*no_days), (24,W1,W2,W3)), 
                "f_FD1_y_up_tw" => reshape(repeat(FD1_up_accept, no_days*no_days), (24,W1,W2,W3)), 
                "f_FD1_y_dn_tw" => reshape(repeat(FD1_down_accept, no_days*no_days), (24,W1,W2,W3)),
                "SOC_0" => Data_Battery["SOC_0"], 
                "SOC_max" => Data_Battery["SOC_max"], 
                "eta_dis" => Data_Battery["eta_dis"],
                "eta_ch" => Data_Battery["eta_ch"],
                "p_dis_max" => Data_Battery["p_dis_max"],
                "p_ch_max" => Data_Battery["p_ch_max"], 
                "Cost_per_cycle" => Data_Battery["Cost_per_cycle"])
    @info("Data import for stochastic model complete!")
    return Data
end