#data import function for stochastic model

function data_import_stochastic(data_import, forecast_data, Data_index,gamma, W1, W2, W3,temporal = false,correlation = false,Acceptance = "",Model_configuration = "Without forecast in input")
    # Import parameters for stochastic model
    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")

    #Get training data from original df
    N_train_flat = Data_index["N_train_flat"]
    N_forecast_flat = Data_index["N_forecast_flat"]

    # Multiply prices with gamma
    subset = ["FD1_down", "FD2_down", "FD1_up", "FD2_up"]

    # Create a copy of the data_import array
    forecast_data_mod = copy(forecast_data)
    forecast_data_mod_prices_copy = copy(forecast_data[:,subset])
    forecast_data_mod_prices = forecast_data_mod_prices_copy.* gamma

    # Update the modified prices in the data_import_mod array
    forecast_data_mod[:, subset] = forecast_data_mod_prices

    df_train = data_import[N_train_flat, :]
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
    f_DA_t = reshape(forecast_data_mod[N_forecast_flat, "Spot"], (24, 1))
    f_lambda_FD1_dn_t = reshape(forecast_data_mod[N_forecast_flat, "FD1_down"], (24, 1))
    f_lambda_FD2_dn_t = reshape(forecast_data_mod[N_forecast_flat, "FD2_down"], (24, 1))
    f_lambda_FD1_up_t   = reshape(forecast_data_mod[N_forecast_flat, "FD1_up"], (24, 1))
    f_lambda_FD2_up_t   = reshape(forecast_data_mod[N_forecast_flat, "FD2_up"], (24, 1))
    f_act_up_t = reshape(forecast_data_mod[N_forecast_flat,"FD_act_up"], (24, 1))
    f_act_dn_t = reshape(forecast_data_mod[N_forecast_flat,"FD_act_down"], (24, 1))
    f_FD1_down_percentage = reshape(forecast_data_mod[N_forecast_flat,"FD1_down_percentage"], (24, 1))
    f_FD2_down_percentage = reshape(forecast_data_mod[N_forecast_flat,"FD2_down_percentage"], (24, 1))
    f_FD1_up_percentage = reshape(forecast_data_mod[N_forecast_flat,"FD1_up_percentage"], (24, 1))
    f_FD2_up_percentage = reshape(forecast_data_mod[N_forecast_flat,"FD2_up_percentage"], (24, 1))

    # Forecast price comes from the mean of the scenarios
    vol_avg_price_FD1_down = mean(FD1_down_train, dims=2)
    vol_avg_price_FD2_down = mean(FD2_down_train, dims=2)
    vol_avg_price_FD1_up = mean(FD1_up_train, dims=2)
    vol_avg_price_FD2_up = mean(FD2_up_train, dims=2)
    f_Spot_price = mean(spot_train, dims=2)



    ##################    Acceptance:     ###############################

    # Either Volume based, 100 % or based on acceptance construction 

    if Acceptance == "Junes"
        @info("June Acceptance in Stochastic")
        # 100 ACCEPTANCE FORECAST AND WEIGHTED FOR SAMPLE
        f_FD1_down_accept = ones(size(f_FD1_down_percentage))
        f_FD2_down_accept = ones(size(f_FD2_down_percentage))
        f_FD1_up_accept = ones(size(f_FD1_up_percentage))
        f_FD2_up_accept = ones(size(f_FD2_up_percentage))

        FD1_down_accept = construct_acceptance(FD1_down_train, f_lambda_FD1_dn_t)
        FD2_down_accept = construct_acceptance(FD2_down_train, f_lambda_FD2_dn_t)
        FD1_up_accept = construct_acceptance(FD1_up_train, f_lambda_FD1_up_t)
        FD2_up_accept = construct_acceptance(FD2_up_train, f_lambda_FD2_up_t)

    elseif Acceptance == "100"
        @info("100% Acceptance in Stochastic")
        # 100 % ACCEPTANCE
        f_FD1_down_accept = ones(size(f_FD1_down_percentage))
        f_FD2_down_accept = ones(size(f_FD2_down_percentage))
        f_FD1_up_accept = ones(size(f_FD1_up_percentage))
        f_FD2_up_accept = ones(size(f_FD2_up_percentage))

        FD1_down_accept = ones(size(FD1_down_percentage))
        FD2_down_accept = ones(size(FD2_down_percentage))
        FD1_up_accept = ones(size(FD1_up_percentage))
        FD2_up_accept = ones(size(FD2_up_percentage))
    else 
        @info("Volume Acceptance in Stochastic")
        # VOLUME BASED ACCEPTANCE:
        f_FD1_down_accept = f_FD1_down_percentage #Incorporate only volume acceptance
        f_FD2_down_accept = f_FD2_down_percentage
        f_FD1_up_accept = f_FD1_up_percentage
        f_FD2_up_accept = f_FD2_up_percentage

        FD1_down_accept = FD1_down_percentage    #Incorporate only volume acceptance
        FD2_down_accept = FD2_down_percentage
        FD1_up_accept = FD1_up_percentage
        FD2_up_accept = FD2_up_percentage
    end

    # Modify training if forecast is included in training for Learning models
    if Model_configuration == "With forecast in input"

        no_days = no_days + 1 # Add one more day as forecast is now added
        W1 = W1 + 1
        W2 = W2 + 1
        W3 = W3 + 1

        # Collect the sample data and the forecasted price
        spot_train = cat(spot_train, f_DA_t,dims=(2)) # Collect the spot
        FD1_down_train = cat(FD1_down_train, f_lambda_FD1_dn_t,dims=(2))
        FD2_down_train = cat(FD2_down_train, f_lambda_FD2_dn_t,dims=(2))
        FD1_up_train = cat(FD1_up_train, f_lambda_FD1_up_t,dims=(2))
        FD2_up_train = cat(FD2_up_train, f_lambda_FD2_up_t,dims=(2))
        FD_act_up_train = cat(FD_act_up_train, f_act_up_t,dims=(2))
        FD_act_down_train = cat(FD_act_down_train, f_act_dn_t,dims=(2))

        FD1_down_accept = cat(FD1_down_accept, f_FD1_down_accept,dims=(2))
        FD2_down_accept = cat(FD2_down_accept, f_FD2_down_accept,dims=(2))
        FD1_up_accept = cat(FD1_up_accept, f_FD1_up_accept,dims=(2))
        FD2_up_accept = cat(FD2_up_accept, f_FD2_up_accept,dims=(2))

        @info("Forecast added to input for optimization of the Stochastic model!")

    end

    # Create the probabilities for each scenario
    
    pi1 = zeros(W1)
    pi2 = zeros(W1,W2)
    pi3 = zeros(W1,W2,W3)

    #### Putting weights on it:
    if temporal == true
        @info("Temporal added in Stochastic")
        temporal_relation = [0.8, 0.4, 0.3, 0.2, 0.1, 0.8]
        temporal_relation = temporal_relation / sum(temporal_relation)

    else
        temporal_relation = ones(W1) .* 1/W1
    end
    if correlation == true
        @info("Correlation added in Stochastic")
        val_FD2_DA = (0.29 + 0.24) / 2
        val_DA_FD1 = (0.3 + 0.18) / 2
        val_FD2_FD1 = (0.6 + 0.49) / 2
    else
        val_FD2_DA = 1/W1
        val_DA_FD1 = 1/W2
        val_FD2_FD1 = 1/W3
    end

    FD2_DA = ones((W1,W2) )*val_FD2_DA
    DA_FD1 = ones((W2,W3) )*val_DA_FD1
    FD2_FD1 = ones((W1,W3) )*val_FD2_FD1
    
    # Construct pi1
    for fd2 in 1:W1
        pi1[fd2] = temporal_relation[fd2]
    end
    pi1 /= sum(pi1)

    # Construct pi2
    for fd2 in 1:W1
        for da in 1:W2  
            pi2[fd2,da] = FD2_DA[fd2, da] * temporal_relation[abs(fd2 - da)+1]
        end
    end
    pi2 /= sum(pi2)

    # Construct pi3
    for fd2 in 1:W1
        for da in 1:W2
            for fd1 in 1:W3
                fd2_da_prob = FD2_DA[fd2, da] * temporal_relation[abs(fd2 - da)+1]
                da_fd1_prob = DA_FD1[da, fd1] * temporal_relation[abs(fd1 - da)+1]
                fd2_fd1_prob = FD2_FD1[fd2, fd1] * temporal_relation[abs(fd2 - fd1)+1]
                pi3[fd2, da, fd1] = fd2_da_prob * da_fd1_prob * fd2_fd1_prob
            end
        end
    end
    pi3 /= sum(pi3)


    # Bid prices
    f_DA_t = f_DA_t
    f_FD1_up_t = f_lambda_FD1_up_t
    f_FD2_up_t = f_lambda_FD2_up_t
    f_FD1_dn_t = f_lambda_FD1_dn_t
    f_FD2_dn_t = f_lambda_FD2_dn_t
    
    Data = Dict("T" => [i for i in 1:24],
                "size_W1" => W1,
                "size_W2" => W2,
                "size_W3" => W3, 
                "pi1" => pi1,
                "pi2" => pi2,
                "pi3" => pi3,

                # Prices sent as bids from Stochastic 
                "f_FD1_dn" => vol_avg_price_FD1_down,
                "f_FD2_dn" => vol_avg_price_FD2_down,
                "f_FD1_up" => vol_avg_price_FD1_up,
                "f_FD2_up" => vol_avg_price_FD2_up,
                "f_Spot_price" => f_Spot_price,

                "f_DA_t" =>f_DA_t,
                "f_FD1_up_t" => f_FD1_up_t, 
                "f_FD2_up_t" => f_FD2_up_t, 
                "f_FD1_dn_t" => f_FD1_dn_t,
                "f_FD2_dn_t" => f_FD2_dn_t,

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