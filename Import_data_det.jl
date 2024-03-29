#data import function for deterministic model

function data_import_Deterministic(forecast_data, Data_index,gamma)

    #Battery data
    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")
    SOC_0 = Data_Battery["SOC_0"]
    SOC_max = Data_Battery["SOC_max"]
    eta_dis = Data_Battery["eta_dis"]
    eta_ch = Data_Battery["eta_ch"]
    p_dis_max = Data_Battery["p_dis_max"]
    p_ch_max = Data_Battery["p_ch_max"]
    Cost_per_cycle = Data_Battery["Cost_per_cycle"]



    # Multiply prices with gamma
    subset = ["FD1_down", "FD2_down", "FD1_up", "FD2_up"]

    # Create a copy of the data_import array
    forecast_data_mod = copy(forecast_data)
    forecast_data_mod_prices_copy = copy(forecast_data[:, subset])
    forecast_data_mod_prices = forecast_data_mod_prices_copy.* gamma

    # Update the modified prices in the data_import_mod array
    forecast_data_mod[:, subset] = forecast_data_mod_prices


    #Forecasts
    forecast_idx = Data_index["N_forecast_flat"]

    f_lambda_FD1_dn = forecast_data_mod[forecast_idx, "FD1_down"]
    f_lambda_FD2_dn = forecast_data_mod[forecast_idx, "FD2_down"]
    f_lambda_FD1_up   = forecast_data_mod[forecast_idx, "FD1_up"]
    f_lambda_FD2_up   = forecast_data_mod[forecast_idx, "FD2_up"]

    f_DA_t     = forecast_data_mod[forecast_idx, "Spot"]
    f_a_up_t   = forecast_data_mod[forecast_idx, "FD_act_up"]
    f_a_dn_t   = forecast_data_mod[forecast_idx, "FD_act_down"]

    f_FD1_down_percentage = forecast_data_mod[forecast_idx, "FD1_down_percentage"]
    f_FD2_down_percentage = forecast_data_mod[forecast_idx, "FD2_down_percentage"]
    f_FD1_up_percentage   = forecast_data_mod[forecast_idx, "FD1_up_percentage"]
    f_FD2_up_percentage   = forecast_data_mod[forecast_idx, "FD2_up_percentage"]

    if Acceptance == "Junes"
        # 100 ACCEPTANCE FORECAST AND WEIGHTED FOR SAMPLE
        f_FD1_down_accept = ones(size(f_FD1_down_percentage))
        f_FD2_down_accept = ones(size(f_FD2_down_percentage))
        f_FD1_up_accept = ones(size(f_FD1_up_percentage))
        f_FD2_up_accept = ones(size(f_FD2_up_percentage))

    elseif Acceptance == "100"
        # 100 % ACCEPTANCE
        f_FD1_down_accept = ones(size(f_FD1_down_percentage))
        f_FD2_down_accept = ones(size(f_FD2_down_percentage))
        f_FD1_up_accept = ones(size(f_FD1_up_percentage))
        f_FD2_up_accept = ones(size(f_FD2_up_percentage))

    else
        # VOLUME BASED ACCEPTANCE:
        f_FD1_down_accept = f_FD1_down_percentage #Incorporate only volume acceptance
        f_FD2_down_accept = f_FD2_down_percentage
        f_FD1_up_accept = f_FD1_up_percentage
        f_FD2_up_accept = f_FD2_up_percentage

    end


    #Export for a selected forecast/test day
    Data = Dict("Time" => [i for i in 1:24], 
                "f_DA_t" => f_DA_t, 
                "f_FD1_up_t" => f_lambda_FD1_up, 
                "f_FD2_up_t" => f_lambda_FD2_up, 
                "f_FD1_dn_t" => f_lambda_FD1_dn,
                "f_FD2_dn_t" => f_lambda_FD2_dn,
                "f_a_up_t" => f_a_up_t, 
                "f_a_dn_t" => f_a_dn_t, 
                "f_FD1_y_up_t" => f_FD1_up_accept, 
                "f_FD1_y_dn_t" => f_FD1_down_accept, 
                "f_FD2_y_up_t" => f_FD2_up_accept, 
                "f_FD2_y_dn_t" => f_FD2_down_accept, 
                "SOC_0" => SOC_0, 
                "SOC_max" => SOC_max, 
                "eta_dis" => eta_dis,
                "eta_ch" => eta_ch,
                "p_dis_max" => p_dis_max,
                "p_ch_max" => p_ch_max, 
                "Cost_per_cycle" => Cost_per_cycle)
    
    @info("Data import for deterministic model complete!")
    return Data
end