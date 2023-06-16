#data import function for deterministic model

function data_import_Deterministic(forecast_data, Data_index)

    #Battery data
    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")
    SOC_0 = Data_Battery["SOC_0"]
    SOC_max = Data_Battery["SOC_max"]
    eta_dis = Data_Battery["eta_dis"]
    eta_ch = Data_Battery["eta_ch"]
    p_dis_max = Data_Battery["p_dis_max"]
    p_ch_max = Data_Battery["p_ch_max"]
    Cost_per_cycle = Data_Battery["Cost_per_cycle"]

    #Forecasts
    forecast_idx = Data_index["N_forecast_flat"]

    f_lambda_FD1_dn = reshape(forecast_data[forecast_idx, "FD1_down"], (24,1))
    f_lambda_FD2_dn = reshape(forecast_data[forecast_idx, "FD2_down"], (24,1))
    f_lambda_FD1_up   = reshape(forecast_data[forecast_idx, "FD1_up"], (24,1))
    f_lambda_FD2_up   = reshape(forecast_data[forecast_idx, "FD2_up"], (24,1))
    FD1_down_accept_price = reshape(forecast_data[forecast_idx, "FD1_down_percentage"], (24,1))
    FD2_down_accept_price = reshape(forecast_data[forecast_idx, "FD2_down_percentage"], (24,1))
    FD1_up_accept_price   = reshape(forecast_data[forecast_idx, "FD1_up_percentage"], (24,1))
    FD2_up_accept_price   = reshape(forecast_data[forecast_idx, "FD2_up_percentage"], (24,1))
    f_DA_t = reshape(forecast_data[forecast_idx, "Spot"], (24,1))
    f_a_up_t   = reshape(forecast_data[forecast_idx, "FD_act_up"], (24,1))
    f_a_dn_t   = reshape(forecast_data[forecast_idx, "FD_act_down"], (24,1))

    #Export for a selected forecast/test day
    Data = Dict("Time" => [i for i in 1:24], 
                "f_DA_t" => f_DA_t, 
                "f_FD1_up_t" => f_lambda_FD1_up, 
                "f_FD2_up_t" => f_lambda_FD2_up, 
                "f_FD1_dn_t" => f_lambda_FD1_dn,
                "f_FD2_dn_t" => f_lambda_FD2_dn,
                "f_a_up_t" => f_a_up_t, 
                "f_a_dn_t" => f_a_dn_t, 
                "f_FD1_y_up_t" => FD1_up_accept_price, 
                "f_FD1_y_dn_t" => FD1_down_accept_price, 
                "f_FD2_y_up_t" => FD2_up_accept_price, 
                "f_FD2_y_dn_t" => FD2_down_accept_price, 
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