#data import function for deterministic model

function data_import_Deterministic(forecast_data, forecast_day_2023)

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
    D_forecast = floor(Int, size(forecast_data)[1]/24) #number of days for forecasts

    f_lambda_FD1_dn = reshape(forecast_data[:, "FD1_down"], (24,D_forecast))
    f_lambda_FD2_dn = reshape(forecast_data[:, "FD2_down"], (24,D_forecast))
    f_lambda_FD1_up   = reshape(forecast_data[:, "FD1_up"], (24,D_forecast))
    f_lambda_FD2_up   = reshape(forecast_data[:, "FD2_up"], (24,D_forecast))
    FD1_down_accept_price = reshape(forecast_data[:, "FD1_down_percentage"], (24,D_forecast))
    FD2_down_accept_price = reshape(forecast_data[:, "FD2_down_percentage"], (24,D_forecast))
    FD1_up_accept_price   = reshape(forecast_data[:, "FD1_up_percentage"], (24,D_forecast))
    FD2_up_accept_price   = reshape(forecast_data[:, "FD2_up_percentage"], (24,D_forecast))
    f_DA_t = reshape(forecast_data[:, "Spot"], (24,D_forecast))
    f_a_up_t   = reshape(forecast_data[:, "FD_act_up"], (24,D_forecast))
    f_a_dn_t   = reshape(forecast_data[:, "FD_act_down"], (24,D_forecast))

    #Export for a selected forecast/test day
    Data = Dict("Time" => [i for i in 1:24], 
                "f_DA_t" => f_DA_t[:,forecast_day_2023], 
                "f_FD1_up_t" => f_lambda_FD1_up[:,forecast_day_2023], 
                "f_FD2_up_t" => f_lambda_FD2_up[:,forecast_day_2023], 
                "f_FD1_dn_t" => f_lambda_FD1_dn[:,forecast_day_2023],
                "f_FD2_dn_t" => f_lambda_FD2_dn[:,forecast_day_2023],
                "f_a_up_t" => f_a_up_t[:,forecast_day_2023], 
                "f_a_dn_t" => f_a_dn_t[:,forecast_day_2023], 
                "f_FD1_y_up_t" => FD1_up_accept_price[:,forecast_day_2023], 
                "f_FD1_y_dn_t" => FD1_down_accept_price[:,forecast_day_2023], 
                "f_FD2_y_up_t" => FD2_up_accept_price[:,forecast_day_2023], 
                "f_FD2_y_dn_t" => FD2_down_accept_price[:,forecast_day_2023], 
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