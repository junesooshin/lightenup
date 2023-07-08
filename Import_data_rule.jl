#data import function for rule-based model

function data_import_Rule(forecast_data,Data_index,gamma)

    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")
    
    E_nom = Data_Battery["SOC_max"]
    SOC_0 = Data_Battery["SOC_0"]/E_nom
    p_dis_max = Data_Battery["p_dis_max"]
    p_ch_max = Data_Battery["p_ch_max"]
    Cost_per_cycle = Data_Battery["Cost_per_cycle"]

    forecast_idx = Data_index["N_forecast_flat"]

    # Multiply prices with gamma
    subset = ["FD1_down", "FD2_down", "FD1_up", "FD2_up"]

    # Create a copy of the data_import array
    forecast_data_mod = copy(forecast_data)
    forecast_data_mod_prices_copy = copy(forecast_data[:, subset])
    forecast_data_mod_prices = forecast_data_mod_prices_copy.* gamma

    # Update the modified prices in the data_import_mod array
    forecast_data_mod[:, subset] = forecast_data_mod_prices

    ### RESHAPE 
    Spot = forecast_data_mod[forecast_idx,"Spot"]
    FD2_down = forecast_data_mod[forecast_idx,"FD2_down"]
    FD2_up = forecast_data_mod[forecast_idx,"FD2_up"]

    Data = Dict("Spot" => Spot, "FD2_down" => FD2_down, "FD2_up" => FD2_up, "Cost_per_cycle" => Cost_per_cycle,
                "SOC_0"=> SOC_0, "E_nom" => E_nom, "p_dis_max" => p_dis_max, "p_ch_max" => p_ch_max)
    return Data

end

