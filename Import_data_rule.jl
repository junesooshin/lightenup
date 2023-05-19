#data import function for rule-based model

function data_import_Rule(forecast_data)

    Data_Battery = Battery_Specifications("Samsung_SDI_E3_R135_Battery")
    
    E_nom = Data_Battery["SOC_max"]
    SOC_0 = Data_Battery["SOC_0"]/E_nom
    p_dis_max = Data_Battery["p_dis_max"]
    p_ch_max = Data_Battery["p_ch_max"]
    Cost_per_cycle = Data_Battery["Cost_per_cycle"]

    D_forecast = floor(Int, size(forecast_data)[1]/24) #number of days for forecasts

    ### RESHAPE 
    Spot = reshape(forecast_data[:,"Spot"], (24, D_forecast))
    FD2_down = reshape(forecast_data[:,"FD2_down"], (24, D_forecast))
    FD2_up = reshape(forecast_data[:,"FD2_up"], (24, D_forecast))

    Data = Dict("Spot" => Spot, "FD2_down" => FD2_down, "FD2_up" => FD2_up, "Cost_per_cycle" => Cost_per_cycle,
                "SOC_0"=> SOC_0, "E_nom" => E_nom, "p_dis_max" => p_dis_max, "p_ch_max" => p_ch_max)
    return Data

end

