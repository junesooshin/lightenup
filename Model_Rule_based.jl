# Rule based model

# Provide a day of 24 hours of forecasted prices to get the rule based policy and action of the battery
# Assumed that battery can charge or discharge fully in one hour. Assumed that no activation occurs in FCR-D.
function baseline_model_DA_and_FCR_D(Data, forecast_day_2023, Threshold_Max_coef, Threshold_Min_coef,E_nom = 6,output_Capacity = 6,input_Capacity  = 6)

    # Input

    # 
    # FCR_D_Dn , Forecasted price [24,]
    # FCR_D_Up , Forecasted price [24,]
    # Spot     , Forecasted price [24,]
    # Threshold_Max_coef. Decide coefficient for upper bound for when Discharge should occur
    # Threshold_Min_coef. Decide lower bound for when charge should occur
    # E_nom = 6 # [MWh] Energy storage capacity for one unit (2020)
    # output_Capacity = 18 # [MW] discharge rate 
    # input_Capacity  = 3  # [MW] discharge rate

    # Output
    # SOC            , State of charge [24,]. Values are either 0 or 1       
    # Spot_action    , The actions in spot [24,]
    # FCR_D_Up_action, The actions in upregulation FCR-D [24,]
    # FCR_D_Dn_action, The actions in downregulation FCR-D [24,]
    # Threshold_Max  , Upperbound for when spot activates
    # Threshold_Min  , lowerbound for when spot activates

    FCR_D_Dn = Data["FD2_down"][:,forecast_day_2023]
    FCR_D_Up = Data["FD2_up"][:,forecast_day_2023]
    Spot     = Data["Spot"][:,forecast_day_2023]
    SOC_max = Data["E_nom"]
    Cost_per_cycle = Data["Cost_per_cycle"]

    # Initialize values
    SOC = zeros(24) # SOC can be [0,1]
    Spot_Dn = zeros(24) # Spot_action can be [0,1]
    Spot_Up = zeros(24) # Spot_action can be [0,1]
    FCR_D_Dn_action = zeros(24) # FCR_D_Dn_action can be [0,1]
    FCR_D_Up_action = zeros(24) # FCR_D_Up_action can be [0,1]

    Threshold_Max = Threshold_Max_coef*maximum(Spot)
    Threshold_Min = Threshold_Min_coef*minimum(Spot)
    
    # Initialize the first State of charge
    #SOC[1] = Initialize_SOC_baseline_model(FCR_D_Dn, FCR_D_Up, Spot) 
    SOC_ini = Data["SOC_0"]

    # if SOC[1] == 1 # Based on the initial state of charge set the FCR-D up/dn accordingly.
    #     FCR_D_Up_action[1] = 1
    # elseif SOC[1] == 0
    #     FCR_D_Dn_action[1] = 1
    # else
    #     FCR_D_Up_action[1] = SOC[1]
    #     FCR_D_Dn_action[1] = 1 - SOC[1]
    
    # end

    # Across the whole day, 24 hours
    for h in 1:24
        if h == 1
            soc_last_h = SOC_ini
        else
            soc_last_h = SOC[h-1]
        end
        # Do I have a full battery?
        if soc_last_h == 1
            # Then I can't do down regulation or charge battery. i.e. Spot_action[h] != -1 AND FCR_D_Dn_action != 1
            
            # Is the spot price larger than the max threshold?
            if Spot[h] >= Threshold_Max 
                # Then discharge, because it is profitable. i.e. Spot_action[h] = 1
                Spot_Up[h] = output_Capacity/E_nom  
                SOC[h] = soc_last_h - output_Capacity/E_nom  # How much the battery is able to discharge

            elseif Spot[h] < Threshold_Max 
                # Then just bid in upregulation and assume zero activation
                FCR_D_Up_action[h] = 1 # Bid only in upregulation
                SOC[h] = 1 # Remain the SOC
            end
        
        # Do I have a empty battery?
        elseif soc_last_h == 0
            # Then I can't do up regulation or discharge battery. i.e. Spot_action[h] != 1 AND FCR_D_up_action != 1
            
            # Is the spot price smaller than the min threshold
            if Spot[h] <= Threshold_Min
                # Then charge because it is beneficial. i.e. Spot_action[h] = -1
                Spot_Dn[h] = input_Capacity/E_nom  
                SOC[h] = soc_last_h + input_Capacity/E_nom  # How much the battery is able to charge

            elseif Spot[h] > Threshold_Min
                # Then just bid in Downregulation and assume zero activation
                FCR_D_Dn_action[h] = 1 # Bid only in downregulation
                SOC[h] = 0 # Remain the SOC
            end
        
        # Do I have an arbitrary amount in the battery
        elseif soc_last_h < 1 && soc_last_h > 0
            # Now I can do both Upregulation or DownRegulation equivalent to what is in my battery

            # Is the spot price smaller than the min threshold
            if Spot[h] <= Threshold_Min

                if input_Capacity/E_nom < 1- soc_last_h
                    Spot_Dn[h] = input_Capacity/E_nom
                elseif input_Capacity/E_nom >= 1- soc_last_h
                    Spot_Dn[h] = 1 - soc_last_h
                end
                SOC[h] = soc_last_h + abs(Spot_Dn[h]) # How much the battery is able to Charge
            
            # Is the spot price larger than the max threshold
            elseif Spot[h] >= Threshold_Max
                if output_Capacity/E_nom < soc_last_h
                    Spot_Up[h] = output_Capacity/E_nom
                elseif output_Capacity/E_nom >= soc_last_h
                    Spot_Up[h] = soc_last_h
                SOC[h] = soc_last_h - abs(Spot_Up[h])
                end
            # Else just do FCR-D up and down at the same time!!
            else
                FCR_D_Up_action[h] = soc_last_h # 
                FCR_D_Dn_action[h] = 1- soc_last_h # 
                SOC[h] = soc_last_h
            end
        end
    end

    #Calculate expected revenue
    G_DA_t = Spot.*E_nom.*(Spot_Up .- Spot_Dn) 
    G_FD2_t = FCR_D_Up.*E_nom.*FCR_D_Up_action.+FCR_D_Dn.*E_nom.*FCR_D_Dn_action
    C_Deg_t = (E_nom.*(Spot_Up .- Spot_Dn)) ./ (2*SOC_max) .* Cost_per_cycle  #Degradation cost only comes from DA contribution
    obj_t = G_DA_t + G_FD2_t - C_Deg_t

    Results = Dict("obj_t" => obj_t,
                   "f_DA_t" => Spot,
                   "SOC" => SOC, 
                   "b_DA_up" => Spot_Up*E_nom, 
                   "b_DA_dn" => Spot_Dn*E_nom, 
                   "b_FD2_up" => FCR_D_Up_action*E_nom, 
                   "b_FD2_dn" => FCR_D_Dn_action*E_nom, 
                   "Threshold_Max" => Threshold_Max*ones(24), 
                   "Threshold_Min" => Threshold_Min*ones(24),
                   "f_lambda_FD2_dn" => FCR_D_Dn, 
                   "f_lambda_FD2_up" => FCR_D_Up, 
                   "b_FD1_up" => zeros(24), #No participation in D-1
                   "b_FD1_dn" => zeros(24), #No participation in D-1
                   "f_lambda_FD1_up" => zeros(24), #No participation in D-1
                   "f_lambda_FD1_dn" => zeros(24) #No participation in D-1
                   )
    return Results
end


# Initialize and set the State of charge based on the 
function Initialize_SOC_baseline_model(FCR_D_Dn, FCR_D_Up, Spot)

    # Input:
    # FCR_D_Dn. Array of the FCR-D Down prices
    # FCR_D_Up. Array of the FCR-D Down prices
    # Spot    . Array of the spot prices

    # Output:
    # SOC_ini . Initial state of charge. Either 100 % or 0 % SOC.

    # Description of the rules
    # Check the relationship between the prices of Spot and FCR-D up/down. And set the state of charge accordingly

    
    if mean(Spot) > mean(FCR_D_Dn) && mean(Spot) > mean(FCR_D_Up)
        if argmin(Spot) <= argmax(Spot)
            SOC_ini = 0
        elseif argmin(Spot) > argmax(Spot)
            SOC_ini = 1
        end
    elseif mean(Spot) <= mean(FCR_D_Dn) || mean(Spot) <= mean(FCR_D_Up)
        if FCR_D_Dn[1] <= FCR_D_Up[1]
            SOC_ini = 1
        elseif FCR_D_Dn[1] > FCR_D_Up[1]
            SOC_ini = 0
        end
    end

    return SOC_ini
end










