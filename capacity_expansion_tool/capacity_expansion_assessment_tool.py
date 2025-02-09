#This is the main script for the capacity expansion assessment tool. It reads the input data, runs the data processing and writes the output data. This output data is then read into the powerplants script of the optimization model building workflow.
# this code is absolutely not optimized - I want to apologize for that. I will do that as soon as I have the time.

#import modules
import sys
import os
import pandas as pd
import plotly.express as px
import math
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


print('Execute in Directory:')
print(os.getcwd() + "\n")

try:
    #use if run in spine-toolbox
    case_study_path                 = os.path.join(os.getcwd(), "capacity_expansion_tool")
    IRENA_dataset_path              = r".\capacity_expansion_tool\data\IRENA_Stats_Extract_ 2024_H1_V1.xlsx"
    area_potential_vre_path         = r".\capacity_expansion_tool\data\02_dim1_mitDennisWerten_ohneInvestmentCost_100_percent.csv"
    subset_countries_path           = r".\capacity_expansion_tool\data\subset_countries.csv"
except: 
    #use if run in Python environment
    if str(os.getcwd()).find('PythonScripts') > -1:
        os.chdir('..')
    case_study_path                 = os.path.join("capacity_expansion_tool")
    IRENA_dataset_path              = r"./data/IRENA_Stats_Extract_ 2024_H1_V1.xlsx"
    area_potential_vre_path         = r"./data/02_dim1_mitDennisWerten_ohneInvestmentCost_100_percent.csv"
    subset_countries_path           = r"./data/subset_countries.csv"


START = time.perf_counter() 

print("Start reading input Data" + "\n")

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

IRENA_df_country = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Country")
IRENA_df_region = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Regional")
IRENA_df_global = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Global")

area_potential_vre_df_read = pd.read_csv(os.path.join(area_potential_vre_path), sep=";")

subset_countries = pd.read_csv(os.path.join(subset_countries_path), sep=";")
subset_countries["iso_a3"] = subset_countries["Countries"].apply(lambda x: x.split("-")[-1])

###### Fetch the data from Our World in Data ######

electricity_mix_owid_df = pd.read_csv("https://ourworldindata.org/grapher/electricity-prod-source-stacked.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
# Fetch the metadata
metadata = requests.get("https://ourworldindata.org/grapher/electricity-prod-source-stacked.metadata.json?v=1&csvType=full&useColumnShortNames=true").json()

electricity_mix_owid_df["Unit"] = "TWh"
electricity_mix_owid_df = electricity_mix_owid_df.rename(columns={"other_renewables_excluding_bioenergy_generation__twh_chart_electricity_prod_source_stacked": "Other Renewables",
                                                                    "bioenergy_generation__twh_chart_electricity_prod_source_stacked": "Bioenergy",
                                                                    "solar_generation__twh_chart_electricity_prod_source_stacked": "Solar",
                                                                    "wind_generation__twh_chart_electricity_prod_source_stacked": "Wind",
                                                                    "nuclear_generation__twh_chart_electricity_prod_source_stacked": "Nuclear",
                                                                    "gas_generation__twh_chart_electricity_prod_source_stacked": "Gas",
                                                                    "coal_generation__twh_chart_electricity_prod_source_stacked": "Coal",
                                                                    "oil_generation__twh_chart_electricity_prod_source_stacked": "Oil",
                                                                    "hydro_generation__twh_chart_electricity_prod_source_stacked": "Hydro"})

#select all row without Code
el_mix_regions = electricity_mix_owid_df[electricity_mix_owid_df["Code"].isnull()]
#select all rows with Code
el_mix_countries = electricity_mix_owid_df[electricity_mix_owid_df["Code"].notnull()]

start_year = el_mix_countries["Year"].min()
end_year = el_mix_countries["Year"].max()

##### prepare capacity data #####

#First method of extrapolation: Logistic function with manual fit
#Trying to loop the code over all countries to get the logistic function for each country

IRENA_df_country = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Country")
IRENA_df_region = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Regional")
IRENA_df_global = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Global")

area_potential_vre_df_read = pd.read_csv(os.path.join(area_potential_vre_path), sep=";")
area_potential_vre_df = area_potential_vre_df_read.copy()

area_potential_vre_df['Countries'] = area_potential_vre_df['Object_names'].astype(str).str.split('|',expand=True)[2]
area_potential_vre_df['unit'] = area_potential_vre_df['Object_names'].astype(str).str.split('|',expand=True)[1]
area_potential_vre_df = area_potential_vre_df.drop(columns=["Object_class", "Object_names", "Parameter_name", "Alternative"])

#rename unit by string PV to solar
i = 1
imax = 6
for i in range(1, imax):
    area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace(str(i-1), "")

area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("PV", "Solar photovoltaic")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("WIND_OFFSHORE", "Offshore wind energy")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("WIND", "Onshore wind energy")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("Invest", "")
area_potential_vre_df = area_potential_vre_df.rename(columns={"unit": "Technology"})

#aggregate data by country and unit
area_potential_vre_df = area_potential_vre_df.groupby(['Countries','Technology']).sum().reset_index()

region_list = IRENA_df_country["Region"].unique().tolist()
country_list = subset_countries["iso_a3"].unique().tolist()
technology_list = ["Onshore wind energy", "Offshore wind energy", "Solar photovoltaic"] #relevant for looping

capacities_df = IRENA_df_country[IRENA_df_country["ISO3 code"].isin(country_list)]
capacities_df = capacities_df.drop(columns=["M49 code", "Sub-Technology", "Producer Type", "RE or Non-RE", "Group Technology", "Region"]).reset_index(drop=True)
capacities_df = capacities_df.groupby(["ISO3 code", "Year", "Technology"]).agg({"Electricity Installed Capacity (MW)": "sum"}).reset_index()

capacities_df_c = capacities_df.copy()
capacities_df_c = capacities_df_c[capacities_df_c["Technology"].isin(["Solar photovoltaic", "Onshore wind energy", "Offshore wind energy"])]

growth_analysis_df = pd.DataFrame()

#loop for countries
for iso in capacities_df_c["ISO3 code"].unique():
    capacities_df_c_iso = capacities_df_c.loc[capacities_df_c["ISO3 code"] == iso]
    for tech in capacities_df_c_iso["Technology"].unique():
        capacities_df_c_iso_tech = capacities_df_c_iso.loc[capacities_df_c_iso["Technology"] == tech].copy()
        capacities_df_c_iso_tech.loc[:, "capacity_added"] = capacities_df_c_iso_tech["Electricity Installed Capacity (MW)"].diff()
        capacities_df_c_iso_tech.loc[:, "rel_capacity_added"] = capacities_df_c_iso_tech["capacity_added"]/capacities_df_c_iso_tech["Electricity Installed Capacity (MW)"]
        #print(capacities_df_c_iso_tech)

        growth_analysis_df = pd.concat([growth_analysis_df, capacities_df_c_iso_tech], ignore_index=True)

#filter for 2023 
growth_analysis_df = growth_analysis_df[growth_analysis_df["Year"] == 2023]

growth_analysis_df_agg = growth_analysis_df.groupby("Technology").agg({"rel_capacity_added": "mean"}).reset_index()
#mean_growth_df = mean_growth_df.concat([mean_growth_df, growth_analysis_df_tech], ignore_index=True)

##### prepare results data, curve fitting and extrapolation for kickstart case #####

#Second method of extrapolation: Logistic function with curve_fit function
#Trying to loop the code over all countries to get the logistic function for each country

IRENA_df_country = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Country")

area_potential_vre_df_read = pd.read_csv(os.path.join(area_potential_vre_path), sep=";")

#extrapolation parameters
#learning rate power densitiy [W/m²]
learning_rate = 0.005
#Planning delay [years]
planning_delay = 5
#time horizon
time_horizon = 2061
#Area feasibility factor
area_feasibility_factor = 0.3 #according to the area potential analysis

#build power density dict
power_density_dict = {"Solar photovoltaic": 21.457, "Offshore wind energy": 7.58, "Onshore wind energy": 5.821}

tech_area_f_factor_PV = 0.25
tech_area_f_factor_OFFWIND = 0.11
tech_area_f_factor_ONWIND = 0.18

#write dataframe with technology and area feasibility factor
tech_area_f_df = pd.DataFrame({"Technology": ["Solar photovoltaic", "Offshore wind energy", "Onshore wind energy"], "Area Feasibility Factor": [tech_area_f_factor_PV, tech_area_f_factor_OFFWIND, tech_area_f_factor_ONWIND]})

area_potential_vre_df = area_potential_vre_df_read.copy()

area_potential_vre_df['Countries'] = area_potential_vre_df['Object_names'].astype(str).str.split('|',expand=True)[2]
area_potential_vre_df['unit'] = area_potential_vre_df['Object_names'].astype(str).str.split('|',expand=True)[1]
area_potential_vre_df = area_potential_vre_df.drop(columns=["Object_class", "Object_names", "Parameter_name", "Alternative"])

#rename unit by string PV to solar
imax = 6
for i in range(1, imax):
    area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace(str(i-1), "")

area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("PV", "Solar photovoltaic")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("WIND_OFFSHORE", "Offshore wind energy")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("WIND", "Onshore wind energy")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("Invest", "")
area_potential_vre_df = area_potential_vre_df.rename(columns={"unit": "Technology"})

#aggregate data by country and unit
area_potential_vre_df = area_potential_vre_df.groupby(['Countries','Technology']).sum().reset_index()

region_list = IRENA_df_country["Region"].unique().tolist()
country_list = subset_countries["iso_a3"].unique().tolist()
technology_list = ["Onshore wind energy", "Offshore wind energy", "Solar photovoltaic"] #relevant for looping

capacities_df = IRENA_df_country[IRENA_df_country["ISO3 code"].isin(country_list)]
capacities_df = capacities_df.drop(columns=["M49 code", "Sub-Technology", "Producer Type", "RE or Non-RE", "Group Technology", "Region"]).reset_index(drop=True)
capacities_df = capacities_df.groupby(["ISO3 code", "Year", "Technology"]).agg({"Electricity Installed Capacity (MW)": "sum"}).reset_index()

results_df = pd.DataFrame()

# Define the logistic function
def logistic_function(x, L, k, x_0):
    return L / (1 + np.exp(-k * (x - x_0)))

# Define the logistic function with L as a fixed parameter
def logistic_function_fixed_L(x, k, x_0, L):
    return L / (1 + np.exp(-k * (x - x_0)))

for tech_element in technology_list:
    for country_element in country_list:
        sample_df = capacities_df[capacities_df["ISO3 code"] == country_element].reset_index(drop=True)
        sample_df = sample_df[sample_df["Technology"] == tech_element].reset_index(drop=True)

        sample_df.fillna(0, inplace=True)

        start_year = sample_df["Year"].min()
        last_year = sample_df["Year"].max()

        #capacity added
        sample_df["capacity_added"] = sample_df["Electricity Installed Capacity (MW)"].diff()
        sample_df["rel_capacity_added"] = sample_df["capacity_added"] / sample_df["Electricity Installed Capacity (MW)"]
        avg_growth_rate = sample_df["rel_capacity_added"].mean()

        #if rel_capacity_added.mean() in is smaller than average in growth_analysis_df_agg then use the average growth rate from growth_analysis_df_agg
        # if avg_growth_rate < growth_analysis_df_agg[growth_analysis_df_agg["Technology"] == tech_element]["rel_capacity_added"].iloc[0]:
        #     avg_growth_rate = growth_analysis_df_agg[growth_analysis_df_agg["Technology"] == tech_element]["rel_capacity_added"].iloc[0]
    
        #calculate the average growth rate for the whole geographic scope for the last year of the Electricity Installed Capacity (MW) column for each Technology 
        avg_growth_rate_df = sample_df[sample_df["Year"] == 2059].groupby("Technology").agg({"rel_capacity_added": "mean"}).reset_index()

        spec_area_pot_df = area_potential_vre_df[area_potential_vre_df["Countries"] == country_element]
        if spec_area_pot_df.empty or spec_area_pot_df[spec_area_pot_df['Technology'] == tech_element].empty:
            continue
        else:
            sample_df["area_potential"] = spec_area_pot_df[spec_area_pot_df['Technology'] == tech_element].iloc[0]['Paramter_value'] * 1000 * tech_area_f_df[tech_area_f_df["Technology"] == tech_element]["Area Feasibility Factor"].iloc[0]
        
        # Given data points
        x_data = np.array(sample_df["Year"])
        y_data = np.array(sample_df["Electricity Installed Capacity (MW)"])

        if len(x_data) == 0 or len(y_data) == 0:
            continue

        fixed_L = sample_df["area_potential"].max() * (1 + learning_rate)**(time_horizon-last_year)

        # Fit the logistic function to the data with fixed L
        try:
            popt, pcov = curve_fit(lambda x, k, x_0: logistic_function_fixed_L(x, k, x_0, fixed_L), x_data, y_data, p0=[avg_growth_rate, (time_horizon + start_year)/2])
        except RuntimeError:
            print(f"Fit could not be found for {tech_element} in {country_element}")
            continue

        # Extract the parameters
        k, x_0 = popt

        #k = avg_growth_rate führt wieder dazu, dass die Kurven nicht aneinander anschließen

        # Generate x values for extrapolation
        x_extrapolate = np.linspace(end_year, time_horizon, time_horizon-end_year)
        y_extrapolate = logistic_function_fixed_L(x_extrapolate, k, x_0, fixed_L) #k = growth rate

        #expand datarame until time horizon
        if math.isnan(sample_df["Year"].max()):
            continue
            years_to_add_list = list(range(2024, time_horizon))
        else:
            years_to_add_list = list(range(int(sample_df["Year"].max()), time_horizon))
            years_to_add_df = pd.DataFrame(years_to_add_list, columns=["Year"])
            years_to_add_df["Electricity Installed Capacity (MW)"] = np.nan
            years_to_add_df["capacity_added"] = np.nan
            years_to_add_df["Technology"] = tech_element
            years_to_add_df["ISO3 code"] = country_element
            years_to_add_df["log_expol"] = y_extrapolate
            years_to_add_df["area_potential"] = sample_df["area_potential"].iloc[0]
            years_to_add_df["growth_rate"] = k
            #combine the last year of the sample_df with the first year of the years_to_add_df
            years_to_add_df.loc[0, "Electricity Installed Capacity (MW)"] = sample_df["Electricity Installed Capacity (MW)"].iloc[-1]
            years_to_add_df.loc[0, "capacity_added"] = sample_df["capacity_added"].iloc[-1]
            #drop last year of sample_df
            sample_df = sample_df.drop(sample_df[sample_df["Year"] == sample_df["Year"].max()].index)
            sample_df = pd.concat([sample_df, years_to_add_df], ignore_index=True)

        #logistic added capacity
        sample_df["log_expol_cap_add"] = sample_df["log_expol"].diff()

        #account for decomissioning
        #substract the logistic capacity added 20 years ago from the logistic value
        # use new pandas function definition pattern: df.loc[row_indexer, "col"] = values
        sample_df["log_expol_decom"] = np.nan
        i = 0
        for i in sample_df.index:
            if i < 21:
                sample_df.loc[i, "log_expol_decom"] = sample_df["log_expol"].iloc[i]
            if i >= 21:
                sample_df.loc[i, "log_expol_decom"] = sample_df["log_expol"].iloc[i] - sample_df["log_expol_cap_add"].iloc[i-20]

        sample_df["tech_potential"] = sample_df["area_potential"] + (sample_df["Year"]-last_year)*sample_df["area_potential"]*learning_rate #learning rate of 0.5% per year
        
        #if log_expol_decom > tech_potential, use tech_potential as new value for log_expol_decom
        sample_df.loc[sample_df["log_expol"] >= sample_df["tech_potential"], "log_expol"] = sample_df["tech_potential"]

        results_df = pd.concat([results_df, sample_df], ignore_index=True).drop_duplicates()

#insert geographical aggregation using the subset_countries dataframe



results_df_c = results_df.copy()

##### prepare results data, curve fitting and extrapolation for kickstart case #####

### curve_fit from scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Option with average historic values as start k
#Second method of extrapolation: Logistic function with curve_fit function
#Trying to loop the code over all countries to get the logistic function for each country

IRENA_df_country = pd.read_excel(os.path.join(IRENA_dataset_path), sheet_name="Country")

area_potential_vre_df_read = pd.read_csv(os.path.join(area_potential_vre_path), sep=";")

#extrapolation parameters
#learning rate power densitiy [W/m²]
learning_rate = 0.005
#Planning delay [years]
planning_delay = 5
#time horizon
time_horizon = 2061
#Area feasibility factor
area_feasibility_factor = 0.3 #according to the area potential analysis

tech_area_f_factor_PV = 0.25
tech_area_f_factor_OFFWIND = 0.11
tech_area_f_factor_ONWIND = 0.18

#write dataframe with technology and area feasibility factor
tech_area_f_df = pd.DataFrame({"Technology": ["Solar photovoltaic", "Offshore wind energy", "Onshore wind energy"], "Area Feasibility Factor": [tech_area_f_factor_PV, tech_area_f_factor_OFFWIND, tech_area_f_factor_ONWIND]})

area_potential_vre_df = area_potential_vre_df_read.copy()

area_potential_vre_df['Countries'] = area_potential_vre_df['Object_names'].astype(str).str.split('|',expand=True)[2]
area_potential_vre_df['unit'] = area_potential_vre_df['Object_names'].astype(str).str.split('|',expand=True)[1]
area_potential_vre_df = area_potential_vre_df.drop(columns=["Object_class", "Object_names", "Parameter_name", "Alternative"])

#rename unit by string PV to solar
imax = 6
for i in range(1, imax):
    area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace(str(i-1), "")

area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("PV", "Solar photovoltaic")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("WIND_OFFSHORE", "Offshore wind energy")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("WIND", "Onshore wind energy")
area_potential_vre_df['unit'] = area_potential_vre_df['unit'].str.replace("Invest", "")
area_potential_vre_df = area_potential_vre_df.rename(columns={"unit": "Technology"})

#aggregate data by country and unit
area_potential_vre_df = area_potential_vre_df.groupby(['Countries','Technology']).sum().reset_index()

region_list = IRENA_df_country["Region"].unique().tolist()
country_list = subset_countries["iso_a3"].unique().tolist()
technology_list = ["Onshore wind energy", "Offshore wind energy", "Solar photovoltaic"] #relevant for looping

capacities_df = IRENA_df_country[IRENA_df_country["ISO3 code"].isin(country_list)]
capacities_df = capacities_df.drop(columns=["M49 code", "Sub-Technology", "Producer Type", "RE or Non-RE", "Group Technology", "Region"]).reset_index(drop=True)
capacities_df = capacities_df.groupby(["ISO3 code", "Year", "Technology"]).agg({"Electricity Installed Capacity (MW)": "sum"}).reset_index()

results_df = pd.DataFrame()

# Define the logistic function
def logistic_function(x, L, k, x_0):
    return L / (1 + np.exp(-k * (x - x_0)))

# Define the logistic function with L as a fixed parameter
def logistic_function_fixed_L(x, k, x_0, L):
    return L / (1 + np.exp(-k * (x - x_0)))

for tech_element in technology_list:
    for country_element in country_list:
        sample_df = capacities_df[capacities_df["ISO3 code"] == country_element].reset_index(drop=True)
        sample_df = sample_df[sample_df["Technology"] == tech_element].reset_index(drop=True)

        sample_df.fillna(0, inplace=True)

        start_year = sample_df["Year"].min()
        last_year = sample_df["Year"].max()

        #capacity added
        sample_df["capacity_added"] = sample_df["Electricity Installed Capacity (MW)"].diff()
        sample_df["rel_capacity_added"] = sample_df["capacity_added"] / sample_df["Electricity Installed Capacity (MW)"]
        avg_growth_rate = sample_df["rel_capacity_added"].mean()

        #if rel_capacity_added.mean() in is smaller than average in growth_analysis_df_agg then use the average growth rate from growth_analysis_df_agg
        if avg_growth_rate < growth_analysis_df_agg[growth_analysis_df_agg["Technology"] == tech_element]["rel_capacity_added"].iloc[0]:
            avg_growth_rate = growth_analysis_df_agg[growth_analysis_df_agg["Technology"] == tech_element]["rel_capacity_added"].iloc[0]

        last_year_capacity = sample_df.loc[sample_df["Year"] == last_year, "Electricity Installed Capacity (MW)"].values
        if len(last_year_capacity) > 0:
            # Set minimum start capacity to 100 MW in 2023
            if last_year_capacity[0] <= 100:
                print("here: " + str(tech_element) + " " + str(country_element))
                #replace the installed capacities in the last 5 years with the average growth rate ending at 100 MW
                sample_df.loc[sample_df["Year"] >= last_year-5, "Electricity Installed Capacity (MW)"] = 100
                sample_df.loc[sample_df["Year"] >= last_year-4, "Electricity Installed Capacity (MW)"] = 110
                sample_df.loc[sample_df["Year"] >= last_year-3, "Electricity Installed Capacity (MW)"] = 130
                sample_df.loc[sample_df["Year"] >= last_year-2, "Electricity Installed Capacity (MW)"] = 160
                sample_df.loc[sample_df["Year"] >= last_year-1, "Electricity Installed Capacity (MW)"] = 200
                sample_df.loc[sample_df["Year"] == last_year, "Electricity Installed Capacity (MW)"] = 250

        spec_area_pot_df = area_potential_vre_df[area_potential_vre_df["Countries"] == country_element]
        if spec_area_pot_df.empty or spec_area_pot_df[spec_area_pot_df['Technology'] == tech_element].empty:
            continue
        else:
            sample_df["area_potential"] = spec_area_pot_df[spec_area_pot_df['Technology'] == tech_element].iloc[0]['Paramter_value'] * 1000 * tech_area_f_df[tech_area_f_df["Technology"] == tech_element]["Area Feasibility Factor"].iloc[0]
        
        # Given data points
        x_data = np.array(sample_df["Year"])
        y_data = np.array(sample_df["Electricity Installed Capacity (MW)"])

        if len(x_data) == 0 or len(y_data) == 0:
            continue

        fixed_L = sample_df["area_potential"].max() * (1 + learning_rate)**(time_horizon-last_year)

        # Fit the logistic function to the data with fixed L
        try:
            popt, pcov = curve_fit(lambda x, k, x_0: logistic_function_fixed_L(x, k, x_0, fixed_L), x_data, y_data, p0=[avg_growth_rate, (time_horizon + start_year)/2])
        except RuntimeError:
            print(f"Fit could not be found for {tech_element} in {country_element}")
            continue

        # Extract the parameters
        k, x_0 = popt

        #k = avg_growth_rate führt wieder dazu, dass die Kurven nicht aneinander anschließen

        # Generate x values for extrapolation
        x_extrapolate = np.linspace(last_year, time_horizon, time_horizon-end_year)
        y_extrapolate = logistic_function_fixed_L(x_extrapolate, k, x_0, fixed_L) #k = growth rate

        #expand datarame until time horizon
        if math.isnan(sample_df["Year"].max()):
            continue
            years_to_add_list = list(range(2024, time_horizon))
        else:
            years_to_add_list = list(range(int(sample_df["Year"].max()), time_horizon))
            years_to_add_df = pd.DataFrame(years_to_add_list, columns=["Year"])
            years_to_add_df["Electricity Installed Capacity (MW)"] = np.nan
            years_to_add_df["capacity_added"] = np.nan
            years_to_add_df["Technology"] = tech_element
            years_to_add_df["ISO3 code"] = country_element
            years_to_add_df["log_expol"] = y_extrapolate
            years_to_add_df["area_potential"] = sample_df["area_potential"].iloc[0]
            years_to_add_df["growth_rate"] = k
            #combine the last year of the sample_df with the first year of the years_to_add_df
            years_to_add_df.loc[0, "Electricity Installed Capacity (MW)"] = sample_df["Electricity Installed Capacity (MW)"].iloc[-1]
            years_to_add_df.loc[0, "capacity_added"] = sample_df["capacity_added"].iloc[-1]
            #drop last year of sample_df
            sample_df = sample_df.drop(sample_df[sample_df["Year"] == sample_df["Year"].max()].index)
            sample_df = pd.concat([sample_df, years_to_add_df], ignore_index=True)

        #logistic added capacity
        sample_df["log_expol_cap_add"] = sample_df["log_expol"].diff()

        #account for decomissioning
        #substract the logistic capacity added 20 years ago from the logistic value
        # use new pandas function definition pattern: df.loc[row_indexer, "col"] = values
        sample_df["log_expol_decom"] = np.nan
        i = 0
        for i in sample_df.index:
            if i < 21:
                sample_df.loc[i, "log_expol_decom"] = sample_df["log_expol"].iloc[i]
            if i >= 21:
                sample_df.loc[i, "log_expol_decom"] = sample_df["log_expol"].iloc[i] - sample_df["log_expol_cap_add"].iloc[i-20]

        sample_df["tech_potential"] = sample_df["area_potential"] + (sample_df["Year"]-last_year)*sample_df["area_potential"]*learning_rate #learning rate of 0.5% per year

        #if log_expol_decom > tech_potential, use tech_potential as new value for log_expol_decom
        sample_df.loc[sample_df["log_expol"] >= sample_df["tech_potential"], "log_expol"] = sample_df["tech_potential"]

        results_df = pd.concat([results_df, sample_df], ignore_index=True).drop_duplicates()

results_df_kickstart = results_df.copy()

##### geographical aggregation of the results #####

#geographical aggregation of results_df using subset_countries
#map subset_countries regions to results_df according to ISO3 code
results_df_geo = results_df_c.copy()
results_df_geo = results_df_geo.merge(subset_countries[["iso_a3", "Regions"]], left_on="ISO3 code", right_on="iso_a3", how="left")
results_df_geo = results_df_geo.drop(columns=["iso_a3"])
results_df_geo = results_df_geo.groupby(["Regions", "Year", "Technology"]).agg({"Electricity Installed Capacity (MW)": "sum", "log_expol": "sum", "area_potential": "sum", "tech_potential": "sum"}).reset_index()
#set 0 values to nan
results_df_geo = results_df_geo.replace(0, np.nan)
results_df_c = results_df_geo.copy()

results_df_geo = results_df_kickstart.copy()
results_df_geo = results_df_geo.merge(subset_countries[["iso_a3", "Regions"]], left_on="ISO3 code", right_on="iso_a3", how="left")
results_df_geo = results_df_geo.drop(columns=["iso_a3"])
results_df_geo = results_df_geo.groupby(["Regions", "Year", "Technology"]).agg({"Electricity Installed Capacity (MW)": "sum", "log_expol": "sum", "area_potential": "sum", "tech_potential": "sum"}).reset_index()
#set 0 values to nan
results_df_geo = results_df_geo.replace(0, np.nan)
results_df_kickstart = results_df_geo.copy()

country_list = subset_countries["Regions"].unique().tolist()

##### export the results to csv and xlsx #####

#export to csv
results_df_c.to_csv(os.path.join(case_study_path, "results", "capacity_expansion_limits.csv"), index=False, sep=";")
results_df_kickstart.to_csv(os.path.join(case_study_path, "results", "capacity_expansion_kickstart.csv"), index=False, sep=";")
#export to xlsx - Create an ExcelWriter object
with pd.ExcelWriter(os.path.join(case_study_path, "results", "capacity_expansion_limits.xlsx")) as writer:
    # Write each dataframe to a different sheet
    results_df_c.to_excel(writer, index=False, sheet_name="cap_exp_lim")
    results_df_kickstart.to_excel(writer, index=False, sheet_name="cap_exp_lim_kickstart")

##### visualize the results #####

# Assuming results_df and country_list are defined in a cell below
# technology_select = "Onshore wind energy"
# technology_select = "Offshore wind energy"
# technology_select = "Solar photovoltaic"

results_plot_df = results_df_geo

# Drop countries from results_df where the tech_potential is 0 or NaN
no_pot_country_list = results_plot_df[(results_plot_df["area_potential"] <= 0)]["Regions"].unique()
country_list = [x for x in country_list if x not in no_pot_country_list]
print(no_pot_country_list)
results_plot_df = results_plot_df[results_plot_df["area_potential"] > 0]

color_dict_tech_cap = {"Onshore wind energy": "cornflowerblue", "Offshore wind energy": "darkblue", "Solar photovoltaic": "orange"}
color_dict_tech_rep = {"Onshore wind energy": "cornflowerblue", "Offshore wind energy": "darkblue", "Solar photovoltaic": "orange"}
color_dict_tech_pot = {"Onshore wind energy": "cornflowerblue", "Offshore wind energy": "darkblue", "Solar photovoltaic": "orange"}

# Apply colors to results_df
results_plot_df["color_cap"] = results_plot_df["Technology"].map(color_dict_tech_cap)
results_plot_df["color_rep"] = results_plot_df["Technology"].map(color_dict_tech_rep)
results_plot_df["color_pot"] = results_plot_df["Technology"].map(color_dict_tech_pot)

num_countries = len(country_list)
cols_no = 4
rows_no = math.ceil(num_countries / cols_no)
he_wi = cols_no * 400
font_size = cols_no * 4 * 0.8
counter = 0

# Create figure with secondary y-axis
fig = make_subplots(rows=rows_no, cols=cols_no, subplot_titles=country_list)

for i in range(rows_no):
    for j in range(cols_no):
        if counter >= num_countries:
            break
        country = country_list[counter]
        for tech in results_plot_df["Technology"].unique():
            plot_df = results_plot_df[(results_plot_df["Regions"] == country) & (results_plot_df["Technology"] == tech)]

            if not plot_df.empty:
                # Add subplot for capacity installed
                fig.add_trace(go.Scatter(
                    x=plot_df["Year"],
                    y=plot_df["Electricity Installed Capacity (MW)"],
                    mode="lines",
                    name="Capacity Installed [MW]",
                    line=dict(color=plot_df["color_cap"].iloc[0]),
                    marker=dict(size=font_size / 2)
                ), row=i + 1, col=j + 1, secondary_y=False)

                fig.add_trace(go.Scatter(
                    x=plot_df["Year"],
                    y=plot_df["log_expol"],
                    mode="lines",
                    name="Logistic Values [MW]",
                    line=dict(dash="dash", color=plot_df["color_pot"].iloc[0]),
                    marker=dict(size=font_size / 2)
                ), row=i + 1, col=j + 1, secondary_y=False)

                fig.add_trace(go.Scatter(
                    x=plot_df["Year"],
                    y=plot_df["tech_potential"],
                    mode="lines",
                    name="Technical potential [MW]",
                    line=dict(dash="dot", color=plot_df["color_rep"].iloc[0]),
                    marker=dict(size=font_size / 2)
                ), row=i + 1, col=j + 1, secondary_y=False)

                first_time_step = 2023
                first_tech_potential = plot_df[plot_df["Year"] == first_time_step]["tech_potential"]
                if not first_tech_potential.empty:
                    first_tech_potential_value = first_tech_potential.iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[first_time_step],
                        y=[first_tech_potential_value],
                        mode="markers",
                        name="Technical Potential t0 [MW]",
                        marker=dict(size=font_size / 2, color="red")
                    ), row=i + 1, col=j + 1, secondary_y=False)

        counter += 1

# Update and unify legend
fig.for_each_trace(lambda trace: trace.update(showlegend=False))

# Set unified legend
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(size=font_size), showlegend=True, name="Capacity Installed [MW]", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(size=font_size), showlegend=True, name="Repowering Capacity [MW]", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(size=font_size), showlegend=True, name="Technical Potential t0 [MW]", line=dict(color="red")), row=1, col=1)

# Set y-axes titles and make
fig.update_yaxes(title_text="Capacity [MW]", secondary_y=False, title_font=dict(size=font_size))
fig.update_yaxes(title_text="Added Capacity [MW/a]", secondary_y=True, title_font=dict(size=font_size))

# Set y-axes titles and make y-axis logarithmic
for i in range(rows_no):
    for j in range(cols_no):
        if j == 0:  # Only set y-axis title for the first column
            fig.update_yaxes(title_text="Capacity [MW]", secondary_y=False, title_font=dict(size=font_size * 1.5), row=i + 1, col=j + 1)  # type="log",
            fig.update_yaxes(title_text="Added Capacity [MW/a]", secondary_y=True, title_font=dict(size=font_size), row=i + 1, col=j + 1)
        else:
            fig.update_yaxes(title_text="", linewidth=1, linecolor='lightgrey', mirror=True, showgrid=True, gridcolor = "lightgrey", title_font=dict(size=font_size), row=i + 1, col=j + 1)
            fig.update_yaxes(title_text="", secondary_y=True, row=i + 1, col=j + 1)

# Set x-axis title conditionally
for i in range(rows_no):
    for j in range(cols_no):
        if i == rows_no - 1:  # Only set x-axis title for the last row
            fig.update_xaxes(title_text="Year", linewidth=1, linecolor='lightgrey', mirror=True, showgrid=True, gridcolor = "lightgrey", title_font=dict(size=font_size), row=i + 1, col=j + 1)
        else:
            fig.update_xaxes(title_text="", linewidth=1, linecolor='lightgrey', mirror=True, showgrid=True, gridcolor = "lightgrey", title_font=dict(size=font_size), row=i + 1, col=j + 1)

#update and unify legend
fig.for_each_trace(lambda trace: trace.update(showlegend=False))

#update legend by defining the colors for the respective technologies instead of naming the column names
#set unified legend
for com in results_plot_df["Technology"].unique():
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', showlegend=True, name=com, line=dict(color=color_dict_tech_cap[com])))

fig.update_layout(height=he_wi * 2, width=he_wi * 0.8, title="Capacity limit extrapolation",
                  font=dict(size=font_size*1.5, family="Times New Roman", color="black"),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  plot_bgcolor='rgba(0,0,0,0)')

# Export to png
fig.write_image(os.path.join(case_study_path, "results", "capacity_extrapolation.png"), scale=4)
#fig.show()

##### detail view for Germany #####

plot_df = results_df.copy()
plot_df = plot_df[plot_df["ISO3 code"] == "DEU"]

#plot the results only for Germany
fig = go.Figure()

for tech in plot_df["Technology"].unique():
    plot_df_tech = plot_df[plot_df["Technology"] == tech]
    fig.add_scatter(x=plot_df_tech["Year"], y=plot_df_tech["Electricity Installed Capacity (MW)"]/1000, mode="lines", name=f"{tech} - Installed Capacity", line=dict(color=color_dict_tech_cap[tech]))
    fig.add_scatter(x=plot_df_tech["Year"], y=plot_df_tech["log_expol"]/1000, mode="lines", name=f"{tech} - Capacity limit", line=dict(dash="dash", color=color_dict_tech_pot[tech]))
    fig.add_scatter(x=plot_df_tech["Year"], y=plot_df_tech["tech_potential"]/1000, mode="lines", name=f"{tech} - Technical potential", line=dict(dash="dot", color=color_dict_tech_rep[tech]))
    #add dots for technical potential at t0
    fig.add_scatter(x=[2023], y=[plot_df_tech[plot_df_tech["Year"] == 2023]["tech_potential"].iloc[0]/1000], mode="markers", name=f"{tech} - Technical Potential t0", marker=dict(size=10, color="red"), showlegend=False)

fig.update_traces(line=dict(width=5))
fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, showgrid=True, gridcolor = "lightgrey", title="Year")
fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, showgrid=True, gridcolor = "lightgrey", zeroline=True, zerolinecolor='black', zerolinewidth=1, title="Capacity (GW)", range=[0, 500])

#add legend element dot with technical potential at t0
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color="red"), showlegend=True, name="Technical Potential t0"))

fig.update_layout(width=800, plot_bgcolor='rgba(0,0,0,0)', font=dict(size=20, family="Times New Roman", color="black"), title="DEU", title_x=0.5)
fig.write_image(os.path.join(case_study_path, "results", "capacity_extrapolation_DEU.png"), scale=2)

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')  