# Historic development assessment tool
# This is the main script for the historic development assessment tool. It reads the input data via the Our World in Data API, runs the data processing and writes the visualisation.
# Created 202501
# @author OL

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
import os
import time

print('Execute in Directory:')
print(os.getcwd() + "\n")

try:
    #use if run in spine-toolbox
    case_study_path                 = os.path.join(os.getcwd(), "capacity_expansion_tool")
except: 
    #use if run in Python environment
    if str(os.getcwd()).find('PythonScripts') > -1:
        os.chdir('..')
    case_study_path                 = os.path.join("capacity_expansion_tool")

START = time.perf_counter() 

# Fetch the data.
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

#transpose dataframe using the commodity columns to get a long format
el_mix_regions_long = el_mix_regions.melt(id_vars=["Entity", "Year", "Unit"], value_vars=["Coal", "Oil", "Gas", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other Renewables"], var_name="Commodity", value_name="Value")
el_mix_countries_long = el_mix_countries.melt(id_vars=["Entity", "Year", "Code", "Unit"], value_vars=["Coal", "Oil", "Gas", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other Renewables"], var_name="Commodity", value_name="Value")

results_df = el_mix_countries_long.copy()
#set values <= 0.1 to 0
results_df.loc[results_df["Value"] <= 0.1, "Value"] = 0
#drop rows with Commodity = "Other Renewables"
results_df = results_df[~results_df["Commodity"].isin(["Other Renewables", "Oil"])]
results_df = results_df.rename(columns={"Code": "ISO3 code", "Value": "Electricity Production (TWh)"})

# Countries with high growth rates during dataset period
el_mix_countries_agg = el_mix_countries.groupby(["Entity", "Code", "Year"]).agg({"Coal": "sum", "Oil": "sum", "Gas": "sum", "Nuclear": "sum", "Hydro": "sum", "Wind": "sum", "Solar": "sum", "Bioenergy": "sum", "Other Renewables": "sum"}).reset_index()
el_mix_countries_agg["Total"] = el_mix_countries_agg[["Coal", "Oil", "Gas", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy", "Other Renewables"]].sum(axis=1)
el_mix_countries_agg["Electricity cumsum"] = el_mix_countries_agg.groupby("Entity")["Total"].cumsum()
# calculate the growth rate by dividing the difference between the cumsum of the last year and the first year by the cumsum of the first year
el_mix_countries_agg["Growth rate"] = el_mix_countries_agg.groupby("Entity")["Electricity cumsum"].transform(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
# get the last year of the dataset
last_year = el_mix_countries_agg["Year"].max()
# filter for the last year
el_mix_countries_agg = el_mix_countries_agg[el_mix_countries_agg["Year"] == last_year]
#select all countries with total electricity production > 60 TWh
el_mix_countries_agg = el_mix_countries_agg[el_mix_countries_agg["Total"] > 60]
#drop world
el_mix_countries_agg = el_mix_countries_agg[el_mix_countries_agg["Entity"] != "World"]
# select al country Codes with growth rates >70% as country_list
country_list = el_mix_countries_agg[el_mix_countries_agg["Growth rate"] > 100]["Entity"].unique()
# merg Growth rate to results_df
results_df = results_df.merge(el_mix_countries_agg[["Entity", "Growth rate"]], on="Entity", how="left")
# select only countries in country_list
results_df = results_df[results_df["Entity"].isin(country_list)]
#results_df = results_df.sort_values("Growth rate", ascending=False)

rows_no = 3
cols_no = 4
he_wi = cols_no*400
font_size = cols_no*4
counter = 0

#create color dict for commodities
color_dict={"Solar":'gold','Solar old':'lemonchiffon', "Wind Onshore":"cornflowerblue", "Wind Offshore":"blue", "Wind": "blue", "Hydro":"darkblue",
            "Gas":"red", "Coal":"black", "Oil":"brown", "Biomass":"darkgreen", "Bioenergy":"darkgreen", "Nuclear":"purple", "Waste":"orange", "Electrolyzer":"turquoise", "Other":"grey", 'PHS':'grey','H2':'orange'}

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig = make_subplots(rows=rows_no, cols=cols_no,
                    subplot_titles=list(results_df["ISO3 code"].unique()))

for i in range(0, rows_no):
    for j in range(0, cols_no):
                       
        plot_df = results_df[results_df["Entity"] == country_list[counter]]

        for commodity in plot_df["Commodity"].unique():
            fig.add_scatter(x=plot_df["Year"],
                            y=plot_df[plot_df["Commodity"] == commodity]["Electricity Production (TWh)"],
                            mode="lines",
                            name=commodity,
                            #secondary_y=False,
                            marker=dict(size=font_size/2),
                            row=i+1, col=j+1,
                            opacity=0.4,
                            #set color
                            line=dict(color=color_dict[commodity], width=1)
            )

            # Fit a quadratic polynomial
            x = plot_df[plot_df["Commodity"] == commodity]["Year"]
            y = plot_df[plot_df["Commodity"] == commodity]["Electricity Production (TWh)"]
            coeffs = np.polyfit(x, y, 5)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = poly(x_fit)

            # Scatter plot for the quadratic fit
            fig.add_scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=f"{commodity} (fit)",
                line=dict(color=color_dict[commodity]), #dash="dash", 
                row=i + 1,
                col=j + 1
            )

            #white background with light grey dividing lines
            fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, showgrid=True, gridcolor = "lightgrey", row=i+1, col=j+1)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, showgrid=True, gridcolor = "lightgrey", row=i+1, col=j+1)
            #no background color
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

        counter = counter + 1

#update and unify legend
fig.for_each_trace(lambda trace: trace.update(showlegend=False))

#set unified legend
for com in results_df["Commodity"].unique():
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(size=20), showlegend=True, name=com, line=dict(color=color_dict[com])), row=1, col=1)
   
# Set y-axes titles and make y-axis logarithmic
fig.update_yaxes(title_text="Electricity Production (TWh)", secondary_y=False, title_font=dict(size=font_size*0.7)) #type="log", 
fig.update_xaxes(title_text="Year", title_font=dict(size=font_size))

fig.update_layout(height=he_wi*0.4, width=he_wi, font=dict(size=font_size, family="Times New Roman", color="black"), title="Electricity Production in countries with high growth between " + str(start_year) + " - " + str(last_year))
#export to png
fig.write_image(os.path.join(case_study_path, "results", "hist_el_mix_dev.png"), scale=4)
#fig.show()

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')  