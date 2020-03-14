import pandas as pd
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import datetime

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


link = 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-{}-{}-{}.xls'

date = [str(x).zfill(2) for x in [datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day]]

try:
    df = pd.read_excel(link.format(date[0], date[1], date[2]))
except:
    df = pd.read_excel(link.format(date[0], date[1], int(date[2])-1))

    

df['Total infected'] =  df[::-1].groupby('CountryExp')['NewConfCases'].transform(pd.Series.cumsum)[::-1]
df['Total deceased'] = df[::-1].groupby('CountryExp')['NewDeaths'].transform(pd.Series.cumsum)[::-1]
df['Death rate in %'] = (df['Total deceased']/df['Total infected'])*100
df.rename(columns = {'CountryExp':'Country', 'DateRep':'Date', 'NewConfCases':'New Confirmed Cases', 'NewDeaths':'New Deaths'}, inplace=True)


country_codes = pd.read_csv('data/iso_codes.csv')
df['iso_3'] = df.GeoId.apply(lambda x: x.replace(x, country_codes.loc[country_codes['2code'] == x]['3code'].values[0]) if len(country_codes.loc[country_codes['2code'] == x]['3code'])>0 else np.nan)

all_countries = ['Norway','Sweden','Denmark']
land = list(df.Country.unique())
all_countries.extend(land)


columns=['Country', 'First recorded case', 'First recorded death', 'Day count', 'EU']
first_indicent_df = pd.DataFrame(columns=columns)

for country in df.Country.unique():
    
    first_case = df.loc[(df.Country == country) & (df['New Confirmed Cases'] > 0)].Date.min()
    first_death = df.loc[(df.Country == country) & (df['New Deaths'] > 0)].Date.min() 
    
    period = first_death - first_case
    
    if df.loc[(df.Country == country)].EU.unique() in (['EU','EEA']):
        eu = True
    else:
        eu = False
        
        
    first_indicent_df = first_indicent_df.append( pd.DataFrame(data=[[country, first_case, first_death, period.days, eu]], columns=columns))
        
populations = pd.read_csv('data/world_population.csv')
populations.replace('United States','United States of America', inplace=True)

df = df.merge(populations[['Country (or dependency)','Population\n(2020)']].drop_duplicates().rename(columns={'Country (or dependency)':'Country', 'Population\n(2020)':'Population'}), how='left', on='Country')

df['Population'] = df['Population'].apply(lambda x: int(str(x).replace(',','')) if str(x) != 'nan' else x)

df['infected in %'] = (df['Total infected'] / df['Population'] )*100

d = px.data.gapminder()
d.replace('United States','United States of America', inplace=True)


map_df = df.merge(d[['country','iso_alpha']].drop_duplicates().rename(columns={'country':'Country'}), how='left', on='Country')

map_df['Date'] = map_df['Date'].apply(lambda x: str(x.date()))

map_df['iso_alpha'] = map_df.apply(lambda row: row['GeoId'] if str(row['iso_alpha']) == 'nan' else row['iso_alpha'], axis=1)



def fit_func(df, hoverData):
    def func(x, a, b, c):
        return a * np.exp(b * x) + c
    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0)))+b
        return (y)
    
    function = func  
    def log_to_sigmoid(country):    
        test = country[country.Cases == 'modelled'].copy(deep=True)

        test['days from zero'] = test.apply(lambda row: country[country.Cases == 'modelled']['days from zero'].max() + (country[country.Cases == 'modelled']['days from zero'].max() - (row['days from zero'])), axis=1)
        test['# Cases'] = test.apply(lambda row: country[country.Cases == 'modelled']['# Cases'].max() + (country[country.Cases == 'modelled']['# Cases'].max() - (row['# Cases'])), axis=1)
        test['Date'] = test.apply(lambda row: country[country.Cases == 'modelled']['Date'].min() + datetime.timedelta(days=row['days from zero']), axis=1)
        test['Cases'] = 'sigmoid'
        
        return test
    
    data = df.loc[(df.Country == hoverData) & (df['Total infected'] > 0)].melt(id_vars='Date', value_vars=['Total infected', 'Total deceased',  'New Confirmed Cases', 'New Deaths']).rename(columns = {'value':'# Cases'})
    total_infected = data.loc[data['variable'] == 'Total infected'].copy(deep=True)
    total_infected['days from zero'] = total_infected.Date.apply(lambda x: (x - total_infected.Date.min()).days)
    total_infected['Cases'] = 'recorded'

    total_infected_copy = total_infected.copy(deep=True)
    
    xdata = total_infected['days from zero']
    ydata = total_infected['# Cases']

    
    try:
        popt, pcov = curve_fit(function, xdata, ydata)
        total_infected_copy['# Cases'] = func(xdata, *popt)
        total_infected_copy['Cases'] = 'modelled'  

        if (total_infected_copy['# Cases'].corr(total_infected['# Cases'], method='pearson', min_periods=None)) < 0.95:
            function = sigmoid
            p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess
            popt, pcov = curve_fit(function, xdata, ydata, p0, method='dogbox')
            total_infected_copy['# Cases'] = sigmoid(xdata, *popt)
            total_infected_copy['Cases'] = 'modelled'  

        else:
            sigmoid_df = log_to_sigmoid(total_infected_copy)
            total_infected_copy = total_infected_copy.append(sigmoid_df)

        total_infected = total_infected.append(total_infected_copy)

        for date in ([total_infected.loc[total_infected.Cases == 'modelled'].Date.max() + datetime.timedelta(days=(x)) for x in range(3)]):
            total_infected = total_infected.append(pd.DataFrame(data=[[date, np.nan, function((date - total_infected['Date'].min()).days, *popt), (date - total_infected['Date'].min()).days, 'forecast']], columns = total_infected.columns))
        return total_infected
        
    except:
        return total_infected
#     print(total_infected.loc[total_infected.Cases=='modelled']['# Cases'].corr(total_infected.loc[total_infected.Cases=='recorded']['# Cases'], method='pearson', min_periods=None))

def all_vars(df, country):

    data = df.loc[(df.Country == country)].melt(id_vars='Date', value_vars=['Total infected', 'Total deceased',  'New Confirmed Cases', 'New Deaths']).rename(columns = {'value':'# Cases'})
    data['days from zero'] = data.Date.apply(lambda x: (x - data.Date.min()).days)
    
    return data




def p(df, country):
    
    total_infected = fit_func(df, country)
    country_stats = all_vars(df, country)
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("# Infected after day 0 (L)","Overview of infected and detah cases (R)"))



    for case in total_infected['Cases'].unique():
        fig.add_trace(
            go.Scatter(x=total_infected.loc[total_infected.Cases == case]['days from zero'], 
                       y=total_infected.loc[total_infected.Cases == case]['# Cases'], 
                       name = (case+' L'), mode='lines', legendgroup="group1"),
                       row=1, col=1)

    for var in country_stats['variable'].unique():
        fig.add_trace(
            go.Scatter(x=country_stats.loc[country_stats.variable == var]['days from zero'], 
                       y=country_stats.loc[country_stats.variable == var]['# Cases'], 
                       name = (var+' R'), legendgroup="group2"),
                       row=1, col=2)
        
    fig.update_layout(height=600,title_text="{}".format(country), legend_orientation="h")
    fig.show()

    
def scatter_plot(variable):
    fig = px.scatter(first_indicent_df.loc[~np.isnan(first_indicent_df['First recorded death'])], x=variable, y="Day count", color="Country")
    fig.update_layout(height=600, title_text='Days from first recorded infection to <br> first recorded death by date {} '.format(variable))
    fig.show()
    
    
def plot_map(variable, map_df):
    
    color_ranges = {'infected in %':[0,0.01], 'Death rate in %':[1,7]}
    titles = {'infected in %':'Infected population in % by country', 'Death rate in %':'Death rate in % y country'}
    
    fig = px.choropleth(map_df.sort_values(by='Date', ascending=False), locations="iso_3", color=variable, hover_name="Country", animation_frame='Date', range_color=color_ranges[variable])
    fig.update_layout(height=600, title_text="{}".format(titles[variable]))
    
    fig.show()
    
def plot_death_rate_overall(df):
    data = pd.DataFrame(df.groupby('Date')['Total deceased'].sum()/df.groupby('Date')['Total infected'].sum())
    data_add = pd.DataFrame(data.rolling(7).mean(position='center'))

    for d in [data, data_add]:
        d.rename(columns = {0:'%'}, inplace=True)

    data['values'] = 'actual'
    data_add['values'] = 'smoothed (7 days)'

    data = data.append(data_add)

    data['%'] = data['%']*100

    fig = px.line(data.reset_index(), x='Date', y='%', color='values')
    fig.update_traces(textposition='top center')

    fig.update_layout(
        height=500,
        title_text='Death Ratio over time - Average Death Rate by {}: {}%'.format(df.Date.max().date(), round((df.loc[df.Date == df.Date.max()]['Total deceased'].sum() / df.loc[df.Date == df.Date.max()]['Total infected'].sum())*100, 2))
    )
    
    fig.show()
   


def death_rate_by_country(df):
    
    df.loc[df.Date == df.Date.max()]

    fig = px.bar(df.loc[(df.Date == df.Date.max()) & (df['Total deceased']) > 0], x='Country', y='Death rate in %')
    fig.update_layout(
        height=500,
        title_text='Death Ratio by Country per {}'.format(df.Date.max().date())
    )
    fig.show()