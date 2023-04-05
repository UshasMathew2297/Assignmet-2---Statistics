"""
@author: Ushas Mathew

This code plots 3 different visualisations that is bar plot plot, line plot 
and heat map using single dataset based on multiple index.
"""
# importing requiredpackages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis


def datafile(file):
    """
    This 'datafile' function takes name of the file and reads it from local 
    directory and loads this file into a dataframe. After that transposes the 
    dataframe and returns the original and transposed dataframes. 

    Parameters
    ----------
    file : String
        a string containing the name of the excel file to be read

    Returns
    -------
    climate : the pandas DataFrame that contains the data from the excel file
    climate_transpose : the transposed pandas DataFrame

    """

    # using the input string creating the file path assighing into file_address
    file_address = r'C:\Users\User\Documents\Assignment2' + file
    # readng the csv file into a pandas DataFrame using the file path
    climate = pd.read_excel(file_address)
    # transposing the DataFrame
    climate_transpose = climate.transpose()

    return(climate, climate_transpose)

#Calling function to read datafile
datafile('\climate_change.xlsx')
climate = pd.read_excel(
    r'C:\Users\User\Documents\Assignment2\climate_change.xlsx')


def cleandata(data):
    """
    Clean the inputed dataframe & fills any missing values in the provided 
    DataFrame with 0.

    Parameters:
    ----------
    data : pandas DataFrame 
        input DataFrame to be cleaned

    Returns:
    ----------
    Cleaned data without any NaN values

    """

    # count the number of missing values in each column of the DataFrame
    data.isnull().sum()
    # fill any missing values with 0 and update the DataFrame with new values
    data.fillna(0, inplace=True)

    return

#Calling function to clean data
cleandata(climate)


def stat_func(stat):
    """
    This function accepts a pandas DataFrame 'stat' and performs different 
    statistical analyses on the columns. It displays the summary statistics, 
    correlation matrix, skewness, and kurtosis for the specified columns.

    Parameters
    ----------
    stat : pandas DataFrame
        input the datafarame to perform different statistical functions

    Returns
    -------
    None.

    """

    # extract the columns & assign to the variable "fuction"
    function = stat.iloc[:, 4:]

    # calculate the skewness,kurtosis and Covariance
    print('Skewness :','\n',skew(function, axis=0, bias=True))
    print('Kurtosis :','\n',kurtosis(function, axis=0, bias=True))
    print('Describe :', '\n',function.describe())
    print('Covariance :','\n',function.cov())

#Calling function to calculate statistical functions
stat_func(climate)


def methane_bar(barplt):
    """
    This function takes a pandas DataFrame  containing data on worldbank climate
    change data and creates a barplot of the % change in Methane emissions from
    1990

    Parameters
    ----------
    barplt : pandas DataFrame
        passes the values of the selected countries' Methane emissions in 
        relation to the years.

    Returns
    -------
    This function plots the barplot for Methane emission

    """

    # Select rows where the "Indicator Name"  is "Methane emissions (% change from 1990)"
    methane = barplt[barplt['Indicator Name'] ==
                     'Methane emissions (% change from 1990)']

    # Choose rows where the "Country Name" column contains a country from list.
    methane_country = methane[methane['Country Name'].isin(
        ['Australia', 'Canada', 'China', 'Spain', 'India', 'New Zealand', 'Brazil'])]

    # Define the width of each bar
    bar_width = 0.1

    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(methane_country))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [x + bar_width for x in r6]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(10, 8))
    plt.bar(r1, methane_country['1990'], color='darkslategray',
            width=bar_width, edgecolor='black', label='1990')
    plt.bar(r2, methane_country['1994'], color='teal',
            width=bar_width, edgecolor='black', label='1994')
    plt.bar(r3, methane_country['1998'], color='powderblue',
            width=bar_width, edgecolor='black', label='1998')
    plt.bar(r4, methane_country['2002'], color='dodgerblue',
            width=bar_width, edgecolor='black', label='2002')
    plt.bar(r5, methane_country['2006'], color='cadetblue',
            width=bar_width, edgecolor='black', label='2006')
    plt.bar(r6, methane_country['2010'], color='cyan',
            width=bar_width, edgecolor='black', label='2010')
    plt.bar(r7, methane_country['2012'], color='b',
            width=bar_width, edgecolor='black', label='2012')

    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(methane_country))],
               methane_country['Country Name'], fontsize=15)
    plt.yticks(fontsize=15)
    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold', fontsize=15)
    plt.ylabel('Methane emission in % ', fontweight='bold', fontsize=15)
    plt.title('Methane emissions (% change from 1990)',
              fontweight='bold', fontsize=15)
    plt.savefig("bar1.png")
    plt.legend()
    plt.show()


def electricity_bar(barplt2):
    """
    creates a barplot of the % Electricity production from natural gas sources

    Parameters
    ----------
    barplt2 : pandas DataFrame
        passes the values of the selected countries' Electricity production in 
        relation to the years.

    Returns
    -------
    This function plots the barplot of Electricity production from natural gas

    """

    #Selecting the indicator"Electricity production from natural gas sources 
    electricity = barplt2[barplt2['Indicator Name'] ==
                          'Electricity production from natural gas sources (% of total)']

    # Choose rows where the "Country Name" column contains a country from list
    electricity_production = electricity[electricity['Country Name'].isin(
        ['Australia', 'Canada', 'China', 'Spain', 'India', 'New Zealand', 'Brazil'])]

    # Define the width of each bar
    bar_width = 0.1

    # Define the positions of the bars on the x-axis
    r1 = np.arange(len(electricity_production))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [x + bar_width for x in r6]

    # Create a bar plot of the selected data, with a different color for each year
    plt.subplots(figsize=(10, 8))
    plt.bar(r1, electricity_production['1990'], color='red',
            width=bar_width, edgecolor='black', label='1990')
    plt.bar(r2, electricity_production['1994'], color='indianred',
            width=bar_width, edgecolor='black', label='1994')
    plt.bar(r3, electricity_production['1998'], color='orange',
            width=bar_width, edgecolor='black', label='1998')
    plt.bar(r4, electricity_production['2002'], color='olivedrab',
            width=bar_width, edgecolor='black', label='2002')
    plt.bar(r5, electricity_production['2006'], color='sandybrown',
            width=bar_width, edgecolor='black', label='2006')
    plt.bar(r6, electricity_production['2010'], color='chartreuse',
            width=bar_width, edgecolor='black', label='2010')
    plt.bar(r7, electricity_production['2012'], color='yellow',
            width=bar_width, edgecolor='black', label='2012')

    # Set the x-tick labels to the country names
    plt.xticks([r + bar_width*2 for r in range(len(electricity_production))],
               electricity_production['Country Name'], fontsize=15)
    plt.yticks(fontsize=15)
    # Adding labels to the axis
    plt.xlabel('Countries', fontweight='bold', fontsize=15)
    plt.ylabel('% Electricity production from natural gas',
               fontweight='bold', fontsize=15)
    plt.title('Electricity production from natural gas sources (% of total)',
              fontweight='bold', fontsize=15)
    plt.savefig("bar2.png")
    plt.legend()
    plt.show()


# Calls the functions to create the barplot
methane_bar(climate)
electricity_bar(climate)


def greenhouse_lineplot(line):
    """
    Plots a line plot of Total greenhouse gas emissions (kt of CO2 equivalent)

    Parameters
    ----------
    line : pandas DataFrame
        This parameter passes the values of selected contries' Total greenhouse
        gas emissions respect to the years.

    Returns
    -------
    This function plots the lineplot for greenhouse gas emissions

    """

    # filtering out the data related to forest area for selected countries
    greenhouse_gas = line[line['Indicator Name'] ==
                          'Total greenhouse gas emissions (kt of CO2 equivalent)']

    country_name = greenhouse_gas[greenhouse_gas['Country Name'].isin(
        ['Australia', 'Canada', 'China', 'Spain', 'India', 'New Zealand', 'Brazil'])]

    # creating transpose of the filtered data
    transpose = country_name.transpose()
    transpose.rename(columns=transpose.iloc[0], inplace=True)
    greenhouse_transpose = transpose.iloc[4:]

    # Replacing the null values by zeros
    greenhouse_transpose.fillna(0, inplace=True)

    # plotting the line graph
    plt.figure(figsize=(12, 8))
    plt.plot(greenhouse_transpose.index,
             greenhouse_transpose['Australia'], linestyle='dashed', label='Australia')
    plt.plot(greenhouse_transpose.index,
             greenhouse_transpose['Canada'], linestyle='dashed', label='Canada')
    plt.plot(greenhouse_transpose.index,
             greenhouse_transpose['China'], linestyle='dashed', label='China')
    plt.plot(greenhouse_transpose.index,
             greenhouse_transpose['Spain'], linestyle='dashed', label='Spain')
    plt.plot(greenhouse_transpose.index,
             greenhouse_transpose['India'], linestyle='dashed', label='India')
    plt.plot(greenhouse_transpose.index,
             greenhouse_transpose['New Zealand'], linestyle='dashed', label='New Zealand')
    plt.plot(greenhouse_transpose.index,
             greenhouse_transpose['Brazil'], linestyle='dashed', label='Brazil')
    # Setting x limit
    plt.xlim('2000', '2012')
    plt.xticks(fontsize=20, rotation ='vertical')
    plt.yticks(fontsize=20)
    # Adding labels to the axis
    plt.xlabel('Year', fontsize=20, fontweight='bold')
    plt.ylabel('Total greenhouse gas emissions',
               fontsize=20, fontweight='bold')
    plt.title('Total greenhouse gas emissions (kt of CO2 equivalent)',
              fontsize=15, fontweight='bold')
    plt.savefig("line1.png")
    plt.legend()
    plt.show()


def population_lineplot(line2):
    """
    Plots a line plot of Population growth (annual %)

    Parameters
    ----------
    line2 : pandas DataFrame
        This parameter passes the values of selected contries' Population 
        growth anual respect to the years.

    Returns
    -------
    This function plots the lineplot for Population growth

    """

    # filtering out the data related to forest area for selected countries
    energy = line2[line2['Indicator Name'] ==
                   'Population growth (annual %)']

    energy_consumption = energy[energy['Country Name'].isin(
        ['Australia', 'Canada', 'China', 'Spain', 'India', 'New Zealand', 'Brazil'])]

    # creating transpose of the filtered data
    transpose2 = energy_consumption.transpose()
    transpose2.rename(columns=transpose2.iloc[0], inplace=True)
    energy_transpose = transpose2.iloc[4:]

    # Replacing the null values by zeros
    energy_consumption.fillna(0, inplace=True)

    # plotting the line graph
    plt.figure(figsize=(12, 8))
    plt.plot(energy_transpose.index,
             energy_transpose['Australia'], linestyle='dashed', label='Australia')
    plt.plot(energy_transpose.index,
             energy_transpose['Canada'], linestyle='dashed', label='Canada')
    plt.plot(energy_transpose.index,
             energy_transpose['China'], linestyle='dashed', label='China')
    plt.plot(energy_transpose.index,
             energy_transpose['Spain'], linestyle='dashed', label='Spain')
    plt.plot(energy_transpose.index,
             energy_transpose['India'], linestyle='dashed', label='India')
    plt.plot(energy_transpose.index,
             energy_transpose['New Zealand'], linestyle='dashed', label='New Zealand')
    plt.plot(energy_transpose.index,
             energy_transpose['Brazil'], linestyle='dashed', label='Brazil')
    # Setting x limit
    plt.xlim('2000', '2012')
    plt.xticks(fontsize=20, rotation ='vertical')
    plt.yticks(fontsize=20)
    # Adding labels to the axis
    plt.xlabel('Year', fontsize=20, fontweight='bold')
    plt.ylabel('% Anual population growth',
               fontsize=20, fontweight='bold')
    plt.title('Population growth (annual %)',
              fontsize=15, fontweight='bold')
    plt.savefig("line2.png")
    plt.legend()
    plt.show()


# Calls the functions to create the lineplot
greenhouse_lineplot(climate)
population_lineplot(climate)


def heatmap_brazil(heat):
    """
    A function that creates a heatmap of the correlation matrix between 
    different indicators for China.

    Parameters
    ----------
    heat : pandas DataFrame
        A DataFrame containing data on different indicators for various countries.

    Returns
    -------
    This function plots the heatmap for China

    """

    # Specify the indicators to be used in the heatmap
    indicator = ['Methane emissions (% change from 1990)',
                 'Electricity production from natural gas sources (% of total)',
                 'Population growth (annual %)',
                 'Total greenhouse gas emissions (kt of CO2 equivalent)',
                 'Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent)']

    # Filter the data to keep only Brazil's data and the specified indicators
    country = heat.loc[heat['Country Name'] == 'China']
    brazil = country[country['Indicator Name'].isin(indicator)]
    # Pivot the data to create a DataFrame with each indicator as a column
    brazil_df = brazil.pivot_table(brazil, columns=heat['Indicator Name'])
    # Compute the correlation matrix for the DataFrame
    brazil_df.corr()
    # Plot the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(brazil_df.corr(), fmt='.2g', annot=True,
                cmap='rocket', linecolor='black')
    plt.title('China', fontsize=35, fontweight='bold')
    plt.savefig("heat1.png")
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


def heatmap_India(heat2):
    """
    A function that creates a heatmap of the correlation matrix between 
    different indicators for India.

    Parameters
    ----------
    heat2 : pandas DataFrame
        A DataFrame containing data on different indicators for various countries.

    Returns
    -------
    This function plots the heatmap for India

    """

    # Specify the indicators to be used in the heatmap
    indicator = ['Methane emissions (% change from 1990)',
                 'Electricity production from natural gas sources (% of total)',
                 'Population growth (annual %)',
                 'Total greenhouse gas emissions (kt of CO2 equivalent)',
                 'Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent)']

    # Filter the data to keep only India's data and the specified indicators
    country = heat2.loc[heat2['Country Name'] == 'India']
    india = country[country['Indicator Name'].isin(indicator)]
    # Pivot the data to create a DataFrame with each indicator as a column
    india_df = india.pivot_table(india, columns=heat2['Indicator Name'])
    # Compute the correlation matrix for the DataFrame
    india_df.corr()
    # Plot the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(india_df.corr(), fmt='.2g', annot=True,
                cmap='BuPu', linecolor='black')
    plt.title('India', fontsize=35, fontweight='bold')
    plt.savefig("heat2.png")
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


# Calls the functions to create the heatmaps
heatmap_brazil(climate)
heatmap_India(climate)
