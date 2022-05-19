# IMPORT USEFUL LIBRARIES

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pymysql
import pytz
import pandas as pd
import streamlit as st

from statsmodels.tsa.arima.model import ARIMA


# CONNECT TO AWS RDS DATABASE
# Use Streamlit secrets to store information about database
hostname = st.secrets["hostname"]
dbname = st.secrets["dbname"]
uname = st.secrets["uname"]
pwd = st.secrets["pwd"]

# SET UP MYSQL CONNECTION
conn = pymysql.connect(host=hostname,
                       db=dbname,
                       port=int(3306),
                       user=uname,
                       password=pwd,
                       charset='utf8mb4'
                       )


# DEFINE FUNCTION TO SELECT SPECIFIC PARKING STRUCTURE FROM DATABASE
def getData(structure):
    df = pd.read_sql_query("""
                        SELECT CONCAT(year, '-', month, '-', day, '-', hour) AS date, freespaces AS "Free Spaces"
                        FROM status_history 
                        WHERE parkingname = '{structure}'
                        """.format(structure=structure), conn)
    # Set dataframe index as Year-Month-Day-Hour
    df.index = pd.to_datetime(df['date'], format='%Y-%m-%d-%H')
    # Delete date column
    del df['date']
    # Format columns
    df['Free Spaces'] = pd.to_numeric(df['Free Spaces'])
    return df


# VISUALIZE HISTORICAL DATA SIMPLE VERSION
def historicalDataSimple(df, structure, week):
    # Initialize number of days to display
    num_day_display = 24 * 7 * 2  # (2 weeks)

    # Determine how many days to be included in the plot from user input
    for num_day in range(week+1):
        num_day_display = num_day * 24 * 7

    # Display graph
    st.subheader('Historical Data of Number of Free Spaces at {struct}'.format(struct=structure))
    st.line_chart(df.tail(num_day_display), use_container_width=True)

    # Data source link
    st.write("Data source: [Los Angeles International Airport (LAX) - Parking Lots Current Status](https://data.lacity.org/Transportation/Los-Angeles-International-Airport-LAX-Parking-Lots/dik5-hwp6)")


# DISPLAY PREDICTION MADE BY ARIMA MODEL
def arimaPredict(df, structure):
    # Generate testing dataframe df_test
    # Dataframe contains indexes for future 3 days, 72 hours in total

    # Use PST timezone
    pst = pytz.timezone('America/Los_Angeles')

    # Dynamic datetime (original implementation)
    # dt = datetime.date(datetime.now(pst))

    # Static datetime
    dt = datetime.strptime('18/05/22', '%d/%m/%y').date()

    test_date = []

    # First fill remaining hours of today
    # Dynamic datetime (original implementation)
    # hour_now = datetime.now(pst).hour

    # Static hour
    hour_now = 17

    hour_to_fill = 24 - hour_now
    for hour in range(hour_to_fill):
        test_date.append([str(dt) + '-' + str(hour_now + hour)])

    # Then fill dates within 3 days
    for day in range(1, 3):
        for hour in range(24):
            test_date.append([str(dt + timedelta(days=day)) + '-' + str(hour)])

    # Fill last day's remaining hours
    for hour in range(hour_now):
        test_date.append([str(dt+timedelta(days=3)) + '-' + str(hour)])

    # Format testing dataframe
    df_test = pd.DataFrame(test_date, columns=['date'])
    df_test.index = pd.to_datetime(df_test['date'], format='%Y-%m-%d-%H')
    del df_test['date']
    test = df_test

    # Format training dataframe
    train = df[df.index <= pd.to_datetime(str(dt) + '-' +
                                          str(int(datetime.now(pst).hour-1)),
                                          format='%Y-%m-%d-%H')]
    y = train['Free Spaces']

    # Set up ARIMA model
    arima_model = ARIMA(y, order=(2, 1, 2))
    arima_model = arima_model.fit()

    # Train ARIMA model
    y_pred = arima_model.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = arima_model.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index

    # Select useful information from model
    y_pred_out = y_pred_df[["Predictions"]].copy()
    df_join = train.tail(24).append(y_pred_out)
    # Plot the result
    st.subheader('Free Spaces Prediction at {struct}'.format(struct=structure))
    st.line_chart(df_join, use_container_width=True)


if __name__ == '__main__':
    # MAKE TITLE
    st.header('Historical Data and Prediction')

    # NOTE FOR STOP GETTING DATA
    st.markdown("***We stop fetching data from data source on 18 May 2022 at 5 p.m. PST.***")

    # DESCRIPTION
    st.markdown("""
                On this page, you can get more detailed information about a particular parking structure.
                Feel free to select a parking structure that you want to learn more about in the selection box. 
                You can also adjust the number of weeks of historical data displayed in the graph below. 
                """)
    st.markdown("""
                You can zoom in to inspect the value of free spaces in the graph. Additionally, 
                you can place your mouse on top of the line, and detailed information will show up.
                """)

    # READ USER INPUTS
    # Get name of parking structure from drop down menu
    options = st.selectbox(
        label='Select the parking lot that you want to know in detail:',
        options=('P-1', 'P-2A', 'P-2B', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7', 'LAX Economy Parking')
    )

    # GET NUMBER OF WEEKS FOR DISPLAY IN GRAPH
    num_week_display = st.slider(
        label='Select number of weeks of historical data you want to inspect:',
        min_value=1,
        max_value=4,
        value=1
    )

    # GET DATA FROM DB
    data = getData(str(options))
    # PLOT GRAPH
    historicalDataSimple(data, str(options), int(num_week_display))

    # OBSERVATIONS
    st.subheader('Observations')
    st.markdown("""
                By plotting the data with date on the X-axis and the number of free spaces on the Y-axis,
                we can see that the space available in a particular parking structure at LAX follows some patterns.
                """)
    st.markdown("""
                On an hourly basis, we see that more spaces are available near midnight, from 12 to 3 a.m. late night. 
                In contrast, most of the spaces are occupied at near noon, from 11 a.m. to 3 p.m. everyday.
                """)
    st.markdown("""
                You can select more weeks of historical data to be displayed to see the patter on an weekly basis. 
                Generally speaking, there are more spaces available in early week. Then the number of spaces available
                decreases throughout the week. Typically, more spaces are occupied during the weekend.
                """)

    # ARIMA Model
    st.subheader('Autoregressive Integrated Moving Average Model')
    st.markdown("""
                In this project, given the identity of time series data we employed the 
                autoregressive integrated moving average (ARIMA) model to predict potential free spaces for 
                distinct parking lots at LAX. We use historical data to predict free spaces at the selected 
                parking structure in the future 3 days.
                """)

    # CALL MODEL
    arimaPredict(data, str(options))

    # Conclusion
    st.subheader('Conclusion')
    st.markdown("""
                You can use the free paces prediction for future 3 days to plan you trip to LAX ahead! Notice that this
                model is dynamic based on historical data. We believe the model will improve with more data collected.
                """)
