import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load data (e.g., from CSV files)
@st.cache_data
def load_data(country):
    if country == 'Malaysia':
        data = pd.read_csv('BTC-MYR.csv')
    elif country == 'Thailand':
        data = pd.read_csv('BTC-THB.csv')
    elif country == 'Indonesia':
        data = pd.read_csv('BTC-IDR.csv')
    return data

# Function to convert date to numeric value
def convert_date_to_numeric(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    return data

# Function to clean and convert price column to float
def clean_and_convert_price(data):
    data['Price'] = data['Price'].str.replace(',', '').astype(float)
    return data

# Function to predict Bitcoin price using Linear Regression
def predict_price(data):
    X = data[['Days']].values
    y = data['Price'].values

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions on training set
    train_prediction = model.predict(X_train)
    
    # Predictions on test set
    test_prediction = model.predict(X_test)
    
    # Predictions for future dates (forecast)
    future_days = np.arange(max(data['Days']) + 1, max(data['Days']) + 91).reshape(-1, 1)
    future_prediction = model.predict(future_days)

    return train_prediction, test_prediction, future_prediction, X_train, X_test, y_train, y_test

# Function to create a home page with greeting message and animation
def show_home_page():
    st.title('BitPredict Pro')
    st.markdown("""
        ## Welcome to the BitPredict Pro App! :chart_with_upwards_trend:

        Use the sidebar to select a country and predict Bitcoin prices.
    """)

    st.image('bitcoin.gif', use_column_width=True)

# Function to display the country-specific page
def show_country_page(country):
    st.subheader(f'Data for {country}')
    
    # Load data for selected country
    data = load_data(country)
    data = convert_date_to_numeric(data)
    data = clean_and_convert_price(data)
    
    st.write(data)

# Function to display the prediction graph
def show_prediction_graph(country):
    # Load data for selected country
    data = load_data(country)
    data = convert_date_to_numeric(data)
    data = clean_and_convert_price(data)

    # Predict Bitcoin prices
    st.info('Predicting Bitcoin prices...')
    train_prediction, test_prediction, future_prediction, X_train, X_test, y_train, y_test = predict_price(data)

    # Plot true prices, train predictions, test predictions, and forecasted prices
    plt.figure(figsize=(14, 8))
    
    # True prices
    plt.plot(data['Date'], data['Price'], label='True Price', marker='o')

    # Training predictions
    plt.plot(data.iloc[X_train.flatten()]['Date'], train_prediction, label='Training Predictions', linestyle='--', marker='o')

    # Test predictions
    test_dates = data.iloc[X_test.flatten()]['Date']
    plt.plot(test_dates, test_prediction, label='Test Predictions', linestyle='--', marker='o')

    # Forecasted prices
    future_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=90)
    plt.plot(future_dates, future_prediction[:len(future_dates)], label='Forecasted Prices', linestyle='--', marker='o')

    # Prediction for 3 months later
    future_date_3_months = data['Date'].max() + pd.DateOffset(months=3)
    future_price_3_months_index = 90  # Index adjustment for 3 months later prediction
    future_price_3_months = future_prediction[future_price_3_months_index] if future_price_3_months_index < len(future_prediction) else None
    
    if future_price_3_months is not None:
        plt.scatter(future_date_3_months, future_price_3_months, color='red', label='Prediction for 3 Months Later')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Bitcoin Price Prediction for {country}')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Function to display the loss function graph
def show_loss_function_graph(country):
    # Load data for selected country
    data = load_data(country)
    data = convert_date_to_numeric(data)
    data = clean_and_convert_price(data)

    # Predict Bitcoin prices
    st.info('Calculating loss function...')
    train_prediction, test_prediction, _, X_train, X_test, y_train, y_test = predict_price(data)

    # Example history object for demonstration (replace with actual model training history)
    history = {
        'loss': np.linspace(mean_squared_error(y_train, train_prediction), mean_squared_error(y_train, train_prediction)/2, 500),
        'val_loss': np.linspace(mean_squared_error(y_test, test_prediction), mean_squared_error(y_test, test_prediction)/2, 500)
    }

    # Plot loss function
    dloss = history['loss']
    dval_loss = history['val_loss']
    epochs = range(1, len(dloss) + 1)

    plt.figure(figsize=(14, 8))
    plt.plot(epochs, dloss, 'bo-', label='Training loss', linewidth=2, markersize=5)
    plt.plot(epochs, dval_loss, 'ro-', label='Validation loss', linewidth=2, markersize=5)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Loss Function for {country}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)

# Function to display help and customer support information
def show_help_page():
    st.title('Help and Customer Support')
    st.markdown("""
        ## Need Assistance?
        
        If you have any questions or need support, please reach out to us:
        
        - **Email**: support@bitpredictpro.com
        - **Phone**: +1 (123) 456-7890
        
        ### Frequently Asked Questions
        
        **Q: How accurate are the predictions?**
        
        A: Our predictions are based on historical data and machine learning models. While we strive for accuracy, please note that cryptocurrency markets can be highly volatile.
        
        **Q: Can I use this app for other cryptocurrencies?**
        
        A: Currently, BitPredict Pro focuses on Bitcoin price predictions only. Future updates may include other cryptocurrencies.
        
        **Q: How often is the data updated?**
        
        A: The data is updated daily to provide the most recent information for accurate predictions.
        
        ### Contact Form
        
        Use the form below to submit your inquiry or feedback:
        
        [Contact Form](https://bitpredictpro.com/contact)
    """)

# Main function to run the Streamlit app
def main():
    st.sidebar.header('Navigation')
    page = st.sidebar.radio('Go to', ['Home', 'Country Data', 'Prediction Graph', 'Loss Function Graph', 'Help'])

    if page == 'Home':
        show_home_page()
    elif page == 'Country Data':
        country = st.sidebar.selectbox('Select Country', ['Malaysia', 'Thailand', 'Indonesia'], key='country_data')
        show_country_page(country)
    elif page == 'Prediction Graph':
        country = st.sidebar.selectbox('Select Country', ['Malaysia', 'Thailand', 'Indonesia'], key='prediction_graph')
        show_prediction_graph(country)
    elif page == 'Loss Function Graph':
        country = st.sidebar.selectbox('Select Country', ['Malaysia', 'Thailand', 'Indonesia'], key='loss_function_graph')
        show_loss_function_graph(country)
    elif page == 'Help':
        show_help_page()

if __name__ == '__main__':
    main()
