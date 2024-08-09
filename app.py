from flask import Flask, request, jsonify, render_template, session
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.io as pio


app = Flask(__name__)
app.secret_key = 'your_secret_key' 

# Function to fetch data from Alpha Vantage API
def fetch_data(symbol, data_type):
    api_key = 'your_alpha_vantage_api_key'  # Replace with your Alpha Vantage API key
    url = f'https://www.alphavantage.co/query?function={data_type}&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Route to get data
@app.route('/get_data', methods=['POST'])
def get_data():
    symbol = request.form.get('symbol')
    data_type = request.form.get('data_type')
    data = fetch_data(symbol, data_type)
    
    if 'Error Message' in data:
        return jsonify({'error': data['Error Message']})
    
    # Extract relevant data for display
    if data_type == 'TIME_SERIES_DAILY':
        dates = list(data['Time Series (Daily)'].keys())
        closes = [float(data['Time Series (Daily)'][date]['4. close']) for date in dates]
        data_for_display = {'Date': dates, 'Close Price': closes}
    else:
        data_for_display = {}  # Handle other data types as needed
    
    # Store data in session
    session['data'] = data_for_display
    
    return jsonify(data_for_display)

# Route to store user input data
@app.route('/store_data', methods=['POST'])
def store_data():
    data = request.json
    session['user_data'] = data
    return jsonify({'message': 'Data stored successfully!'})

# Route to predict based on JSON data
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming JSON data is sent from frontend
    df = pd.DataFrame(data)
    X = df[['x']].values
    y = df['y'].values
    model = LinearRegression().fit(X, y)
    prediction = model.predict(X)
    df['prediction'] = prediction.tolist()  # Convert numpy array to list
    return df.to_json(orient='records')

# Route to visualize based on JSON data
@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    
    # Example visualization logic (using Plotly)
    x_values = [item['x'] for item in data]
    y_values = [item['y'] for item in data]
    prediction_values = [item['prediction'] for item in data]
    
    trace1 = {
        'x': x_values,
        'y': y_values,
        'mode': 'markers',
        'type': 'scatter',
        'name': 'Actual Data'
    }
    
    trace2 = {
        'x': x_values,
        'y': prediction_values,
        'mode': 'lines',
        'type': 'scatter',
        'name': 'Prediction'
    }
    
    layout = {
        'title': 'Data Visualization with Prediction',
        'xaxis': {
            'title': 'X Axis'
        },
        'yaxis': {
            'title': 'Y Axis'
        }
    }
    
    return jsonify({'plot_data': [trace1, trace2], 'layout': layout})

def interpret_data(input_text):
    # This is a placeholder function. In a real application, you'd use 
    # more sophisticated natural language processing or AI techniques.
    return f"Interpretation of '{input_text}': This is a sample interpretation."

@app.route('/interpret', methods=['POST'])
def interpret():
    data = request.json
    input_text = data.get('input', '')
    interpretation = interpret_data(input_text)
    return jsonify({'interpretation': interpretation})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
