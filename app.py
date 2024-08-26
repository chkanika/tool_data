from flask import Flask, request, jsonify, render_template, session
# from flask_socketio import SocketIO, emit
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key
# socketio = SocketIO(app)

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
    
    if data_type == 'TIME_SERIES_DAILY':
        dates = list(data['Time Series (Daily)'].keys())
        closes = [float(data['Time Series (Daily)'][date]['4. close']) for date in dates]
        data_for_display = {'Date': dates, 'Close Price': closes}
    else:
        data_for_display = {}  # Handle other data types as needed
    
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
    data = request.json
    print("Received data for prediction:", data)  # Debugging line
    
    df = pd.DataFrame(data)
    X = df[['x']].values
    y = df['y'].values
    model = LinearRegression().fit(X, y)
    prediction = model.predict(X)
    df['prediction'] = prediction.tolist()
    
    return df.to_json(orient='records')

# Route to visualize data with Plotly
@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    print("Received data for visualization:", data)  # Debugging line

    x_values = [item['x'] for item in data]
    y_values = [item['y'] for item in data]
    prediction_values = [item.get('prediction', None) for item in data]

    trace1 = go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        name='Actual Data'
    )
    
    trace2 = go.Scatter(
        x=x_values,
        y=prediction_values,
        mode='lines',
        name='Prediction'
    )
    
    layout = go.Layout(
        title='Data Visualization with Prediction',
        xaxis={'title': 'X Axis'},
        yaxis={'title': 'Y Axis'}
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    graph_json = fig.to_json()
    
    return jsonify({'graph_json': graph_json})

# Function to interpret data
def interpret_data(input_text):
    return f"Interpretation of '{input_text}': This is a sample interpretation."

# Route to interpret user input
@app.route('/interpret', methods=['POST'])
def interpret():
    data = request.json
    input_text = data.get('input', '')
    interpretation = interpret_data(input_text)
    return jsonify({'interpretation': interpretation})

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Socket.IO event handlers for messaging and whiteboard

# @socketio.on('send_message')
# def handle_message(message):
#     emit('new_message', message, broadcast=True)

# @socketio.on('update_whiteboard')
# def handle_whiteboard_update(data):
#     emit('whiteboard_update', data, broadcast=True)

if __name__ == '__main__':
    