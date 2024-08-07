from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.io as pio


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Function to fetch data from Alpha Vantage API
def fetch_data(symbol, data_type):
    api_key = 'https://fred.stlouisfed.org/' 
    
from flask import Flask, request, jsonify, render_template, session
import requests
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

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

# from flask import Flask, request, jsonify, render_template, session
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import plotly.graph_objs as go
# import plotly.io as pio

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/get_data', methods=['POST'])
# def get_data():
#     # Simulating data retrieval
#     dates = pd.date_range(start='1/1/2020', periods=100)
#     closes = np.random.randint(100, 200, size=(100))
#     data = pd.DataFrame({'Date': dates, 'Close': closes})
#     session['data'] = data.to_dict()
#     return jsonify(data.to_dict())

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     data = pd.DataFrame(session.get('data'))
#     analysis = {
#         'mean': data['Close'].mean(),
#         'median': data['Close'].median(),
#         'std': data['Close'].std(),
#         'min': data['Close'].min(),
#         'max': data['Close'].max()
#     }
#     return jsonify(analysis)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = pd.DataFrame(session.get('data'))
#     X = data.index.values.reshape(-1, 1)
#     y = data['Close'].values

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=30)
#     future_X = np.arange(len(X), len(X) + 30).reshape(-1, 1)
#     future_pred = model.predict(future_X)

#     prediction_data = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_pred})
#     return jsonify(prediction_data.to_dict())

# @app.route('/interpret', methods=['POST'])
# def interpret():
#     data = pd.DataFrame(session.get('data'))
    
#     # Simple interpretation
#     latest_close = data['Close'].iloc[-1]
#     avg_close = data['Close'].mean()
    
#     if latest_close > avg_close:
#         trend = "upward"
#     else:
#         trend = "downward"
    
#     interpretation = f"The latest closing price ({latest_close:.2f}) shows a {trend} trend compared to the average ({avg_close:.2f})."
#     return jsonify({'interpretation': interpretation})

# @app.route('/visualize', methods=['POST'])
# def visualize():
#     data = pd.DataFrame(session.get('data'))
    
#     trace = go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price')
#     layout = go.Layout(title='Stock Close Price Over Time', xaxis=dict(title='Date'), yaxis=dict(title='Close Price'))
#     fig = go.Figure(data=[trace], layout=layout)
    
#     plot_json = pio.to_json(fig)
#     return jsonify({'plot': plot_json})

# if __name__ == '__main__':
#     app.run(debug=True)