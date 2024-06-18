import random
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.dates as mdates

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
data = pd.read_csv("final_data_prediction.csv")
month_map_reverse = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'Novermber': 11, 'December': 12}
data['Month'] = data['Month'].map(month_map_reverse)
data['Year'] = 2023
data['Date'] = pd.to_datetime(data['Month'].astype(str) + ' ' + data['Year'].astype(str), format='%m %Y')
data.set_index('Date', inplace=True)
data.drop(columns=['Unnamed: 0'], inplace=True)

final_data = data[["Inventory", "Value", "code_p"]]

# Normalize data
scaler_value = MinMaxScaler()
scaler_inventory = MinMaxScaler()


final_data[['Value']] = scaler_value.fit_transform(final_data[['Value']])
final_data[['Inventory']] = scaler_inventory.fit_transform(final_data[['Inventory']])

joblib.dump(scaler_value, './scaler_value.pkl')
joblib.dump(scaler_inventory, './scaler_inventory.pkl')

# Split data into training and testing sets
train_size = int(len(final_data) * 0.8)
train_data = final_data[:train_size]
test_data = final_data[train_size:]

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

sequence_length = 6

X_train, y_train, X_test, y_test = [], [], [], []

# Split data and create sequences for each product
for code_p, group in final_data.groupby('code_p'):
    group_values = group['Value'].values
    
    # Determine split index
    train_size = int(len(group_values) * 0.8)
    
    # Training data
    group_train = group_values[:train_size]
    X_tr, y_tr = create_sequences(group_train, sequence_length)
    X_train.append(X_tr)
    y_train.append(y_tr)
    
    # Testing data
    group_test = group_values[train_size - sequence_length:]  
    X_te, y_te = create_sequences(group_test, sequence_length)
    X_test.append(X_te)
    y_test.append(y_te)

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

# Convert to tensors
X_train = torch.FloatTensor(X_train).unsqueeze(2) 
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(2)  
y_test = torch.FloatTensor(y_test)

def create_lstm_model(input_size, hidden_size, num_layers, dropout):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.linear = nn.Linear(hidden_size, 1)
            self.softplus = nn.Softplus()

        def forward(self, x):
            hidden = torch.zeros(num_layers, x.size(0), hidden_size)
            cell = torch.zeros(num_layers, x.size(0), hidden_size)
            out, _ = self.lstm(x, (hidden, cell))
            out = self.linear(out[:, -1, :])
            out = self.softplus(out)
            return out

    return LSTMModel(input_size, hidden_size, num_layers,dropout)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 1
num_layers = 3
hidden_size = 80
dropout=0.2
model = create_lstm_model(input_size, hidden_size, num_layers, dropout).to(device)
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 12

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train_lstm_model(model, train_loader, test_loader, num_epochs, optimizer, loss_fn, device):
    train_hist = []
    test_hist = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average training loss
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)
        
        # Validation on test data
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0
            
            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                predictions_test = model(batch_X_test)
                test_loss = loss_fn(predictions_test, batch_y_test)
                
                total_test_loss += test_loss.item()
            
            # Calculate average test loss
            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')
        
        torch.save(model.state_dict(), 'lstm_model.pth')
    return train_hist, test_hist

def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

num_epochs = 50
train_hist, test_hist = train_lstm_model(model, train_loader, test_loader, num_epochs, optimizer, loss_fn, device)

train_loss = evaluate_model(model, train_loader, loss_fn, device)
test_loss = evaluate_model(model, test_loader, loss_fn, device)
print(f"Train Loss: {train_loss}, Test Loss: {test_loss}")


x = np.linspace(1, num_epochs, num_epochs)
plt.plot(x, train_hist, label="Training loss")
plt.plot(x, test_hist, label="Test loss")
plt.legend()
plt.show()

def prepare_sequences(data, sequence_length):
    sequences = []
    code_p_values = []
    
    for code_p, group in data.groupby('code_p'):
        values = group['Value'].values
        for i in range(len(values) - sequence_length + 1):
            sequences.append(values[i:i + sequence_length])
            code_p_values.append(code_p)
    
    return np.array(sequences), code_p_values

def forecast_future_values(model, X_test, test_data, device, num_forecast_steps):
    all_forecasted_values = []
    code_p_values = []

    model.eval()
    with torch.no_grad():
        # Use the last data points as the starting point for each product
        for code_p, group in final_data.groupby('code_p'):
            code_p_values.append(code_p)  # Store code_p for labeling
            group_values = group['Value'].values
            historical_data = torch.FloatTensor(group_values[-sequence_length:]).view(1, -1, 1).to(device)

            forecasted_values = []
            for _ in range(num_forecast_steps):
                predicted_value = model(historical_data).cpu().numpy()[0, 0]
                forecasted_values.append(predicted_value)
                
                # Update the historical_data sequence by removing the oldest value and adding the predicted value
                historical_data = torch.cat((historical_data[:, 1:, :], torch.tensor([[[predicted_value]]]).to(device)), dim=1)
            
            all_forecasted_values.append(forecasted_values)

    return np.array(all_forecasted_values), code_p_values


def predict(df, model, scaler_value, scaler_inventory, sequence_length, num_forecast_steps, device):
    # Normalize the input dataframea
    df[['Value']] = scaler_value.transform(df[['Value']])
    df[['Inventory']] = scaler_inventory.transform(df[['Inventory']])

    X_test, code_p_values = prepare_sequences(df, sequence_length)
    X_test = torch.FloatTensor(X_test).unsqueeze(2).to(device)

    forecasted_values, code_p_values = forecast_future_values(model, X_test, df, device, num_forecast_steps)

    # Generate future dates
    last_date = df.index[-1]
    future_dates = []
    for _ in range(num_forecast_steps):
        last_date += pd.DateOffset(months=1)
        future_dates.append(last_date)
    
    # Create DataFrame for predictions
    predictions_df = pd.DataFrame(forecasted_values, columns=future_dates)
    predictions_df.index = code_p_values

    # Inverse transform the predictions to actual values
    for column in predictions_df.columns:
        predictions_df[column] = scaler_value.inverse_transform(predictions_df[[column]])

    return predictions_df

new_df = final_data.copy()  # Replace with your new DataFrame
predictions_df = predict(new_df, model, scaler_value, scaler_inventory, sequence_length, 12, device)

predictions_df = predictions_df.apply(lambda x: x.astype(int))

print(predictions_df)

# Output the predictions as CSV
#predictions_df.to_csv('predictions.csv', index_label='code_p')

# Visualize the predictions
def visualize_predictions(predictions_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    for code_p in predictions_df.index:
        ax.plot(predictions_df.columns, predictions_df.loc[code_p], marker='o', label=f'code_p {code_p}')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.title('Future Predictions')
    plt.show()

visualize_predictions(predictions_df)

subset_df = predictions_df.head(3)

# Call the visualization function with this subset
visualize_predictions(subset_df)
