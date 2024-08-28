import seaborn as sns
import math
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.dates as mdates

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data = pd.read_csv("final_data_prediction.csv")
month_map_reverse = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'Novermber': 11, 'December': 12}
data['Month'] = data['Month'].map(month_map_reverse)
data['Year'] = 2023
data['Date'] = pd.to_datetime(data['Month'].astype(str) + ' ' + data['Year'].astype(str), format='%m %Y')
data.set_index('Date', inplace=True)
data.drop(columns=['Unnamed: 0'], inplace=True)

data_2022 = pd.read_csv("data_for_pred2022.csv")
data_2022.set_index('Date', inplace=True)
data_2022 = data_2022[["Minimum stock", "Value", "code", "purchase_r"]]
data_2023 = data[["Minimum stock", "Value", "code", "purchase_r"]]

final_data = pd.concat([data_2022, data_2023])
final_data['Value'] = final_data['Value'].replace(0, np.nan).rolling(window=3).mean().fillna(final_data['Value'])
final_data['purchase_r'] = final_data['purchase_r'].replace(0, np.nan).rolling(window=3).mean().fillna(final_data['purchase_r'])
final_data['Minimum stock'] = final_data['Minimum stock'].replace(0, np.nan).rolling(window=3).mean().fillna(final_data['Minimum stock'])

scaler_value = MinMaxScaler()
scaler_Minimum_stock = MinMaxScaler()
scaler_purchase_r = MinMaxScaler()

final_data[['Value']] = scaler_value.fit_transform(final_data[['Value']])
final_data[['Minimum stock']] = scaler_Minimum_stock.fit_transform(final_data[['Minimum stock']])
final_data[['purchase_r']] = scaler_purchase_r.fit_transform(final_data[['purchase_r']])

train_len = math.ceil(len(final_data) * 0.8)
train_data = final_data[:train_len]
test_data = final_data[train_len:]
#print(f"Test data shape: {test_data.shape}")

seq_length = 4 #3
x_train, y_train = [], []
grouped_by = train_data.groupby('code')

for code, group in grouped_by:
    scaled_train = group[['Value', 'Minimum stock', 'purchase_r']].values
    num_sequences = len(scaled_train) - seq_length
    if num_sequences > 0:
        for i in range(num_sequences):
            x_train.append(scaled_train[i:i + seq_length])
            y_train.append(scaled_train[i + 1:i + seq_length + 1])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

x_test, y_test = [], []
grouped_by_test = test_data.groupby('code')

for code, group in grouped_by_test:
    scaled_test = group[['Value', 'Minimum stock', 'purchase_r']].values
    num_sequences = len(scaled_test) - seq_length
    if num_sequences > 0:
        for i in range(num_sequences):
            x_test.append(scaled_test[i:i + seq_length])
            y_test.append(scaled_test[i + 1:i + seq_length + 1])

x_test, y_test = np.array(x_test), np.array(y_test)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

def makeModel(input_size, hidden_size, num_layers):
    class BiLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers): 
            super(BiLSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(hidden_size * 2, input_size)
            self.softplus = nn.Softplus()
 
        def forward(self, x): 
            out, _ = self.lstm(x)
            out = self.linear(out)
            out = self.softplus(out)
            return out
    return  BiLSTMModel(input_size, hidden_size, num_layers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training(model, train_loader, test_loader, optimizer, loss_fn, num_epochs, device=device, patience=20):
    train_hist = []
    test_hist = []

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_loss = 0.0

        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                predictions_test = model(batch_X_test)
                test_loss = loss_fn(predictions_test, batch_y_test)

                total_test_loss += test_loss.item()

            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)

        if average_test_loss < best_loss:
            best_loss = average_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping at epoch {epoch+1} because there is no improvement in validation loss.')
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')

    return train_hist, test_hist

input_size = 3 
num_layers = 2 #3
hidden_size = 64

model = makeModel(input_size, hidden_size, num_layers).to(device)
 
loss_fn  = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)

batch_size = 64 #32 16

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_hist, test_hist = training(model, train_loader, test_loader, optimizer, loss_fn, num_epochs=70)

"""
plt.figure(figsize=(14, 7))
plt.plot(train_hist, label='Train Loss')
plt.plot(test_hist, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss for Nadam Optimizer')
plt.legend()
plt.show()
"""

def predict(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_seq = input_seq.unsqueeze(0).to(device)
        output = model(input_seq)
    return output.squeeze().cpu().numpy()

def predict_by_product(model, test_data, seq_length, num_predictions=12):
    predictions = {}
    grouped = test_data.groupby('code')
    
    for code, group in grouped:
        scaled_data = group[['Value', 'Minimum stock', 'purchase_r']].values
        product_predictions = []
        seq = torch.FloatTensor(scaled_data[-seq_length:]).to(device)
        
        for _ in range(num_predictions):
            pred = predict(model, seq)
            product_predictions.append(pred[-1])
            new_seq = pred[-1].reshape(1, -1)
            seq = torch.cat((seq[1:], torch.FloatTensor(new_seq).to(device)), dim=0)
        
        predictions[code] = product_predictions
    
    return predictions

def generate_future_dates(start_date, periods):
    dates = pd.date_range(start=start_date, periods=periods, freq='M')
    return dates

last_date = data.index[-1]
future_dates = generate_future_dates(last_date, 12)

codes_2022 = set(data_2022['code'].unique())
codes_2023 = set(data_2023['code'].unique())
common_codes = codes_2022.intersection(codes_2023)

test_data_common = test_data[test_data['code'].isin(common_codes)]

predictions = predict_by_product(model, test_data_common, seq_length)

for code in predictions:
    predictions[code] = scaler_value.inverse_transform(np.array(predictions[code]).reshape(-1, 1)).flatten()

df = pd.DataFrame(dict([(k, pd.Series(v[:12])) for k, v in predictions.items()]))
df = df.round().astype(int)

date_range = pd.date_range(start='2024-01-01', periods=12, freq='MS')
formatted_dates = date_range.strftime('%d-%m-%Y')

df.index = formatted_dates
print(df)

model_save_path = "bilstm_model.pth"
torch.save(model.state_dict(), model_save_path)

