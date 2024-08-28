from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import pandas as pd

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
    
    return BiLSTMModel(input_size, hidden_size, num_layers)


input_size = 3  
num_layers = 2 
hidden_size = 64
model_load_path = r"C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/bilstm_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = makeModel(input_size, hidden_size, num_layers).to(device)

def prepare_data(df):
    # Initialize scalers
    scaler_value = MinMaxScaler(feature_range=(0, 1))
    scaler_min_stock = MinMaxScaler(feature_range=(0, 1))
    scaler_purchase_r = MinMaxScaler(feature_range=(0, 1))
    
    # Scale the features
    df[['Value']] = scaler_value.fit_transform(df[['Value']])
    df[['Minimum stock']] = scaler_min_stock.fit_transform(df[['Minimum stock']])
    df[['purchase_r']] = scaler_purchase_r.fit_transform(df[['purchase_r']])
    
    # Convert to tensor
    data = torch.tensor(df[['Value', 'Minimum stock', 'purchase_r']].values, dtype=torch.float32)

    
    return data, scaler_value, scaler_min_stock, scaler_purchase_r

def predict(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_seq = input_seq.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_seq)
    return output.squeeze().cpu().numpy()

def predict_by_product(model, test_data, seq_length, num_predictions=12):
    if isinstance(test_data, torch.Tensor):
        test_data = pd.DataFrame(test_data.numpy(), columns=['Value', 'Minimum stock', 'purchase_r'])
    elif not isinstance(test_data, pd.DataFrame):
        raise TypeError("test_data should be a DataFrame or Tensor.")
    
    if 'code' in test_data.columns:
        test_data.set_index('code', inplace=True)

    required_columns = ['Value', 'Minimum stock', 'purchase_r']
    missing_columns = [col for col in required_columns if col not in test_data.columns]

    for col in missing_columns:
        if col == 'purchase_r':
            if 'Value' in test_data.columns:
                test_data['purchase_r'] = test_data['Value'].mean()
            else:
                raise ValueError("Both 'purchase_r' and 'Value' columns are missing.")
        else:
            raise ValueError(f"Missing required column: {col}")

    predictions = {}
    grouped = test_data.groupby('code') if 'code' in test_data.columns else [(None, test_data)]

    for code, group in grouped:
        scaled_data = torch.FloatTensor(group[required_columns].values).to(device)
        product_predictions = []

        seq = scaled_data[-seq_length:].unsqueeze(0)

        for _ in range(num_predictions):
            with torch.no_grad():
                pred = model(seq.to(device))
            pred = pred.squeeze().cpu().numpy()
            product_predictions.append(pred[-1])

            new_seq = torch.FloatTensor(pred[-1].reshape(1, -1)).to(device)
            seq = torch.cat((seq[:, 1:, :], new_seq.unsqueeze(0)), dim=1)

        predictions[code] = product_predictions
        print(f"Predictions for code {code}: {product_predictions}")

    return predictions