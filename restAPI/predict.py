import torch
from train import predict, visualize_predictions
import joblib

model = "C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/lstm_model.pth" 

# Load the scalers
scaler_value = joblib.load('./scaler_value.pkl') # Load the scaler used for 'Value' column normalization
scaler_inventory = scaler_inventory = joblib.load('./scaler_inventory.pkl')  # Load the scaler used for 'Inventory' column normalization

# Load the data to be predicted
new_data = "final_data_prediction.csv"  # Load your new data here

# Make predictions
sequence_length=3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictions_df = predict(new_data, model, scaler_value, scaler_inventory, sequence_length, 12, device)

# Postprocess the predictions
predictions_df = predictions_df.apply(lambda x: x.astype(int))

# Output the predictions as CSV
predictions_df.to_csv('predictions.csv', index_label='code_p')

# Visualize the predictions
visualize_predictions(predictions_df)
