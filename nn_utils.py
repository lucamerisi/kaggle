import torch
from torch.utils.data import DataLoader, TensorDataset

def training_loop(epochs, batch_size, model, criterion, optimizer, X_train_t, y_train_t, X_val_t, y_val_t):
    # Dataset and DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch

            # Forward pass
            outputs = model(inputs).squeeze()  # Remove extra dimension
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            model_0_preds = model(X_val_t).squeeze()
            # print(model_0_preds.shape)
            # print(y_val_t.shape)
            test_loss = criterion(model_0_preds, y_val_t)

        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')