from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.svm import SVC
import pickle
if TYPE_CHECKING:
    pass


class EMGNet(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) designed for real-time EMG classification.
    Optimized for Time-Domain features (MAV, RMS, WL).
    """
    def __init__(self, input_size: int, num_classes: int):
        super(EMGNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class Model1:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_size = 96  # 32 Channels * 3 Features (MAV, RMS, WL)
        self.num_classes = None

    def fit(self, training_data: Dict[str, Any], testing_data: Dict[str, Any]) -> None:
        """
        Fits the model to the training data.
        """
        # Convert numpy arrays to PyTorch tensors
        x_train = torch.FloatTensor(training_data["x"]).to(self.device)
        y_train = torch.LongTensor(training_data["y"]).to(self.device)
        x_test = torch.FloatTensor(testing_data["x"]).to(self.device)
        y_test = torch.LongTensor(testing_data["y"]).to(self.device)

        # Detect number of classes
        unique_labels = torch.unique(y_train)
        self.num_classes = len(unique_labels)
        
        if self.num_classes < 2:
            print(f"Warning: Only {self.num_classes} class found ({unique_labels}). "
                  "Classification requires at least 2 different labels.")
      

        self.model = EMGNet(self.input_size, self.num_classes).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        batch_size = 32
        epochs = 100

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        print(f"Starting training on {self.device} with {self.num_classes} classes...")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Periodic validation
            if (epoch + 1) % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model(x_test)
                    _, predicted = torch.max(test_outputs, 1)
                    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    
                    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Test Acc: {accuracy:.2f}")
                self.model.train()

    def save(self, model_path: str) -> None:
        """
        Saves the model state, input size, and num_classes to model_path.
        """
        if self.model is None:
            print("Error: No model to save.")
            return

        checkpoint = {
            'state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'num_classes': self.num_classes
        }
        torch.save(checkpoint, model_path)
        print(f"Model successfully saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Loads the model from the model_path and sets it to evaluation mode.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        
        self.model = EMGNet(self.input_size, self.num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"Model successfully loaded from {model_path}")

    def predict(self, x: Any) -> List[float]:
        """
        Predicts the output for the input x.
        Returns a list of class probabilities.
        
        x: numpy array of shape (96,)
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        self.model.eval()
        with torch.no_grad():
            # Handle both single samples and batches
            x_tensor = torch.FloatTensor(x).to(self.device)
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)
            
            outputs = self.model(x_tensor)
            # Apply Softmax to get probabilities (standard in classification)
            probs = torch.softmax(outputs, dim=1)
            
            return probs.cpu().numpy().tolist()[0]
        






















class ModelMLP:
    # This is the MLP from https://link.springer.com/chapter/10.1007/978-3-032-13012-9_12 also used for online!
    def __init__(self, num_input_features = None) -> None:
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(
            hidden_layer_sizes=(128,64,8),
            activation='tanh',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=1e-5,
            max_iter=1000,
            tol=1e-8
            )
        
        if num_input_features is not None:
            self.input_size = num_input_features
        else:
            self.input_size = 144  # 32 Channels * 3 Features (MAV, RMS, WL)
            print("Gibscht mir bidde die features, danke!")

        self.num_classes = None

    def fit(self, training_data: Dict[str, Any], testing_data: Dict[str, Any] | None = None) -> None:
        """
        Fits the model to the training data.
        """

        # Detect number of classes
        unique_labels = np.unique(training_data.get("y"))
        self.num_classes = len(unique_labels)
        
        if self.num_classes < 2:
            print(f"Warning: Only {self.num_classes} class found ({unique_labels}). "
                "Classification requires at least 2 different labels.")

        print(f"Starting training MLP with {self.num_classes} classes...")

        self.model.fit(training_data.get("x"), training_data.get("y"))
        

    def save(self, model_path: str) -> None:
        """
        Saves the model state, input size, and num_classes to model_path.
        """
        if self.model is None:
            print("Error: No model to save.")
            return

        checkpoint = {
            'model': self.model,
            'input_size': self.input_size,
            'num_classes': self.num_classes
        }

        with open(model_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        print(f"Model successfully saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Loads the model from the model_path.
        """
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        
        print(f"Model successfully loaded from {model_path}")

    def predict(self, x: Any) -> List[float]:
        """
        Predicts the output for the input x.
        Returns a list of class probabilities.
        
        x: numpy array of shape (96,)
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # SVC expects 2D array
        x_2d = np.array(x).reshape(1, -1)
        # Get probabilities
        probs = self.model.predict_proba(x_2d)
        return probs[0].tolist()


        




















class ModelSVM:
    # This is the SVM / SVC model
    def __init__(self, num_features = None) -> None:
        self.model = SVC(kernel="rbf", probability=True)
        self.input_size = 96  # 32 Channels * 3 Features (MAV, RMS, WL)
        self.num_classes = None

    def fit(self, training_data: Dict[str, Any], testing_data: Dict[str, Any]) -> None:
        """
        Fits the model to the training data.
        """

        # Detect number of classes
        unique_labels = np.unique(training_data.get("y"))
        self.num_classes = len(unique_labels)
        
        if self.num_classes < 2:
            print(f"Warning: Only {self.num_classes} class found ({unique_labels}). "
                "Classification requires at least 2 different labels.")
    
        batch_size = 32
        epochs = 100

        print(f"Starting training SVC with {self.num_classes} classes...")

        self.model.fit(training_data.get("x"), training_data.get("y"))
        

    def save(self, model_path: str) -> None:
        """
        Saves the model state, input size, and num_classes to model_path.
        """
        if self.model is None:
            print("Error: No model to save.")
            return

        checkpoint = {
            'model': self.model,
            'input_size': self.input_size,
            'num_classes': self.num_classes
        }

        with open(model_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        print(f"Model successfully saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Loads the model from the model_path.
        """
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        
        print(f"Model successfully loaded from {model_path}")

    def predict(self, x: Any) -> List[float]:
        """
        Predicts the output for the input x.
        Returns a list of class probabilities.
        
        x: numpy array of shape (96,)
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")

        # SVC expects 2D array
        x_2d = np.array(x).reshape(1, -1)
        # Get probabilities
        probs = self.model.predict_proba(x_2d)
        return probs[0].tolist()
