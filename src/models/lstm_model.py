import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model, layers, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import os
import pickle

class PasswordLSTM:
    """
    LSTM model for password sequence prediction and strength evaluation
    """
    def __init__(
        self, 
        vocab_size, 
        embedding_dim=64, 
        lstm_units=128, 
        max_length=30,
        dropout_rate=0.2
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.model = None
        self.sequence_model = None
        self.strength_model = None
        
    def build_sequence_model(self):
        """Build sequence prediction model (character-by-character)"""
        model = Sequential()
        model.add(layers.Embedding(
            self.vocab_size, 
            self.embedding_dim, 
            input_length=self.max_length,
            mask_zero=True
        ))
        model.add(layers.Bidirectional(layers.LSTM(
            self.lstm_units, 
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        )))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Bidirectional(layers.LSTM(
            self.lstm_units,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        )))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(self.vocab_size, activation='softmax'))
        
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,
            clipvalue=0.5
        )
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        self.sequence_model = model
        return model
    
    def build_strength_model(self):
        """Build password strength prediction model"""
        model = Sequential()
        
        # Embedding layer with masking
        model.add(layers.Embedding(
            self.vocab_size, 
            self.embedding_dim, 
            input_length=self.max_length,
            mask_zero=True
        ))
        
        # First LSTM layer with more units and return sequences
        model.add(layers.Bidirectional(layers.LSTM(
            self.lstm_units * 2,  # Double the units
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        )))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Second LSTM layer with attention
        lstm_output = layers.Bidirectional(layers.LSTM(
            self.lstm_units,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        ))
        model.add(lstm_output)
        model.add(layers.Dropout(self.dropout_rate))
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_output)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(self.lstm_units * 2)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = layers.Multiply()([lstm_output, attention])
        sent_representation = layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(sent_representation)
        
        # Dense layers with batch normalization
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.dropout_rate))
        
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer with temperature scaling
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile with custom optimizer settings
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,
            clipvalue=0.5
        )
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['mae']
        )
        
        self.strength_model = model
        return model
    
    def prepare_sequence_data(self, passwords_tokens):
        """
        Prepare data for sequence prediction
        For each password, create pairs of (sequence[:i], sequence[i])
        """
        X, y = [], []
        
        for tokens in passwords_tokens:
            for i in range(1, len(tokens)):
                # Pad the input sequence to max_length
                padded_sequence = tokens[:i] + [0] * (self.max_length - i)
                
                # Ensure we don't exceed max_length
                if i <= self.max_length:
                    X.append(padded_sequence[:self.max_length])
                    y.append(tokens[i])
        
        return np.array(X), np.array(y)
    
    def prepare_strength_data(self, passwords_tokens, strength_scores):
        """
        Prepare data for strength prediction
        Pad all passwords to max_length
        """
        X = []
        
        for tokens in passwords_tokens:
            # Truncate or pad the sequence to max_length
            padded_sequence = tokens[:self.max_length] if len(tokens) >= self.max_length else tokens + [0] * (self.max_length - len(tokens))
            X.append(padded_sequence)
        
        return np.array(X), np.array(strength_scores)
    
    def train_sequence_model(
        self, 
        X_train, 
        y_train, 
        X_val=None, 
        y_val=None,
        batch_size=128, 
        epochs=20,
        save_path=None
    ):
        """Train the sequence prediction model"""
        if self.sequence_model is None:
            self.build_sequence_model()
        
        # Callbacks
        model_callbacks = []
        
        if X_val is not None and y_val is not None:
            model_callbacks.append(callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model_callbacks.append(callbacks.ModelCheckpoint(
                save_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            ))
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.sequence_model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=model_callbacks
        )
        
        return history
    
    def train_strength_model(
        self, 
        X_train, 
        y_train, 
        X_val=None, 
        y_val=None,
        batch_size=128, 
        epochs=20,
        save_path=None
    ):
        """Train the strength prediction model"""
        if self.strength_model is None:
            self.build_strength_model()
        
        # Callbacks
        model_callbacks = []
        
        if X_val is not None and y_val is not None:
            model_callbacks.append(callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model_callbacks.append(callbacks.ModelCheckpoint(
                save_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            ))
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.strength_model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=model_callbacks
        )
        
        return history
    
    def predict_next_chars(self, seed_password_tokens, num_chars=10, temperature=1.0):
        """Generate password by predicting next characters given a seed"""
        if self.sequence_model is None:
            raise ValueError("Sequence model not trained yet")
        
        # Start with the seed
        current_tokens = seed_password_tokens.copy()
        
        for _ in range(num_chars):
            # Prepare the input
            padded_tokens = current_tokens[-self.max_length:] if len(current_tokens) >= self.max_length else current_tokens + [0] * (self.max_length - len(current_tokens))
            x_pred = np.array([padded_tokens[:self.max_length]])
            
            # Make prediction
            preds = self.sequence_model.predict(x_pred)[0]
            
            # Apply temperature to control randomness
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            
            # Sample from the distribution
            probas = np.random.multinomial(1, preds, 1)[0]
            next_char_idx = np.argmax(probas)
            
            # Add the predicted character
            current_tokens.append(next_char_idx)
        
        return current_tokens[len(seed_password_tokens):]
    
    def predict_strength(self, password_tokens):
        """Predict the strength of a given password"""
        if self.strength_model is None:
            raise ValueError("Strength model not trained yet")
        
        # Prepare the input
        padded_tokens = password_tokens[:self.max_length] if len(password_tokens) >= self.max_length else password_tokens + [0] * (self.max_length - len(password_tokens))
        x_pred = np.array([padded_tokens])
        
        # Make prediction
        strength_pred = self.strength_model.predict(x_pred, verbose=0)[0][0]
        
        return strength_pred
    
    def evaluate_sequence_model(self, X_test, y_test):
        """Evaluate the sequence prediction model"""
        if self.sequence_model is None:
            raise ValueError("Sequence model not trained yet")
        
        # Get predictions
        y_pred_probs = self.sequence_model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'loss': self.sequence_model.evaluate(X_test, y_test, verbose=0)[0]
        }
    
    def evaluate_strength_model(self, X_test, y_test):
        """Evaluate the strength prediction model"""
        if self.strength_model is None:
            raise ValueError("Strength model not trained yet")
        
        # Get predictions
        y_pred = self.strength_model.predict(X_test, verbose=0)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def save_models(self, sequence_model_path=None, strength_model_path=None):
        """Save models to disk"""
        if self.sequence_model is not None and sequence_model_path:
            self.sequence_model.save(sequence_model_path)
            print(f"Sequence model saved to {sequence_model_path}")
            
        if self.strength_model is not None and strength_model_path:
            self.strength_model.save(strength_model_path)
            print(f"Strength model saved to {strength_model_path}")
    
    def load_models(self, sequence_model_path=None, strength_model_path=None):
        """Load models from disk"""
        if sequence_model_path and os.path.exists(sequence_model_path):
            self.sequence_model = tf.keras.models.load_model(sequence_model_path)
            print(f"Sequence model loaded from {sequence_model_path}")
            
        if strength_model_path and os.path.exists(strength_model_path):
            self.strength_model = tf.keras.models.load_model(strength_model_path)
            print(f"Strength model loaded from {strength_model_path}")

if __name__ == "__main__":
    # Example usage (would need data preparation first)
    print("Password LSTM model defined. To use it, load prepared password data first.") 