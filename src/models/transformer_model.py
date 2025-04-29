import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import os
import pickle

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head self-attention and feed-forward network"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim,
            dropout=rate
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for transformer models"""
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PasswordTransformer:
    """
    Transformer model for password sequence prediction and strength evaluation
    """
    def __init__(
        self, 
        vocab_size, 
        embedding_dim=64, 
        num_heads=4, 
        ff_dim=128,
        max_length=30,
        num_transformer_blocks=2,
        dropout_rate=0.1
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_length = max_length
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.sequence_model = None
        self.strength_model = None
        
    def build_sequence_model(self):
        """Build sequence prediction model (character-by-character)"""
        inputs = layers.Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding_layer = layers.Embedding(self.vocab_size, self.embedding_dim)
        x = embedding_layer(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.max_length, self.embedding_dim)(x)
        
        # Add transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                self.embedding_dim, 
                self.num_heads, 
                self.ff_dim, 
                self.dropout_rate
            )(x)
        
        # Output layer
        outputs = layers.Dense(self.vocab_size, activation="softmax")(x[:, -1, :])
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.sequence_model = model
        return model
    
    def build_strength_model(self):
        """Build password strength prediction model"""
        # Input layer
        inputs = layers.Input(shape=(self.max_length,))
        
        # Embedding layer with gradient clipping
        embedding = layers.Embedding(
            self.vocab_size, 
            self.embedding_dim,
            mask_zero=True
        )(inputs)
        
        # Positional encoding with learned embeddings
        pos_encoding = layers.Embedding(
            self.max_length,
            self.embedding_dim
        )(tf.range(self.max_length))
        
        # Add positional encoding to embeddings
        x = embedding + pos_encoding
        
        # Transformer blocks with increased complexity
        for _ in range(self.num_transformer_blocks):
            # Multi-head attention with more heads
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads * 2,  # Double the number of heads
                key_dim=self.embedding_dim,
                dropout=self.dropout_rate
            )(x, x)
            
            # Add & Norm
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed-forward network with more layers
            ffn = tf.keras.Sequential([
                layers.Dense(self.ff_dim * 2, activation="relu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.embedding_dim),
                layers.Dropout(self.dropout_rate)
            ])
            ffn_output = ffn(x)
            
            # Add & Norm
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with increased complexity
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with temperature scaling
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with gradient clipping and learning rate schedule
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
    print("Password Transformer model defined. To use it, load prepared password data first.") 