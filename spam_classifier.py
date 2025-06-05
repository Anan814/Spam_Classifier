import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    column_names = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
        'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order',
        'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people',
        'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business',
        'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
        'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
        'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab',
        'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
        'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999',
        'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs',
        'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
        'word_freq_edu', 'word_freq_table', 'word_freq_conference',
        'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$',
        'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
        'capital_run_length_total', 'spam'
    ]
    try:
        data = pd.read_csv(url, header=None, names=column_names)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    # Load and prepare data
    data = load_data()
    if data is None:
        return
    
    X = data.drop('spam', axis=1).values
    y = data['spam'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(57,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile and train
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    print("Training model...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=20,
                        batch_size=32,
                        verbose=1)
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")
    
    # Save model
    model.save('spam_classifier.keras')
    print("Model saved as 'spam_classifier.keras'")

if __name__ == "__main__":
    main()