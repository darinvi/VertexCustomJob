from re import X
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using GPU:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, training will use CPU.")

def main(args):
    
    data = pd.read_parquet(f"gs://{args.bucket_name}/{args.dataset_name}")

    ''' INSERT CUSTOM CODE HERE '''

    # e.g
    X = data.drop(columns=['y'])
    y = data['y']

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X, y, epochs=5)

    print(model.evaluate(X, y))

    ''' CUSTOM CODE ENDS HERE '''
    
    model.save(f"gs://{args.bucket_name}/models/{args.model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    args = parser.parse_args()
    main(args)
