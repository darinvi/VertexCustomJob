import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

def main(args):
    
    data = pd.read_parquet(f"gs://{args.bucket_name}/{args.dataset_name}")

    '''
    
    Insert Code Here.
    
    '''

    model.save(f"gs://{args.bucket_name}/models/{args.model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    args = parser.parse_args()
    main(args)
