# raspberry_pi/ml/model_loader.py
import os
import logging
import numpy as np
from pathlib import Path

class ModelLoader:
    def __init__(self, model_path):
        # Print debug information
        print(f"\nDEBUG: Received model_path: {model_path}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        
        # Get the absolute path in two different ways
        abs_path1 = Path(model_path).resolve()
        abs_path2 = (Path(__file__).parent.parent / model_path).resolve()
        
        print(f"DEBUG: Attempt 1 absolute path: {abs_path1}")
        print(f"DEBUG: Attempt 2 absolute path: {abs_path2}")
        
        # Try both possible paths
        if abs_path1.exists():
            final_path = abs_path1
        elif abs_path2.exists():
            final_path = abs_path2
        else:
            raise FileNotFoundError(
                f"Model file not found at either:\n"
                f"1. {abs_path1}\n"
                f"2. {abs_path2}\n"
                f"Current directory: {os.getcwd()}"
            )
        
        print(f"DEBUG: Using model at: {final_path}")
        
        # Rest of your loading code...
        try:
            from ai_edge_litert import Interpreter
            self.interpreter = Interpreter(model_path=str(final_path))
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(final_path))
        
        self.interpreter.allocate_tensors()
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logging.info(f"Model loaded. Input shape: {self.input_details[0]['shape']}")

    def predict(self, input_data):
        try:
            # Verify input shape
            expected_shape = self.input_details[0]['shape']
            if input_data.shape != tuple(expected_shape):
                raise ValueError(
                    f"Input shape {input_data.shape} doesn't match "
                    f"expected {expected_shape}"
                )
            
            # Set input tensor
            self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data.astype(np.float32)
            )
        
            # Run inference
            self.interpreter.invoke()
        
            return self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0][0]
        
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise

'''
def run_inference(interpreter, input_details, output_details, features):
    """Handle feature reshaping for your specific model"""
    # Reshape to model's expected input shape
    if features.ndim == 2:  # MFCC features (time, features)
        features = np.expand_dims(features, axis=0)  # Add batch dim
        if features.shape[1] > input_details[0]['shape'][1]:
            features = features[:, :input_details[0]['shape'][1], :]
        features = np.pad(
            features,
            ((0,0), (0, max(0, input_details[0]['shape'][1] - features.shape[1])), (0,0)),
            mode='constant'
        )
    
    interpreter.set_tensor(input_details[0]['index'], features.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])
'''