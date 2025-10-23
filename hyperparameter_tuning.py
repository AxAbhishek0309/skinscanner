import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class HyperparameterTuner:
    def __init__(self):
        self.results = []
        self.best_accuracy = 0
        self.best_config = None
        
    def create_data_generators(self, batch_size=32):
        """Create data generators - same as your working model"""
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        training_set = train_datagen.flow_from_directory(
            'train',
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        test_set = test_datagen.flow_from_directory(
            'validation',
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        return training_set, test_set
    
    def create_model(self, config):
        """Create model with given configuration"""
        
        model = Sequential()
        
        # First Conv Layer
        model.add(Conv2D(
            filters=config['filters_1'], 
            kernel_size=3, 
            activation='relu', 
            input_shape=(128, 128, 3)
        ))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        
        # Second Conv Layer  
        model.add(Conv2D(
            filters=config['filters_2'], 
            kernel_size=3, 
            activation='relu'
        ))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        
        # Add dropout if specified
        if config['dropout_conv'] > 0:
            model.add(Dropout(config['dropout_conv']))
        
        # Flatten
        model.add(Flatten())
        
        # Dense Layer
        model.add(Dense(units=config['dense_units'], activation='relu'))
        
        # Add dropout if specified
        if config['dropout_dense'] > 0:
            model.add(Dropout(config['dropout_dense']))
        
        # Output Layer
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate(self, config, config_name):
        """Train and evaluate a configuration"""
        
        print(f"\n{'='*50}")
        print(f"Testing: {config_name}")
        print(f"Config: {config}")
        print(f"{'='*50}")
        
        try:
            # Create data
            train_set, test_set = self.create_data_generators(config['batch_size'])
            
            # Create model
            model = self.create_model(config)
            
            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            # Train
            history = model.fit(
                train_set,
                validation_data=test_set,
                epochs=25,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Get results
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            best_val_acc = max(history.history['val_accuracy'])
            overfitting_gap = final_train_acc - final_val_acc
            
            result = {
                'config_name': config_name,
                'config': config,
                'final_val_acc': final_val_acc,
                'best_val_acc': best_val_acc,
                'overfitting_gap': overfitting_gap,
                'epochs_trained': len(history.history['accuracy'])
            }
            
            self.results.append(result)
            
            print(f"\nResults for {config_name}:")
            print(f"  Best Val Accuracy: {best_val_acc:.4f}")
            print(f"  Final Val Accuracy: {final_val_acc:.4f}")
            print(f"  Overfitting Gap: {overfitting_gap:.4f}")
            print(f"  Epochs Trained: {result['epochs_trained']}")
            
            # Save best model
            if best_val_acc > self.best_accuracy:
                self.best_accuracy = best_val_acc
                self.best_config = config_name
                model.save('best_hypertuned_model.h5')
                print(f"  ✅ NEW BEST MODEL SAVED!")
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return result
            
        except Exception as e:
            print(f"❌ Error with {config_name}: {str(e)}")
            return None
    
    def run_hyperparameter_search(self):
        """Run systematic hyperparameter search"""
        
        # Define configurations to test
        configurations = {
            'baseline': {
                'filters_1': 32, 'filters_2': 32,
                'dense_units': 128, 'dropout_conv': 0.0, 'dropout_dense': 0.0,
                'learning_rate': 0.001, 'batch_size': 32
            },
            'lr_low': {
                'filters_1': 32, 'filters_2': 32,
                'dense_units': 128, 'dropout_conv': 0.0, 'dropout_dense': 0.0,
                'learning_rate': 0.0001, 'batch_size': 32
            },
            'lr_medium': {
                'filters_1': 32, 'filters_2': 32,
                'dense_units': 128, 'dropout_conv': 0.0, 'dropout_dense': 0.0,
                'learning_rate': 0.0005, 'batch_size': 32
            },
            'dropout_light': {
                'filters_1': 32, 'filters_2': 32,
                'dense_units': 128, 'dropout_conv': 0.0, 'dropout_dense': 0.25,
                'learning_rate': 0.0005, 'batch_size': 32
            },
            'dropout_medium': {
                'filters_1': 32, 'filters_2': 32,
                'dense_units': 128, 'dropout_conv': 0.25, 'dropout_dense': 0.5,
                'learning_rate': 0.0005, 'batch_size': 32
            },
            'filters_64': {
                'filters_1': 64, 'filters_2': 64,
                'dense_units': 128, 'dropout_conv': 0.0, 'dropout_dense': 0.25,
                'learning_rate': 0.0005, 'batch_size': 32
            },
            'dense_256': {
                'filters_1': 32, 'filters_2': 32,
                'dense_units': 256, 'dropout_conv': 0.0, 'dropout_dense': 0.25,
                'learning_rate': 0.0005, 'batch_size': 32
            },
            'batch_16': {
                'filters_1': 32, 'filters_2': 32,
                'dense_units': 128, 'dropout_conv': 0.0, 'dropout_dense': 0.25,
                'learning_rate': 0.0005, 'batch_size': 16
            }
        }
        
        print("🚀 Starting Hyperparameter Tuning")
        print(f"Testing {len(configurations)} configurations...")
        
        # Test each configuration
        for config_name, config in configurations.items():
            self.train_and_evaluate(config, config_name)
        
        # Print final results
        self.print_final_results()
        
        return self.best_config, self.best_accuracy
    
    def print_final_results(self):
        """Print summary of all results"""
        
        print(f"\n{'='*60}")
        print("🎉 HYPERPARAMETER TUNING COMPLETED")
        print(f"{'='*60}")
        
        # Sort by best validation accuracy
        sorted_results = sorted(self.results, key=lambda x: x['best_val_acc'], reverse=True)
        
        print(f"\n🏆 BEST CONFIGURATION:")
        best = sorted_results[0]
        print(f"  Name: {best['config_name']}")
        print(f"  Best Val Accuracy: {best['best_val_acc']:.4f} ({best['best_val_acc']*100:.1f}%)")
        print(f"  Overfitting Gap: {best['overfitting_gap']:.4f}")
        print(f"  Configuration: {best['config']}")
        
        print(f"\n📊 ALL RESULTS (sorted by best validation accuracy):")
        print("-" * 80)
        print(f"{'Config':<15} {'Best Val Acc':<12} {'Overfit Gap':<12} {'Epochs':<8}")
        print("-" * 80)
        
        for result in sorted_results:
            print(f"{result['config_name']:<15} "
                  f"{result['best_val_acc']:<12.4f} "
                  f"{result['overfitting_gap']:<12.4f} "
                  f"{result['epochs_trained']:<8}")
        
        # Save results to file
        with open('hyperparameter_results.txt', 'w') as f:
            f.write("Hyperparameter Tuning Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Best Configuration: {best['config_name']}\n")
            f.write(f"Best Validation Accuracy: {best['best_val_acc']:.4f}\n")
            f.write(f"Best Parameters: {best['config']}\n\n")
            
            f.write("All Results:\n")
            for result in sorted_results:
                f.write(f"{result['config_name']}: {result['best_val_acc']:.4f} "
                       f"(Gap: {result['overfitting_gap']:.4f})\n")
        
        print(f"\n💾 Results saved to 'hyperparameter_results.txt'")
        print(f"🎯 Best model saved as 'best_hypertuned_model.h5'")

def main():
    """Main function to run hyperparameter tuning"""
    
    print("Hyperparameter Tuning for Skin Cancer Detection")
    print("Testing 8 different configurations systematically...")
    
    tuner = HyperparameterTuner()
    best_config, best_accuracy = tuner.run_hyperparameter_search()
    
    print(f"\n🎉 Tuning Complete!")
    print(f"Best Configuration: {best_config}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    return tuner

if __name__ == "__main__":
    tuner = main()