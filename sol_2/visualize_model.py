import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from house_predict_pytorch_my import CustomDeepNeuralNetwork
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import platform

# Font configuration (no longer needed for English, but kept for compatibility)
def setup_font():
    """Setup matplotlib font configuration"""
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

setup_font()


def print_model_summary(model, input_size):
    """
    Print detailed summary information of the model
    """
    print("=" * 80)
    print("Neural Network Model Detailed Summary")
    print("=" * 80)
    
    # Print model structure
    print("\nModel Structure:")
    print(model)
    
    # Calculate parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\nParameter Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-trainable Parameters: {non_trainable_params:,}")
    
    # Calculate model size (MB)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    print(f"\nModel Size:")
    print(f"  Parameter Memory: {param_size / (1024 * 1024):.2f} MB")
    print(f"  Buffer Memory: {buffer_size / (1024 * 1024):.2f} MB")
    print(f"  Total Model Size: {model_size_mb:.2f} MB")
    
    # Layer-wise statistics
    print(f"\nLayer-wise Parameter Statistics:")
    print("-" * 80)
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
            num_params = sum(p.numel() for p in module.parameters())
            if isinstance(module, nn.Linear):
                print(f"  Layer {layer_idx}: {name}")
                print(f"    Type: Linear")
                print(f"    Input Features: {module.in_features}")
                print(f"    Output Features: {module.out_features}")
                print(f"    Parameter Count: {num_params:,}")
                print(f"    Weight Shape: {module.weight.shape}")
                print(f"    Bias Shape: {module.bias.shape if module.bias is not None else 'None'}")
                layer_idx += 1
            elif isinstance(module, nn.BatchNorm1d):
                print(f"  BatchNorm: {name}")
                print(f"    Features: {module.num_features}")
                print(f"    Parameter Count: {num_params:,}")
            print()
    
    print("=" * 80)


def visualize_model_architecture(model, input_size, save_path='model_architecture.png'):
    """
    Visualize model architecture using matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Get model configuration
    hidden_sizes = model.hidden_sizes
    output_size = model.output_size
    dropout_rate = model.dropout_rate
    use_batch_norm = model.use_batch_norm
    
    # Build layer information
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # Parse network layers
    layer_idx = 0
    for i, module in enumerate(model.network):
        if isinstance(module, nn.Linear):
            layers.append({
                'type': 'Linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
                'name': f'Linear_{layer_idx}'
            })
            layer_idx += 1
        elif isinstance(module, nn.BatchNorm1d):
            layers.append({
                'type': 'BatchNorm',
                'features': module.num_features,
                'name': 'BatchNorm'
            })
        elif isinstance(module, nn.ReLU):
            layers.append({'type': 'ReLU', 'name': 'ReLU'})
        elif isinstance(module, nn.Tanh):
            layers.append({'type': 'Tanh', 'name': 'Tanh'})
        elif isinstance(module, nn.LeakyReLU):
            layers.append({'type': 'LeakyReLU', 'name': 'LeakyReLU'})
        elif isinstance(module, nn.Dropout):
            layers.append({'type': 'Dropout', 'rate': dropout_rate, 'name': 'Dropout'})
        elif isinstance(module, nn.Sigmoid):
            layers.append({'type': 'Sigmoid', 'name': 'Sigmoid'})
    
    # Draw layers
    y_positions = np.linspace(10, 1, len(layers))
    x_center = 5
    
    # Color mapping
    color_map = {
        'Linear': '#4A90E2',
        'BatchNorm': '#7ED321',
        'ReLU': '#F5A623',
        'Tanh': '#F5A623',
        'LeakyReLU': '#F5A623',
        'Dropout': '#BD10E0',
        'Sigmoid': '#D0021B'
    }
    
    boxes = []
    for i, layer in enumerate(layers):
        y = y_positions[i]
        
        # Determine box size and color
        if layer['type'] == 'Linear':
            width, height = 2.5, 0.6
            label = f"{layer['name']}\n{layer['in_features']} → {layer['out_features']}"
            color = color_map['Linear']
        elif layer['type'] == 'BatchNorm':
            width, height = 2.0, 0.5
            label = f"BatchNorm\n{layer['features']}"
            color = color_map['BatchNorm']
        elif layer['type'] in ['ReLU', 'Tanh', 'LeakyReLU']:
            width, height = 1.8, 0.5
            label = layer['type']
            color = color_map[layer['type']]
        elif layer['type'] == 'Dropout':
            width, height = 1.8, 0.5
            label = f"Dropout\n({layer['rate']})"
            color = color_map['Dropout']
        elif layer['type'] == 'Sigmoid':
            width, height = 1.8, 0.5
            label = 'Sigmoid'
            color = color_map['Sigmoid']
        else:
            width, height = 1.8, 0.5
            label = layer['type']
            color = '#CCCCCC'
        
        # Draw box
        box = FancyBboxPatch(
            (x_center - width/2, y - height/2),
            width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(box)
        boxes.append((x_center, y, width, height))
        
        # Add text
        ax.text(x_center, y, label, 
               ha='center', va='center',
               fontsize=9, fontweight='bold',
               color='white' if layer['type'] in ['Linear', 'BatchNorm', 'Dropout'] else 'black')
        
        # Draw arrow (except for the last layer)
        if i < len(layers) - 1:
            arrow = FancyArrowPatch(
                (x_center, y - height/2),
                (x_center, y_positions[i+1] + boxes[i+1][3]/2 if i+1 < len(boxes) else y_positions[i+1] + 0.3),
                arrowstyle='->',
                mutation_scale=20,
                linewidth=2,
                color='#333333',
                alpha=0.6
            )
            ax.add_patch(arrow)
    
    # Add title and description
    ax.text(5, 11.5, 'Neural Network Model Architecture', 
           ha='center', va='center',
           fontsize=18, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=color_map['Linear'], label='Linear Layer'),
        mpatches.Patch(facecolor=color_map['BatchNorm'], label='Batch Normalization'),
        mpatches.Patch(facecolor=color_map['ReLU'], label='Activation Function'),
        mpatches.Patch(facecolor=color_map['Dropout'], label='Dropout'),
        mpatches.Patch(facecolor=color_map['Sigmoid'], label='Sigmoid Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add model information
    info_text = f"Input Dimension: {input_size}\n"
    info_text += f"Hidden Layers: {hidden_sizes}\n"
    info_text += f"Output Dimension: {output_size}\n"
    info_text += f"Dropout Rate: {dropout_rate}\n"
    info_text += f"BatchNorm: {'Yes' if use_batch_norm else 'No'}"
    
    ax.text(8.5, 6, info_text,
           ha='left', va='center',
           fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nModel architecture diagram saved to: {save_path}")
    plt.close()


def visualize_model_parameters(model, save_path='model_parameters.png'):
    """
    Visualize model parameter distributions
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect parameters from all layers
    linear_layers = []
    weights_data = []
    biases_data = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)
            weights_data.append(module.weight.data.cpu().numpy().flatten())
            if module.bias is not None:
                biases_data.append(module.bias.data.cpu().numpy().flatten())
    
    # 1. Weight distribution histogram
    ax = axes[0, 0]
    for i, weights in enumerate(weights_data):
        ax.hist(weights, bins=50, alpha=0.6, label=f'Layer {i+1}')
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Weight Distribution by Layer', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Bias distribution histogram
    ax = axes[0, 1]
    for i, biases in enumerate(biases_data):
        ax.hist(biases, bins=50, alpha=0.6, label=f'Layer {i+1}')
    ax.set_xlabel('Bias Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Bias Distribution by Layer', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Parameter count bar chart per layer
    ax = axes[1, 0]
    layer_names = [f'Layer {i+1}' for i in range(len(linear_layers))]
    param_counts = []
    for name in linear_layers:
        module = dict(model.named_modules())[name]
        param_count = sum(p.numel() for p in module.parameters())
        param_counts.append(param_count)
    
    bars = ax.bar(layer_names, param_counts, color='#4A90E2', alpha=0.7)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Parameter Count', fontsize=11)
    ax.set_title('Parameter Count per Layer', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}',
               ha='center', va='bottom', fontsize=9)
    
    # 4. Weight statistics box plot
    ax = axes[1, 1]
    box_data = [weights for weights in weights_data]
    bp = ax.boxplot(box_data, tick_labels=[f'L{i+1}' for i in range(len(box_data))],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4A90E2')
        patch.set_alpha(0.7)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Weight Value', fontsize=11)
    ax.set_title('Weight Statistics Distribution by Layer', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Parameter Visualization Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Parameter distribution diagram saved to: {save_path}")
    plt.close()


def visualize_model_flowchart(model, input_size, save_path='model_flowchart.png'):
    """
    Draw model data flow diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    hidden_sizes = model.hidden_sizes
    layer_sizes = [input_size] + hidden_sizes + [model.output_size]
    
    # Calculate positions
    num_layers = len(layer_sizes)
    x_positions = np.linspace(1, 14, num_layers)
    y_center = 4
    
    # Draw each layer
    for i, size in enumerate(layer_sizes):
        x = x_positions[i]
        
        # Determine layer type and color
        if i == 0:
            layer_type = 'Input'
            color = '#50C878'
            width, height = 2, 1.5
        elif i == len(layer_sizes) - 1:
            layer_type = 'Output'
            color = '#FF6B6B'
            width, height = 2, 1.5
        else:
            layer_type = 'Hidden'
            color = '#4A90E2'
            width, height = 2.5, 1.5
        
        # Draw layer box
        box = FancyBboxPatch(
            (x - width/2, y_center - height/2),
            width, height,
            boxstyle="round,pad=0.15",
            edgecolor='black',
            facecolor=color,
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        
        # Add text
        if i == 0:
            label = f'Input\n{size} features'
        elif i == len(layer_sizes) - 1:
            label = f'Output\n{size} neuron'
        else:
            label = f'Hidden {i}\n{size} neurons'
        
        ax.text(x, y_center, label,
               ha='center', va='center',
               fontsize=10, fontweight='bold',
               color='white')
        
        # Draw arrow
        if i < len(layer_sizes) - 1:
            arrow = FancyArrowPatch(
                (x + width/2, y_center),
                (x_positions[i+1] - width/2, y_center),
                arrowstyle='->',
                mutation_scale=25,
                linewidth=2.5,
                color='#333333',
                alpha=0.7
            )
            ax.add_patch(arrow)
            
            # Add operation labels
            mid_x = (x + x_positions[i+1]) / 2
            operations = ['Linear', 'BatchNorm', 'ReLU', 'Dropout']
            for j, op in enumerate(operations):
                if j < 3:  # Only show first 3 operations
                    ax.text(mid_x, y_center + 1.2 - j*0.3, op,
                           ha='center', va='center',
                           fontsize=7, style='italic',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))
    
    # Add title
    ax.text(8, 7.5, 'Neural Network Data Flow Diagram', 
           ha='center', va='center',
           fontsize=18, fontweight='bold')
    
    # Add description
    info_text = f"Total Layers: {num_layers} (Input + {len(hidden_sizes)} Hidden + Output)\n"
    info_text += f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}"
    
    ax.text(8, 0.5, info_text,
           ha='center', va='center',
           fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Data flow diagram saved to: {save_path}")
    plt.close()


def create_model_comparison_table(model, save_path='model_comparison.txt'):
    """
    Create model comparison table (text format)
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Neural Network Model Detailed Comparison Table\n")
        f.write("=" * 80 + "\n\n")
        
        # Model basic information
        f.write("Model Basic Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Input Dimension: {model.input_size}\n")
        f.write(f"Hidden Layer Configuration: {model.hidden_sizes}\n")
        f.write(f"Output Dimension: {model.output_size}\n")
        f.write(f"Dropout Rate: {model.dropout_rate}\n")
        f.write(f"Use BatchNorm: {model.use_batch_norm}\n\n")
        
        # Layer-wise detailed information
        f.write("Layer-wise Detailed Information:\n")
        f.write("-" * 80 + "\n")
        
        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                f.write(f"\nLayer {layer_idx + 1}: {name}\n")
                f.write(f"  Type: Linear (Fully Connected Layer)\n")
                f.write(f"  Input Features: {module.in_features}\n")
                f.write(f"  Output Features: {module.out_features}\n")
                f.write(f"  Weight Shape: {module.weight.shape}\n")
                f.write(f"  Weight Parameter Count: {module.weight.numel():,}\n")
                if module.bias is not None:
                    f.write(f"  Bias Shape: {module.bias.shape}\n")
                    f.write(f"  Bias Parameter Count: {module.bias.numel():,}\n")
                else:
                    f.write(f"  Bias: None\n")
                f.write(f"  Total Parameter Count: {sum(p.numel() for p in module.parameters()):,}\n")
                layer_idx += 1
            elif isinstance(module, nn.BatchNorm1d):
                f.write(f"\n  BatchNorm Layer: {name}\n")
                f.write(f"  Features: {module.num_features}\n")
                f.write(f"  Parameter Count: {sum(p.numel() for p in module.parameters()):,}\n")
        
        # Overall statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("Overall Statistics:\n")
        f.write("-" * 80 + "\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Parameter Count: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
        
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        f.write(f"Model Size: {param_size / (1024 * 1024):.2f} MB\n")
        f.write("=" * 80 + "\n")
    
    print(f"Model comparison table saved to: {save_path}")


if __name__ == '__main__':
    print("Starting neural network model visualization...")
    print("=" * 80)
    
    # Load data to get input dimension
    print("\nLoading data to determine input dimension...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Data preprocessing (simplified version, only to get feature count)
    from house_predict_pytorch_my import preprocess_data
    train_processed, test_processed, _ = preprocess_data(train_df, test_df)
    
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    input_size = len(feature_cols)
    
    print(f"Input feature dimension: {input_size}")
    
    # Create model (using the same configuration as the original file)
    print("\nCreating model...")
    model = CustomDeepNeuralNetwork(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=1,
        dropout_rate=0.3,
        use_batch_norm=True,
        activation='relu'
    )
    
    # Print model summary
    print_model_summary(model, input_size)
    
    # Generate various visualizations
    print("\nGenerating visualization charts...")
    print("-" * 80)
    
    # 1. Model architecture diagram
    print("1. Generating model architecture diagram...")
    visualize_model_architecture(model, input_size, 'model_architecture.png')
    
    # 2. Parameter distribution diagram
    print("2. Generating parameter distribution diagram...")
    visualize_model_parameters(model, 'model_parameters.png')
    
    # 3. Data flow diagram
    print("3. Generating data flow diagram...")
    visualize_model_flowchart(model, input_size, 'model_flowchart.png')
    
    # 4. Comparison table
    print("4. Generating model comparison table...")
    create_model_comparison_table(model, 'model_comparison.txt')
    
    print("\n" + "=" * 80)
    print("All visualization files have been generated successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - model_architecture.png: Model architecture visualization diagram")
    print("  - model_parameters.png: Model parameter distribution diagram")
    print("  - model_flowchart.png: Model data flow diagram")
    print("  - model_comparison.txt: Model detailed comparison table")

