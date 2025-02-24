import matplotlib.pyplot as plt
import numpy as np

def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1

def plot_network():
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Network visualization
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title('Neural Network Architecture', pad=20)
    
    # Draw neurons
    circle_radius = 0.15
    positions = {
        'input1': (0.1, 0.7),
        'input2': (0.1, 0.3),
        'hidden1': (0.4, 0.7),
        'hidden2': (0.4, 0.3),
        'output1': (0.7, 0.6),
        'output2': (0.7, 0.4),
        'b1': (0.4, 0.9),
        'b2': (0.7, 0.9)
    }
    
    # Draw connections with weights
    connections = [
        ('input1', 'hidden1', w1), ('input2', 'hidden1', w2),
        ('input1', 'hidden2', w3), ('input2', 'hidden2', w4),
        ('hidden1', 'output1', w5), ('hidden2', 'output1', w6),
        ('hidden1', 'output2', w7), ('hidden2', 'output2', w8),
        ('b1', 'hidden1', b1), ('b1', 'hidden2', b1),
        ('b2', 'output1', b2), ('b2', 'output2', b2)
    ]
    
    for connection in connections:
        start = positions[connection[0]]
        end = positions[connection[1]]
        weight = connection[2]
        color = 'green' if weight > 0 else 'red'
        linewidth = abs(weight) * 5
        ax1.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=color, linewidth=linewidth, alpha=0.7))
    
    # Draw neurons
    for node, pos in positions.items():
        color = 'skyblue' if node.startswith('input') else \
                'lightgreen' if node.startswith('hidden') else \
                'gold' if node.startswith('output') else 'gray'
        ax1.add_patch(plt.Circle(pos, circle_radius, color=color, ec='black', lw=2))
        ax1.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold'))
    
    # Activation function visualization
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.linspace(-5, 5, 100)
    y = tanh(x)
    ax2.plot(x, y, lw=2, label='tanh activation')
    ax2.set_title('Tanh Activation Function')
    ax2.grid(True)
    
    # Plot network values
    net_values = [neth1, neth2, neto1, neto2]
    for val in net_values:
        ax2.axvline(val, color='r', linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()

# Initialize weights and biases
w1, w2, w3, w4 = 0.15, 0.2, 0.35, 0.45
w5, w6, w7, w8 = 0.5, 0.1, 0.33, 0.23
b1, b2 = 0.5, 0.7
i1, i2 = 1, 1

# Forward pass calculations
neth1 = w1 * i1 + w2 * i2 + b1
neth2 = w3 * i1 + w4 * i2 + b1
outh1 = tanh(neth1)
outh2 = tanh(neth2)
neto1 = w5 * outh1 + w6 * outh2 + b2
neto2 = w7 * outh1 + w8 * outh2 + b2
o1 = tanh(neto1)
o2 = tanh(neto2)

# Print results with formatted table
print("┌───────────────────────┬───────────────────────┐")
print("│ Intermediate Values   │ Results               │")
print("├───────────────────────┼───────────────────────┤")
print(f"│ net_h1               │ {neth1:>20.4f} │")
print(f"│ net_h2               │ {neth2:>20.4f} │")
print(f"│ out_h1               │ {outh1:>20.4f} │")
print(f"│ out_h2               │ {outh2:>20.4f} │")
print(f"│ net_o1               │ {neto1:>20.4f} │")
print(f"│ net_o2               │ {neto2:>20.4f} │")
print("├───────────────────────┼───────────────────────┤")
print(f"│ Final Output 1       │ {o1:>20.4f} │")
print(f"│ Final Output 2       │ {o2:>20.4f} │")
print("└───────────────────────┴───────────────────────┘")

# Generate visualizations
plot_network()
plt.show()