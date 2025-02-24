# feed-Forword
# Neural Network Visualization and Computation

This project visualizes a simple feedforward neural network and demonstrates the forward pass computations using the hyperbolic tangent (tanh) activation function.

## Code Explanation

### Importing Required Libraries
```python
import matplotlib.pyplot as plt
import numpy as np
```
We import `matplotlib.pyplot` for visualization and `numpy` for numerical computations.

### Defining the Activation Function
```python
def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1
```
This function implements the tanh activation function, which scales values between -1 and 1.

### Plotting the Neural Network Architecture
```python
def plot_network():
    fig = plt.figure(figsize=(12, 8))
```
We create a figure with a specific size for better visualization.

```python
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title('Neural Network Architecture', pad=20)
```
We define the first subplot for the neural network diagram and disable axis labels.

#### Defining Neuron Positions
```python
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
```
We define the positions of each neuron in a dictionary.

#### Drawing Connections with Weights
```python
connections = [
    ('input1', 'hidden1', w1), ('input2', 'hidden1', w2),
    ('input1', 'hidden2', w3), ('input2', 'hidden2', w4),
    ('hidden1', 'output1', w5), ('hidden2', 'output1', w6),
    ('hidden1', 'output2', w7), ('hidden2', 'output2', w8),
    ('b1', 'hidden1', b1), ('b1', 'hidden2', b1),
    ('b2', 'output1', b2), ('b2', 'output2', b2)
]
```
We define the neural network connections along with corresponding weights.

```python
for connection in connections:
    start = positions[connection[0]]
    end = positions[connection[1]]
    weight = connection[2]
    color = 'green' if weight > 0 else 'red'
    linewidth = abs(weight) * 5
    ax1.annotate("", xy=end, xytext=start,
                 arrowprops=dict(arrowstyle="->", color=color, linewidth=linewidth, alpha=0.7))
```
We iterate through the connections and draw arrows representing weighted links between neurons. Green indicates positive weights, and red indicates negative weights.

#### Drawing Neurons
```python
for node, pos in positions.items():
    color = 'skyblue' if node.startswith('input') else \
            'lightgreen' if node.startswith('hidden') else \
            'gold' if node.startswith('output') else 'gray'
    ax1.add_patch(plt.Circle(pos, circle_radius, color=color, ec='black', lw=2))
    ax1.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
```
Each neuron is drawn as a colored circle with a label.

### Activation Function Visualization
```python
ax2 = fig.add_subplot(1, 2, 2)
x = np.linspace(-5, 5, 100)
y = tanh(x)
ax2.plot(x, y, lw=2, label='tanh activation')
ax2.set_title('Tanh Activation Function')
ax2.grid(True)
```
We plot the tanh activation function and add a grid for clarity.

### Initializing Weights and Biases
```python
w1, w2, w3, w4 = 0.15, 0.2, 0.35, 0.45
w5, w6, w7, w8 = 0.5, 0.1, 0.33, 0.23
b1, b2 = 0.5, 0.7
i1, i2 = 1, 1
```
These are the predefined weight and bias values for the forward pass.

### Forward Pass Computations
```python
neth1 = w1 * i1 + w2 * i2 + b1
neth2 = w3 * i1 + w4 * i2 + b1
outh1 = tanh(neth1)
outh2 = tanh(neth2)
neto1 = w5 * outh1 + w6 * outh2 + b2
neto2 = w7 * outh1 + w8 * outh2 + b2
o1 = tanh(neto1)
o2 = tanh(neto2)
```
We compute the weighted sum for hidden and output layers and apply the tanh activation function.

### Printing Computation Results
```python
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
```
A formatted table prints all intermediate and final outputs.

### Generating Visualizations
```python
plot_network()
plt.show()
```
Finally, we call the `plot_network()` function and display the visualizations.

## Summary
This script:
1. Implements a simple feedforward neural network with one hidden layer.
2. Uses the tanh activation function.
3. Visualizes the neural network architecture and activation function.
4. Computes and prints forward pass results.



