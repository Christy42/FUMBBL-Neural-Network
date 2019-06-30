import numpy as np

from layer import NeuralNet
# Create data
X = np.matrix([[1, 2], [3, 4]])
# Create Neural Net
N = NeuralNet(hidden_layers=2, input_nodes=2, output_nodes=3, hidden_nodes=4, input_data=X)
Theta_1 = np.matrix([[1, 4, 5, 6], [2, 3, 4, 1], [1, 2, 3, 4]])
Theta_2 = np.matrix([[1, 2, 1, 3], [2, 1, 3, 1], [1, 4, 1, 2], [1, 2, 1, 2], [3, 1, 1, 1]])
Theta_3 = np.matrix([[1, 2, 1], [2, 1, 1], [3, 1, 2], [4, 1, 2], [1, 2, 2]])
N.set_layer(1, Theta_1)
N.set_layer(2, Theta_2)
N.set_layer(3, Theta_3)
# Add Roll Forward to the Stack

# Run through the stack - this functionality needs some work
