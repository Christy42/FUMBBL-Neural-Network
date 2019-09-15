import numpy as np

from layer import NeuralNet
from procedures.procedure import ForwardProp, BackwardProp, Cost
# Create data
X = np.matrix([[0.5403, -0.4161], [-0.99, -0.6536], [0.2837, 0.9602]])
# Create Neural Net
output = np.matrix([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
N = NeuralNet(hidden_layers=2, input_nodes=2, output_nodes=3, hidden_nodes=2, input_data=X, output_data=output)
# Theta_1 = np.matrix([[1, 4, 5, 6], [2, 3, 4, 1], [1, 2, 3, 4]])
Theta_1 = np.matrix([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
# Theta_2 = np.matrix([[1, 2, 1, 3], [2, 1, 3, 1], [1, 4, 1, 2], [1, 2, 1, 2], [3, 1, 1, 1]])
Theta_2 = np.matrix([[0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]])
Theta_3 = np.matrix([[1, 2, 1], [2, 1, 1], [3, 1, 2], [4, 1, 2], [1, 2, 2]])

N.set_layer(1, Theta_1)
N.set_layer(2, Theta_2)
# N.set_layer(3, Theta_3)
# Add Roll Forward to the Stack

final = Cost(N)
roll_forward = ForwardProp(N)
# back_prop = BackwardProp(N)

# Run through the stack - this functionality needs some work
N.step()

print(N.layers[-2].nodes)