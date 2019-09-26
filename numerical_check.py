import copy

from procedures.procedure import Cost, ForwardProp


def numeric(neural_net, epsilon):
    for i in range(neural_net.size):
        for j in range(neural_net.layers[i].theta.shape[0]):
            for k in range(neural_net.layers[i].theta.shape[1]):
                new_net = copy.deepcopy(neural_net)
                new_net.amend_theta(i, j, k, epsilon)
                Cost(new_net)
                ForwardProp(new_net)
                new_net.step()
                c_1 = new_net.cost
                new_net = copy.deepcopy(neural_net)
                new_net.amend_theta(i, j, k, -epsilon)
                Cost(new_net)
                ForwardProp(new_net)
                new_net.step()
                c_2 = new_net.cost
                print((i, j, k, c_1, c_2, (c_1 - c_2) / 2 * epsilon))
