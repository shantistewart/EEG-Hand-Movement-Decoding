# This file contains unit tests for conv_neural_net.py (in neural_nets folder).


from models.neural_nets import conv_neural_net as conv_net


# --------------------TESTING build_model() FUNCTION--------------------
print("\n----------TESTING build_model() FUNCTION----------\n")

# hyperparameters:
input_shape = (30, 25, 3)
num_outputs = 1
num_conv_layers = 1
num_dense_layers = 1
num_filters = 2
kernel_size = 3
pool_size = 2
num_hidden_nodes = 50

# call function:
model = conv_net.build_model(input_shape, num_outputs, num_conv_layers=num_conv_layers,
                             num_dense_layers=num_dense_layers, num_filters=num_filters, kernel_size=kernel_size,
                             pool_size=pool_size, num_hidden_nodes=num_hidden_nodes)
print("\n")
# display network architecture:
model.summary()
