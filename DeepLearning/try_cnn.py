from dubnet import *

# NOTE THAT WHEN GOING FROM LINUX TO MAC, YOU MUST ALSO CLEAN THE JCR LIB

# def conv_net():
#     l = [   make_convolutional_layer(3, 8, 3, 1),
#             make_activation_layer(RELU),
#             make_maxpool_layer(3, 2),
#
#             make_convolutional_layer(8, 16, 3, 1),
#             make_activation_layer(RELU),
#             make_maxpool_layer(3, 2),
#
#             make_convolutional_layer(16, 32, 3, 1),
#             make_activation_layer(RELU),
#             make_maxpool_layer(3, 2),
#
#             make_convolutional_layer(32, 64, 3, 1),
#             make_activation_layer(RELU),
#             make_maxpool_layer(3, 2),
#
#
#             make_connected_layer(256, 10),
#             make_activation_layer(SOFTMAX)]
#     return make_net(l)

def conv_net():
        l = [   make_convolutional_layer(3, 8, 3, 1),
                make_activation_layer(RELU),
                make_maxpool_layer(3, 2),

                make_convolutional_layer(8, 16, 3, 1),
                make_activation_layer(RELU),
                make_maxpool_layer(3, 2),

                make_convolutional_layer(16, 32, 3, 1),
                make_activation_layer(RELU),
                make_maxpool_layer(3, 2),

                make_convolutional_layer(32, 64, 3, 1),
                make_activation_layer(RELU),
                make_maxpool_layer(3, 2),

                make_convolutional_layer(64, 128, 3, 1),
                make_activation_layer(RELU),
                make_maxpool_layer(3, 2),


                make_connected_layer(128, 10),
                make_activation_layer(SOFTMAX)]
        return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")


# original rate value is 0.1
# original momentum value is 0.9
# original decay value is 0.005
print("making model...")
batch = 128
iters = 1000
rate = .03
momentum = .6
decay = .008

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

