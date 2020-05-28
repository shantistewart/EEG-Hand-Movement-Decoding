# This file contains a function to train/evaluate a convolutional neural network across subjects.


from models.neural_nets.CNN import conv_neural_net


# Function description: trains and evaluates a CNN for selected subjects (separate instance for each subject).
# Inputs:
#   subject_nums = array subject numbers to evaluate
#       size: (num_subjects, )
# Outputs:
def train_eval_CNN(subject_nums):
    for subject in subject_nums:
        print("Subject {0}\n".format(subject))
