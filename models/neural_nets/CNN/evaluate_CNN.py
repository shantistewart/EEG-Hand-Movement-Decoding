# This file contains a function to train/evaluate a convolutional neural network across subjects.


from models.neural_nets.CNN import conv_neural_net


# Function description: trains different CNN's for all subjects and calculates average validation accuracy across all
#   subjects.
# Inputs:
#   subject_nums = list of subject numbers to evaluate
# Outputs:
def train_eval_CNN(subject_nums):
    for subject in subject_nums:
        print("Subject {0}\n".format(subject))
