import numpy as np
import random
from my_package.sampler import sampler
import random
import copy
import sys
from termcolor import colored


"""Given a context function finds the action with max reward"""
def get_best_action(context: np.ndarray) -> int:
    return np.argmax(context[0])

"""Given an action in an action the function recieves reward and updates average reward and 
    increments the counter for the action taken"""
def update_estimated_avg_reward(context: np.ndarray, action: int, reward_sampler: sampler) -> None: 
    reward = reward_sampler.sample(int(context[1][action]))
    context[0][action] = ((context[0][action]*context[2][action]) + reward)/(context[2][action]+1)
    context[2][action] += 1

"""Given a context function computes the total reward recieved till now"""
def compute_total_reward_of_context(context: np.ndarray)->float:
    sum = 0
    for i in range(context.shape[1]):
        sum += context[0][i]*context[2][i]
    return round(sum,3)

"""Given a context the functions computes the average reward over recieved all actions"""
def compute_average_reward_of_context(context: np.ndarray)->float:
    total_reward = compute_total_reward_of_context(context=context)
    total_sampled = np.sum(a=context[2])
    return round((total_reward/total_sampled),3)

"""Prints Quantities of interest for a context"""
def print_bandit_status(context: np.ndarray)->None:
    for name,obj in globals().items():
        if obj is context:
            print(colored(f"{name}: ","red"))
            print(f"Average rewards of {name} bandit:\n", context[0], "\n")
            print(f"Best action of {name} bandit:\t", np.argmax(context[0]) + 1,"\n")
            print(f"Full display of {name} bandit:\n", context, "\n")
            print(f"Total Reward for the {name} bandit:\t",\
                   f"{compute_total_reward_of_context(context=context) :.3f}","\n")
            print(f"Average Reward of {name} bandit:\t",
                  compute_average_reward_of_context(context=context),"\n")
    print("\n\n\n")

"""Function implements the softmax formula, has input as np.array and out put as np.array"""
def softmax(values: np.array, Temperature: float)->np.array:
    # temporary array for computations
    temp = copy.deepcopy(values)
    temp = np.divide(temp,Temperature)
    # temp is one dimensional but temp.shape is a one dimensioal tuple with only a single number in it
    answer_array = np.zeros(temp.shape[0])
    denominator = np.sum(np.exp(temp))
    for i in range(temp.shape[0]):
        answer_array[i] = np.exp(temp[i])/denominator
    return answer_array


"""Given a context chooses an action based on softmax policy"""
def choose_action(Temperature: float,context: np.ndarray)->int :
    rand_num = np.random.random(1)
    softmax_arr  = softmax(Temperature=Temperature,values=context[0])
    
    # creating cdf
    incremental_sum = np.zeros(softmax_arr.shape[0])
    incremental_sum[0] = softmax_arr[0]
    for i in range(1,softmax_arr.shape[0]):
        incremental_sum[i] = incremental_sum[i-1] + softmax_arr[i]

    # now choosing action
    if rand_num <= incremental_sum[0]:
        return 0
    elif rand_num <= incremental_sum[1]:
        return 1
    elif rand_num <= incremental_sum[2]:
        return 2
    else:
        return 3


"""
Function trains the softmax bandit for number of iterations in an EPOCH
Updates all the contexts
"""
def run_softmax(contexts: list, iterations: int, Temperature: float,\
                reward_sampler: sampler)->np.ndarray:
    for i in range(iterations):
        context = random.choice(contexts)
        action = choose_action(context=context,Temperature=Temperature)
        update_estimated_avg_reward(context=context,action=action,reward_sampler=reward_sampler)
    return contexts


"""Runs over many epochs and averages output"""
def run_epochs(epochs: int, all_contexts: list, reward_sampler: sampler,\
                Temperature: float = 1, one_epoch_iterations: int = 1000)->list:
    Users = copy.deepcopy(all_contexts)
    temp = copy.deepcopy(all_contexts)

    for i in range(epochs):
        run_softmax(contexts=Users,iterations=one_epoch_iterations,\
                   Temperature=Temperature,reward_sampler=reward_sampler)
        temp[0] += Users[0]
        temp[1] += Users[1]
        temp[2] += Users[2]
        Users = copy.deepcopy(all_contexts)

    all_contexts[0] = np.divide(temp[0],epochs)
    all_contexts[1] = np.divide(temp[1],epochs)
    all_contexts[2] = np.divide(temp[2],epochs)
    return all_contexts 


""" 
Note: the structre of the context Array

ROWS:
The 0th row is the average reward values
The 1st row contains the corresponding number for sample reward
The third row contains the number of times the action is sampled

COLUMNS:
The Columns represent the following actions in order:
0th> Entertainment
1st> Education
2nd> Tech
3rd> Crime

"""

third_row_values = np.zeros(4)
second_column_values = np.array([0,1,2,3])
User_1 = np.vstack((np.zeros(4),second_column_values,third_row_values)) # row stack
second_column_values = np.array([4,5,6,7])
User_2 = np.vstack((np.zeros(4),second_column_values,third_row_values))
second_column_values = np.array([8,9,10,11])
User_3 = np.vstack((np.zeros(4),second_column_values,third_row_values))

# Defining a Temperature value
Temperature = 1
# Init for the sampler
reward_sampler = sampler(2)
# Defining number of iterations for epochs for training
ONE_EPOCH = 1000
# Defining number of Epochs
EPOCHS = 100
# A list for all contexts
all_contexts = [User_1,User_2,User_3]



"""Training the bandit now"""
all_contexts=run_epochs(epochs=EPOCHS,all_contexts=all_contexts,reward_sampler=reward_sampler,Temperature=Temperature,one_epoch_iterations=ONE_EPOCH)
User_1 = all_contexts[0]
User_2 = all_contexts[1]
User_3 = all_contexts[2]
print_bandit_status(context=all_contexts[0])
print_bandit_status(context=all_contexts[1])
print_bandit_status(context=all_contexts[2])