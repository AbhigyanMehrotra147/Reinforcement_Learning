import numpy as np
import random
from my_package.sampler import sampler
import sys
import random
import copy
from termcolor import colored



"""Given a context function finds the action with max reward"""
def get_best_action(context: np.ndarray) -> int:
    return np.argmax(context[0])

"""Function implements the UCB formula"""
def ucb_value( exploration: int, \
                context: np.ndarray, action: int, time_steps: int)->float:
    return context[0][action] + exploration*np.sqrt(np.divide(np.log(time_steps),context[2][action]))

"""Given a context finds how many times it has been chanced upon"""
def get_context_time(context: np.ndarray)->int:
    return np.sum(context[2])

"""Given a context function chooses an action to take"""
def choose_action(context: np.ndarray, time_steps: int, exploration: float) -> int:
    max_ = 0
    arg_max = 0
    for i in range(context.shape[1]):
        temp = ucb_value(exploration=exploration,context=context,action=i,time_steps=time_steps)
        if(temp > max_):
            max_ = temp
            arg_max = i
    return arg_max
    
"""Given an action in an action the function recieves reward and updates average reward and 
    increments the counter for the action taken"""
def update_estimated_avg_reward(context: np.ndarray, action: int,
                                 reward_sampler: sampler) -> None: 
    reward = reward_sampler.sample(int(context[1][action]))
    context[0][action] = ((context[0][action]*context[2][action]) + reward)/(context[2][action]+1)
    context[2][action] += 1

"""Learning for agent for one epoch"""
def run_UCB(contexts: list, iterations: int,\
             reward_sampler: sampler, exploration: float)->np.ndarray:
    for i in range(iterations):
        context = random.choice(contexts)
        action = choose_action(context=context,time_steps=get_context_time(context=context),exploration=exploration)
        update_estimated_avg_reward(context=context,action=action,reward_sampler=reward_sampler)
    return contexts


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
 
"""Runs over many epochs and averages output"""
def run_epochs(epochs: int, all_contexts: list,\
                reward_sampler: sampler, epsilon: float,\
                      one_epoch_iterations: int, exploration: float)->list:
    Users = copy.deepcopy(all_contexts)
    temp = copy.deepcopy(all_contexts)

    for i in range(epochs):
        run_UCB(contexts=Users,iterations=one_epoch_iterations,\
                reward_sampler=reward_sampler,exploration=exploration)
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
The 1st row contains the corresponding number to sample reward
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

# Defining a constant EPSILON value
EPSILON = 0.1
# Init for the sampler
reward_sampler = sampler(2)
# Defining number of iterations for epochs for training
ONE_EPOCH = 1000
# Defining number of Epochs
EPOCHS = 1
# A list for all contexts
all_contexts = [User_1,User_2,User_3]
# Temperature
c = 2


#run_greedy(contexts=all_context,iterations=ONE_EPOCH,epsilon=EPSILON,reward_sampler=reward_sampler)

all_contexts=run_epochs(epochs=EPOCHS,all_contexts=all_contexts\
                        ,reward_sampler=reward_sampler,epsilon=EPSILON,\
                            one_epoch_iterations=ONE_EPOCH, exploration=c)
User_1 = all_contexts[0]
User_2 = all_contexts[1]
User_3 = all_contexts[2]
print_bandit_status(context=all_contexts[0])
print_bandit_status(context=all_contexts[1])
print_bandit_status(context=all_contexts[2])

