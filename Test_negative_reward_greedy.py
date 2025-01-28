import numpy as np
import matplotlib.pyplot as plt  
import random
import math


""" Two arms greedy problem """

Arm1 = {
    "reward_prob" : 0.3,
    "punishment_prob" : 0.7,
    "numb_chosen" : 0,
    "value" : 0
}

Arm2 = {
    "reward_prob" : 0.6,
    "punishment_prob" : 0.4,
    "numb_chosen" : 0,
    "value" : 0
}

# Defining rewards
reward = 1
punishment = -1

best_reward = Arm2["reward_prob"]*1 + Arm2["punishment_prob"]*(-1)
print(best_reward)
greedy_0_avg_reward_list = []
greedy_0_total_reward = 0
greedy_0_total_reward_list = []
greedy_point1_avg_reward_list = []
greedy_point1_total_reward = 0
greedy_point1_total_reward_list = []

def update_value(numb_chosen, new_reward, value):
    new_value = (((numb_chosen-1)*value) + new_reward)/(numb_chosen)
    return new_value


def choose_action(epsilon):
    random_number = random.uniform(0,1)
    if(random_number < epsilon):
        return random.choice([1,2])
    else:
        temp = np.array([Arm1["value"], Arm2["value"]])
        if(temp[0] == temp[1]):
            return random.choice([1,2])
        return np.argmax(temp) + 1

def get_reward(reward_prob):
    if random.uniform(0,1) < reward_prob:
        return reward
    else:
        return punishment


numb_iterations = 4000

i = 0
chosen_action = 0
new_reward = 0
while(i < numb_iterations):
    # for greedy
    i+=1
    chosen_action = choose_action(0)
    if(chosen_action == 1):
        new_reward = get_reward(Arm1["reward_prob"])
        Arm1["numb_chosen"] += 1
        Arm1["value"] = update_value(Arm1["numb_chosen"],new_reward,Arm1["value"])
    elif(chosen_action == 2):
        new_reward = get_reward(Arm2["reward_prob"])
        Arm2["numb_chosen"] += 1
        Arm2["value"] = update_value(Arm2["numb_chosen"],new_reward,Arm2["value"])
    else:
        pass
    greedy_0_total_reward = (greedy_0_total_reward + new_reward)
    greedy_0_total_reward_list.append(greedy_0_total_reward)
    greedy_0_avg_reward_list.append(greedy_0_total_reward/i)




Arm1 = {
    "reward_prob" : 0.3,
    "punishment_prob" : 0.7,
    "numb_chosen" : 0,
    "value" : 0
}

Arm2 = {
    "reward_prob" : 0.6,
    "punishment_prob" : 0.4,
    "numb_chosen" : 0,
    "value" : 0
}




i = 0
chosen_action = 0
new_reward = 0
while(i < numb_iterations):
    # for greedy
    i+=1
    chosen_action = choose_action(0.1)
    if(chosen_action == 1):
        new_reward = get_reward(Arm1["reward_prob"])
        Arm1["numb_chosen"] += 1
        Arm1["value"] = update_value(Arm1["numb_chosen"],new_reward,Arm1["value"])
    elif(chosen_action == 2):
        new_reward = get_reward(Arm2["reward_prob"])
        Arm2["numb_chosen"] += 1
        Arm2["value"] = update_value(Arm2["numb_chosen"],new_reward,Arm2["value"])
    else:
        pass
    greedy_point1_total_reward = (greedy_point1_total_reward + new_reward)
    greedy_point1_total_reward_list.append(greedy_point1_total_reward)
    greedy_point1_avg_reward_list.append(greedy_point1_total_reward/i)

print("Greedy_0 total reward: ",greedy_0_total_reward)
print("Greedy_point1_total_reward: ",greedy_point1_total_reward)
best_reward_list = [best_reward]*numb_iterations



plt.plot(greedy_0_total_reward_list,color="blue",label="greedy total reward")
plt.plot(greedy_point1_total_reward_list,color="red",label="epsilon=0.1 total reward")
plt.xlabel("timesteps")
plt.ylabel("total_rewards")
plt.title("greedy vs epsilon_greedy")
plt.legend()
plt.show()

plt.plot(best_reward_list,color="green",label="best_reward_average")
plt.plot(greedy_0_avg_reward_list,color="blue",label="greedy average reward")
plt.plot(greedy_point1_avg_reward_list,color="red",label="epsilon=0.1 average reward")
plt.xlabel("timesteps")
plt.ylabel("average_rewards")
plt.title("greedy vs epsilon_greedy")
plt.legend()
plt.show()
