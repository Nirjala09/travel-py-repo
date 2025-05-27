#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from numpy.random import randint
from random import random as rnd


# In[2]:


import datetime
from datetime import datetime
from datetime import time, timedelta


# In[3]:


# current_location = 'dhapakhel'


# In[4]:


# POIs = [current_location, "Pashupatinath Temple", "Boudhanath (Stupa)", "Swayambhunath Stupa", "Tribhuvan Museum","Dakshinkali Temple", "Narayanhiti Palace", "Garden of Dreams", "Kasthamandap", "Patan Durbar Square"]


# In[5]:


# POIs


# In[6]:


def encode(POIs):
    encoded_POI = []
    for index, POI in enumerate(POIs):
        encoded_POI.append(index)
    return encoded_POI


# In[7]:


# encoded_POI = encode(POIs)


# In[8]:


# encoded_POI


# In[9]:


class Individual:
    #Individual generation
    def generate_individual(self, POIs):
        constant_gene = POIs[0] #in every chromosomes starting location is the current location
        return [constant_gene]+random.sample(POIs[1:], len(POIs)-1)
    
    def population(self, population_size, POIs):
        return [self.generate_individual(POIs) for x in range(population_size)]
            


# In[10]:


def Roulette_wheel_selection(fitness_score):
    Pc = [] #probability of selecting a chromosome
    sum_of_fitness = np.sum(fitness_score)
    for i in fitness_score:
        Pc.append(i/sum_of_fitness)
#     print(Pc)
    return Pc


# In[11]:


path_fitness_pair = {} #for global access
def dict_pair(population, fitness_score):
    for i in range(len(population)):
        path_fitness_pair[i] = fitness_score[i]
    return path_fitness_pair


# In[12]:


# print(path_fitness_pair)


# In[13]:

#for pairing highest ranked individuals
def rank_selected(selected_population, population):#fitness score of selected population
    key = []
    for i in selected_population:
        index = population.index(i)
        key.append(index)
    #fittest pairing => pairing the selected parents
    #sorting dict by value
    mylist = [path_fitness_pair[i] for i in key] #fitness score of selected parents
    # print(mylist)
    mylist.sort()#ascending order
    # print(mylist)
    mylist = mylist[::-1] # reverse ordering list , descending order
    # print(mylist)
    path_index = [list(path_fitness_pair.values()).index(i) for i in mylist]
    return path_index


# In[14]:


# from time_period import get_time
from get_time_matrix import get_time_matrix


# In[52]:

def get_info(POIs):
#    global opening_time, closing_time
    global distance_time
#     opening_time, closing_time = get_time(POIs[1:],'Monday')
    distance_time = get_time_matrix(POIs)
#     print('opening_time',opening_time)
    


# In[53]:


# print(opening_time, closing_time)


# In[17]:


stay_time = 3600 # 1hr stay in each poi


# # In[19]:


# from get_time_matrix import get_time_matrix


# # In[20]:


# distance_time = get_time_matrix(POIs)


# # In[21]:


# distance_time


# In[71]:


# def waiting_cost(starting_time,i, d ):
#     if d == np.inf:
#         d = 1000000;
# #     if type(opening_time[i]) == 'str':
# #         opening_time[i] = time(23, 59)
#     arrival_time = (datetime.combine(datetime.today(),(starting_time))+ timedelta(seconds=d)).time()
# #     print(opening_time[i])
# #     print(i)
# #     print(f'opening_time[{i}]',opening_time[i])
# #     opening = datetime.combine(datetime.today(),(opening_time[i]))
# #     waiting_time = opening-(datetime.combine(datetime.today(),(starting_time))+ timedelta(seconds=d))
#     next_time = (datetime.combine(datetime.today(),starting_time)+ timedelta(seconds=d))+timedelta(seconds = stay_time)
# #     print(stay_time[i])
# #     print(next_time.time())
#     return next_time.time(), arrival_time


# # In[64]:


# def delay_cost(leaving_time,i ):
#     closing = datetime.combine(datetime.today(),(closing_time[i]))
#     delay_time = datetime.combine(datetime.today(),(leaving_time))-closing
#     return max(delay_time.total_seconds(), 0)


# In[81]:


#Partially Mapped Crossover
def PMX(offspring1, offspring2, temp1, pivot_point1, pivot_point2):
    for i in range(len(offspring1)):
        if i < pivot_point1 or i >= pivot_point2: #change the points here
#             print('index',i)
            if temp1[i] not in offspring1:
                offspring1[i] = temp1[i]
#                 print(offspring1[i])
            else:
#                 print(temp1[i] in offspring1[2:4])
                while temp1[i] in offspring1[pivot_point1:pivot_point2]:
                    index = offspring1.index(temp1[i])
                    temp1[i] = offspring2[index]
                offspring1[i] = temp1[i]
#                 print(offspring1[i])


# In[133]:


class GA:
    def __init__(self, population, crossover_rate):
        self.population = population
        self.crossover_rate = crossover_rate
        
    def fitness_evaluation(self, path):
        count = 0
        fitness_value = 0
        current_time = time(10,0)
#         print(path)
        for i in path:
#             print(i)
#             print('path',i)
            if count >= 1:
#                 print(i-1)
#                 cost, current_time, arrival_time = waiting_cost(current_time, i-1, distance_time[j][i]) #i-1 because opening and closing time of current_location is not mentioned
#                 delay = delay_cost(current_time, i-1)
                fitness_value += distance_time[j][i]# + cost + delay# distance consideration
    #             print(fitness_value)
#             print('fitness value', fitness_value/3600)
            j = i
            count += 1
            fitness_score = fitness_value/3600
        return 1/fitness_score
    
    #Choose a random number for making it act like a pointer in the wheel
    def selection(self, fitness_score):
        selected = []
       
        # calculating how many chromosomes to select for crossingover
        total_offspring = len(self.population) * self.crossover_rate
        num_parent_pairs = round(total_offspring / 2)
        num_selection = num_parent_pairs + 1
        
        for x in range(0, num_selection): 
            pointer = rnd()
            prob = 0
#             print(pointer)
            path_fitness_pair = dict_pair(self.population, fitness_score)
            for index, i in enumerate(Roulette_wheel_selection(fitness_score)):#roulette wheel
                prob += i #can take out cumulative sum instead see this............
                if prob > pointer:
#                     print(index)
                    selected.append(self.population[index])
#                     print('fitness_score',path_fitness_pair[index])
                    break
#             print(selected)
        return selected  
    
    #fittest pairing
    def pairing(self, selected_population):
        parents = [[selected_population[x],selected_population[x+1]] for x in range(len(selected_population)-1)]
        return parents
    
    #crossover
    #two point crossover
    def two_point_crossover(self, parent1, parent2):
        pivot_point1 = randint(1, len(parent1)-1)
#         print('pivot', pivot_point1)
        pivot_point2 = randint(1, len(parent1))
        while(pivot_point2 <= pivot_point1):
            pivot_point2 = randint(1, len(parent1))
    #     print(pivot_point1, pivot_point2)
    #     print(parent1, parent2)
        if random.random() < self.crossover_rate: 

            offspring1 = [-1]*len(parent1)
            offspring2 = [-1]*len(parent1)
            offspring1[pivot_point1:pivot_point2] = parent2[pivot_point1:pivot_point2]
            offspring2[pivot_point1:pivot_point2] = parent1[pivot_point1:pivot_point2]

        #     print(parent1, parent2)
            temp1 = parent1.copy()
            temp2 = parent2.copy() #copy garena bhane parent ma ni modification aauxa but why???

        #     print(offspring1)
            PMX(offspring1, offspring2, temp1, pivot_point1, pivot_point2)
            PMX(offspring2, offspring1, temp2, pivot_point1, pivot_point2)

            return [offspring1, offspring2]
        
        else:
            return [parent1, parent2]
    #     return offspring1, offspring2 =>list of tuples of list
    #     print(parent1, parent2)
    
    #Scramble mutation : choose two points then bichko genes lai shuffle garne
    def individual_for_mutation(self, mutation_rate = 0.1):
        individual_to_mutate = []
        for x in range(round(len(self.population)*mutation_rate)):
            individual_to_mutate.append(self.population[randint(0, len(self.population)-1)])
        return individual_to_mutate
    
    def scramble_mutation(self, individual):
        p1 = randint(1, len(individual)-1)
        p2 = randint(1, len(individual))
        while p1 >= p2:
            p2 = randint(1, len(individual))
        c2 = individual[p1:p2]
        random.shuffle(c2)
        for i in c2:
            individual[p1] = i
            p1 +=1
        return individual


# In[57]:


def best_solution(population, fitness_score):
    best_fittest = np.max(fitness_score)
    index = fitness_score.index(best_fittest)
    best_individual = population[index]
    return best_fittest, best_individual


# In[112]:
def population_initialization(POIs):
    global population_size
    encoded_POI = encode(POIs)
    get_info(POIs)
    i = Individual()
    population_size = 100
    initial_population = i.population(population_size, encoded_POI)
    population = initial_population
#     print(population)
    return population


# In[137]:


def run(population):
    g = GA(population, 0.8)
    fitness_score = [g.fitness_evaluation(individual) for individual in population]
    best_fittest, best_individual = best_solution(population, fitness_score)
    
    # Perform elitism: Retain the best individual(s) without modification
    elite_population = [best_individual]
    
    selected = g.selection(fitness_score)
#     print(selected)
    path_index = rank_selected(selected, population)
#     print(path_index)
    paired_parents = g.pairing(selected)
#     print(paired_parents)
    offsprings = []
    for i in paired_parents:
        offsprings.append(g.two_point_crossover(i[0], i[1]))
#     print(offsprings, len(offsprings))
    next_population = []
    for i in offsprings:
        next_population.extend(i)
#     print('nextpopulation',next_population)
    individual_to_mutate = g.individual_for_mutation()
#     print('individual_to_mutate',individual_to_mutate)
    for x in individual_to_mutate:
        next_population.append(g.scramble_mutation(x))
        
    # Sort the remaining individuals based on their fitness scores
    remaining_population = [(ind, score) for ind, score in zip(population, fitness_score) if ind not in elite_population]
    remaining_population.sort(key=lambda x: x[1], reverse=True)

    # Select additional elite individuals beyond the best one
    additional_elite_count = min(10, len(remaining_population))  # You can adjust the number of additional elites as needed
    additional_elite = [ind for ind, _ in remaining_population[:additional_elite_count]]

    # Add additional elite individuals to the elite population
    elite_population.extend(additional_elite)

    # Add elite population to the next generation
    next_population.extend(elite_population)

#     print('nextpopulation',next_population)
#     initial_population = next_population
    return best_fittest, best_individual, next_population


# In[138]:

def run_GA(population, POIs):
    generation = 100
    for i in range(generation):
    #     print(population)
        best_fittest, best_individual, next_population = run(population)
        print('Generation',i,':', 'Best Fittest:',best_fittest," Best Individual:",best_individual)
        while len(next_population) < population_size:
    #         print(len(next_population))
    #         print(population[randint(0, len(population))])
            next_population.append(population[randint(0, len(population))])
#         print(len(next_population))
        population = next_population

    route = []
    for i in best_individual:
        route.append(POIs[i])
    return route


# # In[79]:

from datetime import datetime, timedelta

def waiting_cost(starting_time,i, d ):
    arrival_time = (starting_time+ timedelta(seconds=d))
# #     print(opening_time[i])
#     opening = datetime.combine(datetime.today(),(opening_time[i]))
#     waiting_time = opening-(datetime.combine(datetime.today(),(starting_time))+ timedelta(seconds=d))
    next_time = ((starting_time+ timedelta(seconds=d))+timedelta(seconds = stay_time))
#     print(stay_time[i
#     print(next_time.time())
    return next_time, arrival_time



import pandas as pd

df = pd.read_csv('locationsandimage.csv')

def get_image(place):
    index = df.index[df['locations']==place].tolist()
    if len(index)==0:
        return ''
    return df['image'][index[0]]
   

from datetime import datetime, timedelta

def waiting_cost(starting_time,i, d ):
    if d == np.inf:
        d = 3000;
    arrival_time = (starting_time+ timedelta(seconds=d))
# #     print(opening_time[i])
#     opening = datetime.combine(datetime.today(),(opening_time[i]))
#     waiting_time = opening-(datetime.combine(datetime.today(),(starting_time))+ timedelta(seconds=d))
    next_time = ((starting_time+ timedelta(seconds=d))+timedelta(seconds = stay_time))
#     print(stay_time[i
#     print(next_time.time())
    return next_time, arrival_time



import pandas as pd

df = pd.read_csv('locationsandimage.csv')

def get_image(place):
    index = df.index[df['locations']==place].tolist()
    if len(index)==0:
        return ''
    return df['image'][index[0]]
   

def timing(POIs, route, start_time=datetime.strptime("10:00:00", "%H:%M:%S"), end_time=datetime.strptime("18:00:00", "%H:%M:%S")):
    # Estimate time needed for each POI (you need to define this)
    poi_times = {poi: 1 for poi in route}  # Example: Assuming each POI takes 1 hour

    # Travel times between POIs (you need to define this)
    travel_times = {}  # Example: travel_times[('POI1', 'POI2')] = 0.5 (0.5 hours or 30 minutes)

    # Initialize variables
    current_time = start_time
    arrival_time = start_time
    leave_time = start_time
#     leave_time = ''
    total_time = timedelta()
    current_day = 1
    daily_itineraries = []
    count = 0

    # Iterate through the route
    for place in route:
        
        i = POIs.index(place)

        # Check if adding this POI exceeds the end time of the day
        if current_time + timedelta(hours=poi_times[place]) > end_time:
            # Move to the next day
            current_day += 1
            current_time = start_time
            total_time = timedelta()
            
        if count == 0:
            if len(daily_itineraries) < current_day:
                print(place)
                daily_itineraries.append({'day': current_day, 'POIs': []})
            daily_itineraries[current_day - 1]['POIs'].append({'location': place, 'image_url':get_image(place),                         'arrival_time':str(arrival_time.time()), 'leaving_time':str(leave_time.time())})
                 
        else:
                    
            current_time, arrival_time = waiting_cost(current_time, i-1, distance_time[j][i])
#             print(arrival_time)
            leave_time = current_time
#             if len(daily_itineraries) == current_day:
#                 print(place)
# #                 daily_itineraries.append({'day': current_day, 'POIs': []})
#             if len(daily_itineraries) == current_day:
#                 daily_itineraries[current_day - 1]['POIs'].append({
#                     'location': place, 
#                     'image_url': get_image(place),
#                     'arrival_time': str(arrival_time.time()), 
#                     'leaving_time': str(leave_time.time())
#                 })
                
            if len(daily_itineraries) < current_day:
#                 daily_itineraries[current_day - 1]['POIs'].append({'location': POIs[0], 'image_url':'',                         'arrival_time':str(start_time.time()), 'leaving_time':str(start_time.time())})
# #                 current_time, arrival_time = waiting_cost(start_time, i-1, distance_time[0][i])
# #             print(arrival_time)
#                 leave_time = current_time
                daily_itineraries.append({'day': current_day, 'POIs': []})
                daily_itineraries[current_day - 1]['POIs'].append({'location': POIs[j], 'image_url':'',                         'arrival_time':str(start_time.time()), 'leaving_time':str(start_time.time())})
            daily_itineraries[current_day - 1]['POIs'].append({'location': place, 'image_url':get_image(place),                         'arrival_time':str(arrival_time.time()), 'leaving_time':str(leave_time.time())})
            
        j = i
        count += 1
        
#     daily_itineraries[1]['POIs'].append({'location': POIs[0], 'image_url':'',                         'arrival_time':str(start_time.time()), 'leaving_time':str(start_time.time())})        
    return daily_itineraries
