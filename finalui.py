import streamlit as st
import pandas as pd
import random
import numpy as np
from random import random as rnd
from numpy.random import randint
import base64  # For encoding background images
from IPython.display import IFrame
from database import add_user, check_user  # Import database functions

# Load the dataset
csv_file = "finaldata_1.csv"
df = pd.read_csv(csv_file)

# Function to set background image (only on login/signup page)
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    
    background_style = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
    
# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "page" not in st.session_state:
    st.session_state.page = "home"  
    
# Function to display Google Maps Route
def show_map(route):
    api_key = "AIzaSyAtYX5WZLCCZT7T-zh61LLZJwByaFphlak"  
    if len(route) > 2:
        waypoints = "|".join(route[1:-1])
    else:
        waypoints = ""
    
    url = f"https://www.google.com/maps/embed/v1/directions?key={api_key}&origin={route[0]}&destination={route[-1]}&waypoints={waypoints}"
    st.components.v1.iframe(url, width=800, height=600)

# Home Page
def home_page():
    set_background("images.png") 
    st.title("Welcome to the Home Page")
    # set_background("images.png") 
    if st.button("Wanna plan your day?"):
        st.session_state.page = "login"
        st.experimental_rerun()

# Login Page
def login_page():
    set_background("images.png")  
    st.subheader("Welcome! Please Login to Continue")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_user(username, password):
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.session_state.page = "main"
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password!")
    
    if st.button("Need an account? Sign up now"):
        st.session_state.page = "signup"
        st.experimental_rerun()

# Signup Page
def signup_page():
    set_background("images.png")
    st.subheader("Create a New Account")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up"):
        if new_password != confirm_password:
            st.warning("Passwords do not match!")
        elif add_user(new_username, new_password):
            st.success("Signup successful! You can now log in.")
            st.session_state.page = "login"
            st.experimental_rerun()
        else:
            st.warning("Username already taken! Try another.")
    
    if st.button("Already have an account? Login now"):
        st.session_state.page = "login"
        st.experimental_rerun()
        

        
        
#GA_DEFINE        
def fitness_prob(population, distance_matrixs):
  
    # Calculate total distance for each individual (chromosome)
    total_dist_all_individuals = [total_dist_individual(chromosome, distance_matrixs) for chromosome in population]
    
    # Compute multiplicative inverse of the distances
    fitness_probs = [1 / dist if dist != 0 else float('inf') for dist in total_dist_all_individuals]
    return fitness_probs

def total_dist_individual(chromosome, distance_matrixs):
    # Calculate the total distance for a single chromosome.
    total_distance = 0
    for i in range(len(chromosome) - 1):
        total_distance += distance_matrixs[chromosome[i]][chromosome[i + 1]]
    
    total_distance += distance_matrixs[chromosome[-1]][chromosome[0]]
    return total_distance
file_path = "/Users/nirja/finalp/distance_matrix.csv"  # Example path

def load_distance_matrixs(file_path):
    #  Load the distance matrix from a CSV file and return as a numpy array. 
    distance_matrixs = pd.read_csv(file_path, index_col=0)
    return distance_matrixs.values  # Return as numpy array

#selection
def Roulette_wheel_selection(fitness_score):
    Pc = [] #probability of selecting a chromosome
    sum_of_fitness = np.sum(fitness_score)
    for i in fitness_score:
        Pc.append(i/sum_of_fitness)
#     print(Pc)
    # print("Pc",Pc)
    return Pc
path_fitness_pair = {} #for global access
def dict_pair(population, fitness_score):
    for i in range(len(population)):
        path_fitness_pair[i] = fitness_score[i]
    return path_fitness_pair

def rank_selected(selected_population, population, path_fitness_pair):  # Pass population and path_fitness_pair as arguments
    key = []
    for i in selected_population:
        index = population.index(i)  # Get the index of the selected individual in the population
        key.append(index)
    
    # Sorting the fitness score of selected parents
    mylist = [path_fitness_pair[i] for i in key]  # Get fitness scores for selected parents
    mylist.sort()  # Sort in ascending order
    mylist = mylist[::-1]  # Reverse the list to get descending order
    
    # Get the path index of the sorted fitness scores
    path_index = [list(path_fitness_pair.values()).index(i) for i in mylist]
    
    return path_index


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

def selection(self, fitness_score):
        selected = []
       
        
        total_offspring = len( self.population) * self.crossover_rate
        num_parent_pairs = round(total_offspring / 2)
        num_selection = num_parent_pairs + 1
        
        for x in range(0, num_selection): 
            pointer = rnd()
            prob = 0
#             print(pointer)
            
            path_fitness_pair = dict_pair(self.population, fitness_score)
            for index, i in enumerate(Roulette_wheel_selection(fitness_score)):#roulette wheel
                prob += i 
                if prob > pointer:
#                     print(index)
                    selected.append(self.population[index])
#                     print('fitness_score',path_fitness_pair[index])
                    break
#             print(selected)
        return selected 

        
# GA class implementation
class GA:
    def __init__(self, population, crossover_rate, generation):
        self.population = population
       
        self.crossover_rate = crossover_rate

     
    def load_distance_matrixs(self, file_path):
        """
        Load the distance matrix from a file.
        """
        
        return np.load(file_path)


    def total_dist_individual(self, distance_matrixs, chromosome):
        total_distance = 0
        for i in range(len(chromosome) - 1):
            total_distance += distance_matrixs[chromosome[i]][chromosome[i + 1]]
        return total_distance

    def fitness_prob(self, population, distance_matrixs):  
        
        total_dist_all_individuals = [self.total_dist_individual(distance_matrixs, chromosome) for chromosome in population]
    
        # Compute multiplicative inverse of the distances
        fitness_probs = [1 / dist if dist != 0 else float('inf') for dist in total_dist_all_individuals]
        
        return fitness_probs

    def best_solution(self,fitness_score):
        best_fittest = np.max(fitness_score)
        index = fitness_score.index(best_fittest)
        best_individual = self.population[index]
        return best_fittest, best_individual

   

    def selection(self,fitness_prob):
        selected = []
        # calculating how many chromosomes to select for crossingover
        total_offspring = len(self.population) * self.crossover_rate
        num_parent_pairs = round(total_offspring / 2)
        num_selection = num_parent_pairs + 1
        
        for x in range(0, num_selection): 
            pointer = rnd()
            prob = 0
#             print(pointer)
            path_fitness_pair = dict_pair(self.population, fitness_prob)
            for index, i in enumerate(Roulette_wheel_selection(fitness_prob)):#roulette wheel
                prob += i 
                if prob > pointer:
#                     print(index)
                    selected.append(self.population[index])
#                     print('fitness_score',path_fitness_pair[index])
                    break
#             print(selected)
        return selected 

    def rank_selected(self,selected_population):#fitness score of selected population
        key = []
        for i in selected_population:
            index = self.population.index(i)
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

    def pairing(self,selected):
        parents = [[selected[x],selected[x+1]] for x in range(len(selected)-1)]
        return parents

    def two_point_crossover(self,parent1, parent2):
        pivot_point1 = randint(1, len(parent1)-2)
        # print('pivot', pivot_point1)
        pivot_point2 = randint(1, len(parent1)-1)
        while(pivot_point2 <= pivot_point1):
            pivot_point2 = randint(1, len(parent1))
        # print(pivot_point1, pivot_point2)
    #     print(parent1, parent2)
        if random.random() < 0.6: 

            offspring1 = [-1]*len(parent1)
            offspring2 = [-1]*len(parent1)
            offspring1[pivot_point1:pivot_point2] = parent2[pivot_point1:pivot_point2]
            offspring2[pivot_point1:pivot_point2] = parent1[pivot_point1:pivot_point2]

            # print(parent1, parent2)
            temp1 = parent1.copy()
            temp2 = parent2.copy() 

        #     print(offspring1)
            PMX(offspring1, offspring2, temp1, pivot_point1, pivot_point2)
            PMX(offspring2, offspring1, temp2, pivot_point1, pivot_point2)
            # print(f"offspring1:{offspring1},offspring2:{offspring2}")

            return [offspring1, offspring2]
        
        else:
            return [parent1, parent2]
    #     return offspring1, offspring2 =>list of tuples of list
    #     print(parent1, parent2)

    def individual_for_mutation(self, mutation_rate=0.1):
        """
        Select individuals for mutation based on mutation rate.
        """
        num_to_mutate = max(1, round(len(self.population) * mutation_rate))  
        individual_to_mutate = random.sample(self.population, num_to_mutate)  
        
        return individual_to_mutate



    def scramble_mutation(self,individual):
        p1 = randint(1, len(individual)-2)
        p2 = randint(1, len(individual)-1)
        while p1 >= p2:
            p2 = randint(1, len(individual))
        c2 = individual[p1:p2]
        random.shuffle(c2)
        for i in c2:
            individual[p1] = i
            p1 +=1
        return individual


def initialize_population(csv_file, n_population):
    df = pd.read_csv(csv_file)
    POI = {city: idx for idx, city in enumerate(df['locations'])}

    city_list = df['locations'].tolist()
    st.write("Available Cities:")
    for idx, city in enumerate(city_list, start=1):
        st.write(f"{idx}. {city}")

    start_idx = st.selectbox("Choose your starting point", range(len(city_list)), format_func=lambda x: city_list[x])
    start_city = city_list[start_idx]

    
    visit_cities = st.multiselect("Choose places to visit", city_list, max_selections=11)
    
    if len(visit_cities) > 11:
        st.warning("Too many places to visit in one day! Please select a maximum of 10 places.")
        return []  

    if start_city not in visit_cities:
        visit_cities.insert(0, start_city)

    population = []
    for _ in range(n_population):
        random_route = visit_cities[1:]
        random.shuffle(random_route)
        random_route = [start_city] + random_route + [start_city]
        population.append([POI[city] for city in random_route])

    return population


# Run the genetic algorithm
def run(population):
    g = GA(population, crossover_rate=0.6,generation=300)

    
    # Load the distance matrix
    distance_matrixs_path = "C:\\Users\\nirja\\finalp\\formatted_distance_matrix.npy"
    distance_matrixs = np.load(distance_matrixs_path)
    

    # Compute fitness scores for the entire population
    fitness_scores = g.fitness_prob(population, distance_matrixs)


    # Find the best individual and fitness score
    best_fittest, best_individual = g.best_solution(fitness_scores)

    # Perform elitism: Retain the best individual(s) without modification
    elite_population = [best_individual]

    # Selection process using sorted fitness scores
    selected = g.selection(fitness_scores)

    # Pair selected individuals for crossover
    paired_parents = g.pairing(selected)
    
    # Perform crossover to generate offspring
    offsprings = []
    for pair in paired_parents:
        offsprings.extend(g.two_point_crossover(pair[0], pair[1]))

    # Prepare the next generation population
    next_population = list(offsprings)

   
    individuals_to_mutate = g.individual_for_mutation()
    for individual in individuals_to_mutate:
        mutated_individual = g.scramble_mutation(individual)
        next_population.append(mutated_individual)

    # Sort population based on fitness scores
    population_with_fitness = list(zip(population, fitness_scores))
    sorted_population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

    # Select additional elite individuals beyond the best one
    additional_elite_count = min(10, len(sorted_population_with_fitness))  
    additional_elite = [ind for ind, _ in sorted_population_with_fitness[:additional_elite_count]]

    # Add additional elite individuals to the elite population
    elite_population.extend(additional_elite)

    # Add elite population to the next generation
    next_population.extend(elite_population)

    # Ensure population size remains constant
    while len(next_population) < len(population):
        next_population.append(random.choice(population))

    return best_fittest, best_individual, next_population





# Main App UI
# Main App UI
def main_app():
    st.title(f"Welcome, {st.session_state.current_user}! 🚀")
    st.subheader("Genetic Algorithm-based Travel Optimizer")
    
    population = initialize_population(csv_file, n_population=200)
    POI = {city: idx for idx, city in enumerate(df['locations'])}
    n_population = 200
    generation = 300
    fitness_scores = []  # Store fitness scores

    global_best_fittest = float('inf')  # Assuming lower fitness score is better
    global_best_individual = None
    
    if st.button("Find Best Route"):
        for i in range(generation):
            best_fittest, best_individual, next_population = run(population)
            fitness_scores.append(best_fittest)  # Store fitness score

            if best_fittest < global_best_fittest:
                global_best_fittest = best_fittest
                global_best_individual = best_individual.copy()
            # f"Generation {i + 1}: Best Fittest: {best_fittest}, Best Individual: {best_individual}"
            while len(next_population) < n_population:
                next_population.append(population[randint(0, len(population) - 1)])
            population = next_population
            # POI = {city: idx for idx, city in enumerate(df['locations'])}
        route = [city for idx in best_individual for city, index in POI.items() if index == idx]
        
        st.subheader("Best Route:")
        st.write(" -> ".join(route))
        st.subheader("Shortest Distance (Fitness Score):")
        st.write(f"{1/best_fittest} km")
        
        
        # Display Google Map with route
        st.subheader("Route Map:")
        show_map(route)
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.page = "home"
        st.experimental_rerun()

# Page Router
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.logged_in:
    main_app()