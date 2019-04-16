#!/usr/bin/python3
import numpy as np
import multiprocessing as mp
from load_data import load_data_form_file
import sys
import time
import random

class LegoInformation:
    initial_lego = None
    lego_price = None
    lego_models = None
    nb_models = None
    nb_lego = None

    def __init__(self, initial_lego, lego_price, lego_models):
        self.initial_lego = initial_lego
        self.lego_price = lego_price
        self.lego_models = lego_models
        self.nb_models = len(lego_models)
        self.nb_lego = len(initial_lego)


def is_valid_model(lego_information=LegoInformation, index_model=0, current_lego=[]):
    for i in range(0, lego_information.nb_lego):
        if current_lego[i] > 0 and lego_information.lego_models[index_model][i] > 0:
            return True
    return False


def reduce_size_objects(lego_information=LegoInformation):
    # models used for basic solution
    models_used = np.zeros(len(models)).astype("int32")
    initial_sum = np.sum(lego_information.initial_lego)
    # reduce the maximum
    index = check_best_solution_without_tracking(lego_information.initial_lego)
    current_legos = lego_information.initial_lego - models[index]
    models_used[index] += 1
    while np.min(current_legos) > 0:
        index = check_best_solution_without_tracking(current_legos)
        current_legos = current_legos - models[index]
        models_used[index] += 1
    return current_legos, models_used


def check_best_solution_without_tracking(objects):
    best_total_cost_priority = sys.maxsize
    best_model_index = -1

    for index in range(0, len(models)):
        test = objects - models[index]
        value = np.amax(test) - np.amin(test)
        if value < best_total_cost_priority and change_was_made(objects, test):
            best_total_cost_priority = value
            best_model_index = index
    return best_model_index


def random_valid_index_model(lego_information=LegoInformation, current_lego=[]):
    valid_index = []
    for i in range(0, lego_information.nb_lego):
        if current_lego[i] > 0:
            for j in range(0, lego_information.nb_models):
                if lego_information.lego_models[j][i] > 0:
                    valid_index.append(j)
    return valid_index[np.random.randint(0, len(valid_index))]


# todo remove duplicate code
def genetic_algorithm(lego_information=LegoInformation):
    # solution_greedy =mp.Array('i', lego_information.nb_models)
    # start_solution=mp.Array('i', lego_information.nb_models)
    # # Start greedy algorithm 
    # greedy_process = mp.Process(target=greedy_algorithm, args=(start_solution, lego_information, solution_greedy))
    # to compare models

    best_solution_models = np.zeros(lego_information.nb_models).astype("int32")
    best_solution_price = -1 * sys.maxsize
    nb_species_by_iteration = 75
    # crossover
    parent_a = -1
    parent_b = -1
    previous_parent_a_cost = 0
    mutation_probability = 0

    population_models = np.zeros((nb_species_by_iteration, lego_information.nb_models)).astype("int32")
    population_models_cost = np.zeros(nb_species_by_iteration).astype("int32")
    current_legos, models_used_by_generation = reduce_size_objects(lego_information)

    # best_solution_models, current_lego = greedy_algorithm(np.copy(lego_information.initial_lego), lego_information, best_solution_models)
    while True:
        for j in range(0, nb_species_by_iteration):
            if parent_a != -1:
                species_selection(lego_information, population_models, parent_a, parent_b, models_used_by_generation, mutation_probability)

            updated_legos = np.copy(lego_information.initial_lego)
            updated_current_lego(lego_information, models_used_by_generation, updated_legos)
            while not current_lego_done(updated_legos):
                model_index = random_valid_index_model(lego_information=lego_information, current_lego=updated_legos)
                updated_legos = np.subtract(updated_legos, lego_information.lego_models[model_index])
                models_used_by_generation[model_index] += 1

            cost_generation = np.dot(updated_legos, lego_information.lego_price)
            if cost_generation > best_solution_price:
                best_solution_price = cost_generation
                best_solution_models = np.copy(models_used_by_generation)
                print(' '.join(map(str, best_solution_models)), " : {}".format(best_solution_price))
                # print("{} : {}".format(repr(best_solution_models), best_solution_price))

            population_models[j] = models_used_by_generation
            population_models_cost[j] = cost_generation

        parent_a = np.argmax(population_models_cost)
        if population_models_cost[parent_a] <= previous_parent_a_cost:
            mutation_probability += 0.2 if mutation_probability < 1 else 0
        else:
            mutation_probability = 0.01
        previous_parent_a_cost = population_models_cost[parent_a]
        population_models_cost[parent_a] = sys.maxsize * -1
        parent_b = np.argmax(population_models_cost)


def species_selection(lego_information, population_models, parent_a, parent_b, models_used_by_generation, mutation_probability):
    for h in range(0, lego_information.nb_models):
        value_to_set = population_models[parent_a][h]
        if population_models[parent_b][h] < population_models[parent_a][h]:
            value_to_set = population_models[parent_b][h]
        if value_to_set > 0:
            models_used_by_generation[h] = value_to_set

    # mutation
    if np.random.random() < mutation_probability:
        random_index_for_mutation = random.choice([
            index for index in range(0, lego_info.nb_models) if models_used_by_generation[index] > 0
        ])
        models_used_by_generation[random_index_for_mutation] -= 1


def updated_current_lego(lego_information, models_used_by_generation, updated_legos):
    for model_index in range(0, lego_information.nb_models):
        if models_used_by_generation[model_index] > 0:
            updated_legos -= lego_information.lego_models[model_index] * models_used_by_generation[model_index]


def not_negative_value(current_lego):
    for i in range(0, len(current_lego)):
        if current_lego[i] < 0:
            return False
    return


def change_was_made(current_legos, new_current_legos):
    positive_equals = True
    for i in range(0, len(current_legos)):
        if current_legos[i] > 0 and current_legos[i] != new_current_legos[i]:
            positive_equals = False
    return not positive_equals


def current_lego_done(current_legos):
    for i in range(0, len(current_legos)):
        if current_legos[i] > 0:
            return False
    return True


if __name__ == "__main__":
    start = time.time()
    file_name = sys.argv[1]
    lego, price, models = load_data_form_file(file_name)
    lego_info = LegoInformation(lego, price, models)
    genetic_algorithm(lego_information=lego_info)
