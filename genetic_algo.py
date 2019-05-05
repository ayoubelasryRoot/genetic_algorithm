#!/usr/bin/python3
import numpy as np
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


def is_valid_model(lego_information, index_model, current_lego):
    for i in range(0, lego_information.nb_lego):
        if current_lego[i] > 0 and lego_information.lego_models[index_model][i] > 0:
            return True
    return False


def guess_greedy_model(lego_information, current_lego):
    cost_best_model = 0
    best_cost = sys.maxsize * -1
    for i in range(0, lego_information.nb_models):
        if is_valid_model(lego_information, i, current_lego):
            cost = np.dot(np.subtract(current_lego, lego_information.lego_models[i]), lego_information.lego_price)
            if cost > best_cost:
                best_cost = cost
                cost_best_model = i
    return cost_best_model


def random_valid_index_model(lego_information, current_lego):
    valid_index = []
    for i in range(0, lego_information.nb_lego):
        if current_lego[i] > 0:
            for j in range(0, lego_information.nb_models):
                if lego_information.lego_models[j][i] > 0:
                    valid_index.append(j)
    return valid_index[np.random.randint(0, len(valid_index))]


def genetic_algorithm(lego_information=LegoInformation):

    best_solution_models = np.zeros(lego_information.nb_models)
    best_solution_price = -1 * sys.maxsize
    nb_species_by_iteration = 75
    # crossover
    parent_a = -1
    parent_b = -1
    previous_parent_a_cost = 0
    mutation_probability = 0

    population_models = np.zeros((nb_species_by_iteration, lego_information.nb_models))
    population_models_cost = np.zeros(nb_species_by_iteration)

    while True:
        for j in range(0, nb_species_by_iteration):
            models_used_by_generation = np.zeros(lego_information.nb_models).astype("int32")
            if parent_a != -1:
                species_selection(lego_information, population_models, parent_a, parent_b, models_used_by_generation, mutation_probability)

            updated_legos = np.copy(lego_information.initial_lego)
            updated_current_lego(lego_information, models_used_by_generation, updated_legos)
            while not current_lego_done(updated_legos):
                model_index = 0
                if np.min(updated_legos) < 0 and np.random.random() < 0.001:
                    model_index = guess_greedy_model(lego_information, updated_legos)
                else:
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
            mutation_probability += 0.01 if mutation_probability < 1 else 0
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
