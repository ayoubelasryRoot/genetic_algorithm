import numpy as np
from load_data import load_data_form_file
import sys
np.set_printoptions(linewidth=np.inf)


class LegoInformation:
    def __init__(self, initial_lego, lego_price, lego_models):
        self.initial_lego = initial_lego
        self.lego_price = lego_price
        self.lego_models = lego_models
        self.nb_models = len(lego_models)
        self.nb_lego = len(initial_lego)


def greedy_algo(current_lego, lego_information):
    cost_models = np.zeros(lego_information.nb_models)
    for i in range(0, lego_information.nb_models):
        model_to_try = lego_information.lego_models[i]
        updated_current_lego = np.subtract(current_lego, model_to_try)
        if change_was_made(current_lego, updated_current_lego):
            if np.min(updated_current_lego) >= 0:
                cost_models[i] = np.sum(np.subtract(current_lego, updated_current_lego))
            else:
                cost_models[i] = np.dot(updated_current_lego, lego_information.lego_price)
        else:
            cost_models[i] = sys.maxsize * -1
    return np.argmax(cost_models)


def genetic_algo(lego_information=LegoInformation):
    best_solution_models = None
    best_solution_price = -1 * sys.maxsize
    POPULATION = lego_information.nb_lego * 10

    population_models = np.zeros((POPULATION, lego_information.nb_lego))
    population_models_cost = np.zeros(POPULATION)

    print("random population")
    for i in range(0, POPULATION):
        models_used_by_generation = np.zeros(lego_information.nb_models).astype("int32")
        updated_legos = np.copy(lego_information.initial_lego)
        while not current_legos_done(updated_legos):
            if almot_done(updated_legos):
                index_model = greedy_algo(current_lego=updated_legos, lego_information=lego_information)
                updated_legos -= lego_information.lego_models[index_model]
                models_used_by_generation[index_model] += 1
            else:
                test_updated_lego = None
                while True:
                    random_index = np.random.randint(0, len(lego_information.lego_models) - 1)
                    test_updated_lego = np.subtract(updated_legos, lego_information.lego_models[random_index])
                    if change_was_made(updated_legos, test_updated_lego):
                        models_used_by_generation[random_index] += 1
                        break
                updated_legos = np.copy(test_updated_lego)


        cost_generation = np.dot(updated_legos, lego_information.lego_price)
        if cost_generation > best_solution_price:
            best_solution_price = cost_generation
            best_solution_models = np.copy(models_used_by_generation)
            print("{} : {}".format(repr(best_solution_models), best_solution_price))

        population_models[i] = models_used_by_generation
        population_models_cost[i] = cost_generation

    print("genes selection")
    POPULATION = lego_information.nb_lego * 2
    for i in range(0, 400000):
        # selection
        parent_a = np.argmax(population_models_cost)
        original_value = population_models_cost[parent_a]
        population_models_cost[parent_a] = sys.maxsize * -1
        parent_b = np.argmax(population_models_cost)
        population_models_cost[parent_a] = original_value

        for j in range(0, POPULATION):
            models_used_by_generation = np.zeros(lego_information.nb_models).astype("int32")
            for h in range(0, lego_information.nb_lego):
                value_to_set = population_models[parent_a][h]
                if population_models[parent_b][h] < population_models[parent_a][h]:
                    value_to_set = population_models[parent_b][h]

                if value_to_set > 0:
                    if np.random.random() < 0.99:
                        models_used_by_generation[h] = value_to_set
                    else:
                        models_used_by_generation[h] = value_to_set - 1

            updated_legos = np.copy(lego_information.initial_lego)
            for model_index in range(0, lego_information.nb_models):
                if models_used_by_generation[model_index] > 0:
                    updated_legos -= lego_information.lego_models[model_index] * models_used_by_generation[model_index]

            while not current_legos_done(updated_legos):
                if almot_done(updated_legos):
                    index_model = greedy_algo(current_lego=updated_legos, lego_information=lego_information)
                    updated_legos -= lego_information.lego_models[index_model]
                    models_used_by_generation[index_model] += 1
                else:
                    test_updated_lego = None
                    while True:
                        random_index = np.random.randint(0, len(lego_information.lego_models) - 1)
                        test_updated_lego = np.subtract(updated_legos, lego_information.lego_models[random_index])
                        if change_was_made(updated_legos, test_updated_lego):
                            models_used_by_generation[random_index] += 1
                            break
                    updated_legos = np.copy(test_updated_lego)

            cost_generation = np.dot(updated_legos, lego_information.lego_price)
            if cost_generation > best_solution_price:
                best_solution_price = cost_generation
                best_solution_models = np.copy(models_used_by_generation)
                print("{} : {}".format(repr(best_solution_models), best_solution_price))

            population_models[j] = models_used_by_generation
            population_models_cost[j] = cost_generation


def almot_done(current_lego):
    nb_negative_lego = 0
    for i in range(0, len(current_lego)):
        if current_lego[i] < 0:
            nb_negative_lego += 1
    return len(current_lego) - nb_negative_lego == 2


def change_was_made(current_legos, new_current_legos):
    positive_equals = True
    for i in range(0, len(current_legos)):
        if current_legos[i] > 0 and current_legos[i] != new_current_legos[i]:
            positive_equals = False
    return not positive_equals


def current_legos_done(current_legos):
    for i in range(0, len(current_legos)):
        if current_legos[i] > 0:
            return False
    return True


if __name__ == "__main__":
    file_name = "/home/ayoub/Desktop/school/INF8775/TP3/exemplaires/LEGO_50_50_1000"
    lego, price, models = load_data_form_file(file_name)
    lego_info = LegoInformation(lego, price, models)
    genetic_algo(lego_information=lego_info)
