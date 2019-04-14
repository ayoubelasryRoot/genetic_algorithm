import numpy as np
from load_data import load_data_form_file
import sys

np.set_printoptions(linewidth=np.inf)

models = None
objects_price = None
best_solution_models = None
best_solution_price = -1 * sys.maxsize


common_models = None


def solve_problem(lego_quantities, models_construction, model_index):
    global best_solution_price
    global best_solution_models

    copy_models_construction = np.copy(models_construction)

    if 0 <= model_index < len(copy_models_construction):
        copy_models_construction[model_index] += 1
        lego_quantities = np.subtract(lego_quantities, models[model_index])
    else:
        return
    if current_legos_done(lego_quantities):
        cost = np.dot(lego_quantities, objects_price)
        if cost is not None and (best_solution_price is None or cost > best_solution_price):
            best_solution_price = cost
            best_solution_models = np.copy(copy_models_construction)
            print("{} : {}".format(best_solution_models, best_solution_price))
        return

    if current_price(lego_quantities) <= best_solution_price:
        # no need to continue with this solution
        return

    tried_every_solution = False
    used_indices_models = np.zeros(len(models))
    done_due_to_one_object = is_only_one_left(lego_quantities)
    nb_tries = 0
    while not tried_every_solution and nb_tries < 4:
        model_index_to_try_first = check_best_solution(lego_quantities, used_indices_models)
        if current_price(lego_quantities - models[model_index]) > best_solution_price:
            solve_problem(lego_quantities, copy_models_construction, model_index_to_try_first)
        used_indices_models[model_index_to_try_first] = 1
        tried_every_solution = np.count_nonzero(used_indices_models) == len(used_indices_models)
        if done_due_to_one_object:
            break
        nb_tries += 1


def genetic_algo(initial_objects):
    global best_solution_price
    global best_solution_models
    global models

    POPULATION = 1000

    nb_lego_by_model = len(models[0])
    nb_models = len(models)
    population_models = np.zeros((POPULATION, nb_lego_by_model))
    population_models_cost = np.zeros(POPULATION)

    print("random population")
    for i in range(0, POPULATION):
        models_used_by_generation = np.zeros(nb_models).astype("int32")
        updated_legos = np.copy(initial_objects)
        while not current_legos_done(updated_legos):
            test_updated_lego = None
            while True:
                random_index = np.random.randint(0, len(models) - 1)
                test_updated_lego = np.subtract(updated_legos, models[random_index])
                if change_was_made(updated_legos, test_updated_lego):
                    models_used_by_generation[random_index] += 1
                    break
            updated_legos = np.copy(test_updated_lego)

        cost_generation = np.dot(updated_legos, objects_price)
        if cost_generation > best_solution_price:
            best_solution_price = cost_generation
            best_solution_models = np.copy(models_used_by_generation)
            print("{} : {}".format(best_solution_models, best_solution_price))

        population_models[i] = models_used_by_generation
        population_models_cost[i] = cost_generation

    print("genes selection")
    POPULATION = 100
    for i in range(0, 400000):
        # selection
        parent_a = np.argmax(population_models_cost)
        original_value = population_models_cost[parent_a]
        population_models_cost[parent_a] = sys.maxsize * -1
        parent_b = np.argmax(population_models_cost)
        population_models_cost[parent_a] = original_value

        for j in range(0, POPULATION):
            models_used_by_generation = np.zeros(nb_models).astype("int32")
            for h in range(0, nb_lego_by_model):
                value_to_set = population_models[parent_a][h]
                if population_models[parent_b][h] < population_models[parent_a][h]:
                    value_to_set = population_models[parent_b][h]

                if value_to_set > 0:
                    if np.random.random() < 0.99:
                        models_used_by_generation[h] = value_to_set
                    else:
                        models_used_by_generation[h] = value_to_set - 1

            updated_legos = np.copy(initial_objects)
            for model_index in range(0, nb_models):
                if models_used_by_generation[model_index] > 0:
                    updated_legos -= models[model_index] * models_used_by_generation[model_index]


            while not current_legos_done(updated_legos):
                test_updated_lego = None
                while True:
                    random_index = np.random.randint(0, len(models) - 1)
                    test_updated_lego = np.subtract(updated_legos, models[random_index])
                    if change_was_made(updated_legos, test_updated_lego):
                        models_used_by_generation[random_index] += 1
                        break
                updated_legos = np.copy(test_updated_lego)

            cost_generation = np.dot(updated_legos, objects_price)
            if cost_generation > best_solution_price:
                best_solution_price = cost_generation
                best_solution_models = np.copy(models_used_by_generation)
                print("{} : {}".format(repr(best_solution_models), best_solution_price))

            population_models[j] = models_used_by_generation
            population_models_cost[j] = cost_generation



def reduce_size_objects(initial_objects):
    global best_solution_price
    global best_solution_models
    global common_models
    POPULATION = 100

    models_selection = np.zeros(len(models)).astype("int32")
    for population in range(0, 100000):
        # print(models_selection)
        models_by_population = np.zeros((POPULATION, len(models[0])))
        models_cost_construction = np.zeros(POPULATION)

        for version in range(0, POPULATION):
            models_used_by_version = np.zeros(len(models))
            current_legos = np.copy(initial_objects)

            for index_selection in range(0, len(models_selection)):
                if np.random.random() < 0.98:
                    current_legos -= models_selection[index_selection] * models[index_selection]
                    models_used_by_version[index_selection] = models_selection[index_selection]

            while not current_legos_done(current_legos):
                new_current_legos = None
                while True:
                    random_index = np.random.randint(0, len(models) - 1)
                    new_current_legos = np.subtract(current_legos, models[random_index])
                    if change_was_made(current_legos, new_current_legos):
                        models_used_by_version[random_index] += 1
                        break
                current_legos = np.copy(new_current_legos)

            models_by_population[version] = np.copy(models_used_by_version)
            models_cost_construction[version] = np.dot(current_legos, objects_price)
            if models_cost_construction[version] > best_solution_price:
                best_solution_price = models_cost_construction[version]
                best_solution_models = np.copy(models_used_by_version)
                print("{} : {}".format(best_solution_models, best_solution_price))

        # selection for next generations
        top_models_indices = models_cost_construction.argsort()[-4:][::-1]
        # find common attributes between models
        for i in range(0, len(models_by_population[0])):
            min_value_model_object = np.zeros(4)
            k = 0
            for model_index in top_models_indices:
                min_value_model_object[k] = models_by_population[model_index][i]
                k += 1
            models_selection[i] = np.min(min_value_model_object)



def check_best_solution_without_tracking(objects):
    lowest_delta = sys.maxsize
    best_model_index = -1
    cost_best_model = -1 * sys.maxsize

    for index in range(0, len(models)):
        new_lego_peices = objects - models[index]
        delta = np.amax(new_lego_peices) - np.amin(new_lego_peices)
        if change_was_made(objects, new_lego_peices):
            cost = current_price(new_lego_peices)
            if delta < lowest_delta:
                cost_best_model = cost
                lowest_delta = delta
                best_model_index = index
            elif delta == lowest_delta and cost > cost_best_model:
                cost_best_model = cost
                lowest_delta = delta
                best_model_index = index
    return best_model_index


def check_best_solution(objects, used_models):
    lowest_delta = sys.maxsize
    best_model_index = -1
    cost_best_model = -1 * sys.maxsize

    for index in range(0, len(models)):
        if used_models[index] == 1:
            continue
        new_current_lego_pieces = objects - models[index]
        if not change_was_made(objects, new_current_lego_pieces):
            used_models[index] = 1
            continue
        delta = np.amax(new_current_lego_pieces) - np.amin(new_current_lego_pieces)
        cost = current_price(new_current_lego_pieces)
        if delta < lowest_delta:
            cost_best_model = cost
            lowest_delta = delta
            best_model_index = index
        elif delta == lowest_delta and cost > cost_best_model:
            cost_best_model = cost
            lowest_delta = delta
            best_model_index = index
    return best_model_index


def is_only_one_left(lego_quantities):
    total_left = 0
    for i in range(0, np.size(lego_quantities)):
        if lego_quantities[i] > 0:
            total_left += lego_quantities[i]
        if total_left > 1:
            return False
    return True


def change_was_made(current_legos, new_current_legos):
    positive_equals = True
    for i in range(0, len(current_legos)):
        if current_legos[i] > 0 and current_legos[i] != new_current_legos[i]:
            positive_equals = False
    return not positive_equals


def current_price(objects):
    result = 0
    for i in range(0, len(objects)):
        if objects[i] < 0:
            result -= objects[i] * objects_price[i]
    return result


def current_legos_done(current_legos):
    for i in range(0, len(current_legos)):
        if current_legos[i] > 0:
            return False
    return True


def current_legos_almost_done(current_legos, sum_initial):
    total = 0
    for i in range(0, len(current_legos)):
        if current_legos[i] > 0:
            total += current_legos[i]
    return total < 0.001*sum_initial


if __name__ == "__main__":
    file_name = "/home/ayoub/Desktop/school/INF8775/TP3/exemplaires/LEGO_50_50_1000"
    objects_price = np.array([4, 1, 2, 7, 6, 1, 1, 2, 3, 7])
    models = np.array([[2, 0, 0, 1, 1], [1, 1, 1, 0, 0]])
    current_objects = np.array([1, 4, 6, 7, 1, 3, 2, 1, 6, 7])

    current_objects, objects_price, models = load_data_form_file(file_name)

    #reduce_size_objects(current_objects)
    genetic_algo(initial_objects=current_objects)

    result = np.copy(current_objects)
    for i in range(0, np.size(best_solution_models)):
        model = best_solution_models[i] * models[i]
        result -= model
    if np.dot(result, objects_price) == best_solution_price:
        print("\nthe answer is legit bro")
    else:
        print("\nhum ... expecting {} not {}".format(np.dot(result, objects_price), best_solution_price))
