import numpy as np

INDEX_TOTAL_OBJECTS_BY_MODEL = 0
INDEX_OF_OBJECTS = 1
INDEX_OF_PRICES = 2
INDEX_OF_TOTAL_MODELS = 3
INDEX_OF_MODELS = INDEX_OF_TOTAL_MODELS + 1

def load_data_form_file(file_name):
    current_objects = []
    objects_price = []
    models = []

    with open(file_name) as f:
        content = f.readlines()
        current_objects = np.array(content[INDEX_OF_OBJECTS].split(' ')[:-1]).astype("int32")
        objects_price = np.array(content[INDEX_OF_PRICES].split(' ')[:-1]).astype("int32")

        total_models = int(content[INDEX_OF_TOTAL_MODELS])
        total_objects = int(content[INDEX_TOTAL_OBJECTS_BY_MODEL])
        models = np.empty([total_models, total_objects]).astype("int32")
        for i in range(INDEX_OF_MODELS, int(content[INDEX_OF_TOTAL_MODELS]) + INDEX_OF_MODELS):
            test = np.array(content[i].split(' ')[:-1]).astype(np.int)
            models[i-INDEX_OF_MODELS] = test
    f.close()
    return current_objects, objects_price, models



