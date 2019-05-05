

################################################################################
#
# Ce script vérifie si votre solution est valide. C'est le script qui sera
# utilisé pour la correction, donc assurez-vous que la sortie de votre
# script tp.sh est compatible avec ce script-ci.
#
# Argument 1 : Path vers l'exemplaire
# Argument 2 : Path vers la solution de cet exemplaire
#
# Exemple d'utilisation :
#
#   1. Vous exécutez votre algorithme avec tp.sh et vous envoyez son résultat
#      vers un fichier texte :
#
#      ./tp.sh -e /home/pholi/INF8775/TP3/LEGO_50_50_1000 > sol_50_50_1000.txt
#
#   2. Vous vérifiez si votre solution est valide avec ce script-ci :
#
#      python3 sol_check.py /home/pholi/INF8775/TP3/LEGO_50_50_1000 sol_50_50_1000.txt
#
#
# Contactez-moi en cas de problème (philippe.olivier@polymtl.ca).
#
################################################################################


import pathlib
import sys


def get_parts_delta(sol):
    # For each model in the solution, multiply the number times this model is
    # used by its list of required parts. Sum all these lists of parts to get
    # the list of all required parts to cover the solution, and substract from
    # that the list of loose parts. From the resulting list: positive numbers
    # represent missing parts, negative numbers represent uncovered parts.
    parts_delta = [required - loose for loose, required in
                   zip(loose_parts,
                       [sum(p) for p in
                        zip(*[[sol[m] * p for p in models[m]]
                                for m in range(num_models)])])]
    return parts_delta


# Initial sanity checks
if len(sys.argv) != 3:
    exit("ERREUR : Ce script de vérification de solution prend deux " + \
         "arguments en entrée. Voir le code source pour un exemple.")
if not pathlib.Path(sys.argv[1]).is_file():
    exit("ERREUR : Fichier " + sys.argv[1] + " inexistant.")
if not pathlib.Path(sys.argv[2]).is_file():
    exit("ERREUR : Fichier " + sys.argv[2] + " inexistant.")

# Parse instance
try:
    raw_instance = open(sys.argv[1]).readlines()
    raw_instance = [raw_instance[i][:-1] for i in range(len(raw_instance))]
    num_part_types = int(raw_instance[0])
    loose_parts = tuple(int(i) for i in raw_instance[1].split(' ')[:-1])
    part_prices = tuple(int(i) for i in raw_instance[2].split(' ')[:-1])
    num_models = int(raw_instance[3])
    models = []
    for i in range(num_models):
        models.append(tuple(int(j) for j in raw_instance[i+4].split(' ')[:-1]))
    models = tuple(models)
except:
    exit("ERREUR : Problème avec le format de l'instance.")

# Parse solution
try:
    raw_solution = open(sys.argv[2]).readlines()
    raw_solution = raw_solution[-1]
    solution = raw_solution.split(' ')
    if '\n' in solution[-1]:
        solution[-1] = solution[-1].strip('\n')
    if solution[-1] == '':
        solution = solution[:-1]
    solution = [int(i) for i in solution]
except:
    exit("ERREUR : Problème avec le format de la solution.")

# Ensure that all parts are covered
if sum([-part for part in [min(0, part) for part in get_parts_delta(solution)]]) > 0:
    print("ERREUR : Certaines de vos pièces n'appartiennent à aucun modèle.")
    exit(0)

# Compute the solution value
z = sum(part_card * part_price for part_card, part_price in
        zip([max(0, part) for part in get_parts_delta(solution)], part_prices))

# Print the solution
print("Votre solution est valide et sa valeur est de ", z, ".", sep='')
