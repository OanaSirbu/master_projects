from tabulate import tabulate
import random
import string


class Production:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.left} -> {' '.join(self.right)}"
    

def generate_unique_symbol(length):
    return 'NewT_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def transform_production(production):
    if len(production.right) > 2:
        new_symbol = generate_unique_symbol(len(production.right) - 1)
        new_productions = []
        new_productions.append(Production(production.left, [production.right[0], new_symbol]))
        for i in range(1, len(production.right) - 1):
            new_productions.append(Production(new_symbol, [production.right[i], production.right[i+1]]))
        return new_productions
    elif len(production.right) in [1, 2]:  
        return [production]
    else:
        return []


def preprocess_productions(productions):
    transformed_productions = []
    for production in productions:
        transformed_productions.extend(transform_production(production))
    return transformed_productions


def parse(sentence, productions):
    words = sentence.split()
    N = len(words)
    T = [[[] for _ in range(N + 1)] for _ in range(N + 1)]

    transformed_productions = preprocess_productions(productions)
    print("Transformed productions: ", transformed_productions)

    for i in range(N):
        for production in transformed_productions:
            if len(production.right) == 1 and production.right[0] == words[i]:
                T[i][i + 1].append(production)

    updated = True
    while updated:
        updated = False
        for i in range(N):
            for j in range(i + 2, N + 1):
                for k in range(i + 1, j):
                    for prod_ik in T[i][k]:
                        for prod_kj in T[k][j]:
                            for prod in transformed_productions:
                                if len(prod.right) == 2:
                                    if prod.right[0] == prod_ik.left and prod.right[1] == prod_kj.left and prod not in T[i][j]:
                                        T[i][j].append(prod)
                                        updated = True
    return T


productions = [
    Production('S', ['NP', 'VP']),
    Production('NP', ['Det', 'N']),
    Production('NP', ['Det', 'JJ', 'N']),
    Production('VP', ['V', 'RB']),
    Production('VP', ['VP', 'PP']),
    Production('PP', ['P', 'NP']),
    Production('Det', ['the']),
    Production('N', ['stars']),
    Production('JJ', ['big']),
    Production('N', ['sky']),
    Production('N', ['night']),  
    Production('RB', ['brightly']),
    Production('V', ['shine']),
    Production('P', ['in'])
]

sentence = 'the big stars shine brightly in the sky'

result = parse(sentence, productions)

for row in result:
    formatted_row = ["" if cell == [] else cell for cell in row]
    print(tabulate([formatted_row], tablefmt="grid"))