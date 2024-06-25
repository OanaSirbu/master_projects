from nltk.tokenize import word_tokenize


def well_formed_substring_table(sent, grammar):
    words = word_tokenize(sent)
    n = len(words)
    # Initialize the table with sets to ensure uniqueness
    table = [[set() for _ in range(n + 1)] for _ in range(n + 1)]  
    
    # Initialize elements T[i][i+1] that will contain the i-th word
    for i in range(n):
        table[i][i + 1] = {words[i]}
    
    # Apply the productions completing the table, until no more changes in the table are made
    changed = True
    while changed:
        changed = False
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for production in grammar:
                        A, *body = production
                        if len(body) == 2: 
                            B, C = body
                            if B in table[i][k] and C in table[k][j]:
                                table[i][j].add(A)
                                changed = True
                        elif len(body) == 1:  
                            B = body[0]
                            if B in table[i][k]:
                                table[i][j].add(A)
                                changed = True
    
    return table


my_grammar = [
    ('S', 'NP', 'VP'),
    ('NP', 'N'),
    ('NP', 'Det', 'N'),
    ('NP', 'NP', 'PP'),
    ('PP', 'P', 'NP'),
    ('VP', 'V'),
    ('VP', 'V', 'Adv'),
    ('VP', 'V', 'NP'),
    ('Det', 'the'),
    ('N', 'stars'),
    ('N', 'night'),
    ('N', 'sky'),
    ('V', 'shine'),
    ('Adv', 'brightly'),
    ('P', 'in'),
]


# define my own sentence
my_sentence = 'Stars shine brightly in the night sky'

# The algorithm finishes when no more reductions can be made.
# If we've obtained S (the sentence node) in T[0][n] we have succesfully parsed the sentence.
def parse_sent(sentence, grammar):
    T = well_formed_substring_table(sentence, grammar)
    if 'S' in T[0][len(sentence)]:
        print(T)
        return True
    else:
        return False

print(parse_sent(sentence=my_sentence, grammar=my_grammar))