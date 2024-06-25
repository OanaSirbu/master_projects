def find_symbols_for_word(word, grammar):
    symbols = []
    for rule in grammar:
        left, right = rule.split(' -> ')
        right_symbols = right.split()
        if word in right_symbols:
            symbols.append(left)
    return symbols


def generate_left_corner_table(grammar):
    left_corner_table = {}

    for rule in grammar:
        left, right = rule.split(' -> ')
        right_symbols = right.split()

        first_symbol = right_symbols[0]

        if left not in left_corner_table:
            left_corner_table[left] = []

        if first_symbol.startswith('"') and first_symbol.endswith('"'):
            left_corner_table[left].append(first_symbol.strip('"'))
        elif first_symbol not in left_corner_table[left]:
            left_corner_table[left].append(first_symbol)

    return left_corner_table


def top_down(grammar, num_top_down, prev_symbol, symbol, words):
    for rule in grammar:
        left, right = rule.split(' -> ')
        right_symbols = right.split()
        if left == symbol:
            for i, right_symbol in enumerate(right_symbols):
                if i >= num_top_down:
                    if right_symbol in words:
                        return prev_symbol
                    else:
                        return right_symbol
    return None


def bottom_up(parse_tree, words, current_word_index, target, left_corner_table):
    print('target: ', target)
    print("bottom word: ", words[current_word_index])
    current_symbols = find_symbols_for_word('"' + words[current_word_index] +'"', grammar) or [target]
    print("current symbol ", current_symbols)

    while current_symbols:
        next_symbols = []
        for current_symbol in current_symbols:
            symbols_associated_with_word = find_symbols_for_word(current_symbol, grammar)
            if symbols_associated_with_word:
                next_symbols.extend(symbols_associated_with_word)
                production = f"{current_symbol} -> {' '.join(symbols_associated_with_word)}"
                parse_tree.append(production)
                if symbols_associated_with_word != [target]:
                    current_word_index += 1
                break
        else:
            parse_tree.pop() if parse_tree else None
            current_word_index -= 1
            current_symbols = [production.split(' -> ')[0] for production in parse_tree]
            continue

        if target in symbols_associated_with_word:
            break

        current_symbols = symbols_associated_with_word

    return parse_tree, current_word_index


def parse(input_string, left_corner_table):
    parse_tree = []
    words = input_string.split()
    current_word_index = 0
    target = 'S'
    num_top_down = 0

    parse_tree, current_word_index = bottom_up(parse_tree, words, current_word_index, target, left_corner_table)

    while current_word_index < len(words):
        prev_target = target
        new_target = top_down(grammar, num_top_down, prev_target, target, words)
        parse_tree, current_word_index = bottom_up(parse_tree, words, current_word_index, new_target, left_corner_table)
        target = new_target
        print(parse_tree)

    return parse_tree


input_string = "I explore the stars"
grammar = [
    'S -> NP VP',
    'S -> VP',
    'NP -> PRP',
    'NP -> DT NN',
    'VP -> VBP NP',
    'VP -> VB NP',
    'NN -> "stars"',
    'PRP -> "I"',
    'VB -> "explore"',
    'DT -> "the"',
    'MD -> "will"'
]

left_corner_table = generate_left_corner_table(grammar)
print("Left corner table as dictionary is: ", left_corner_table)

parse_tree = parse(input_string, left_corner_table)

print(parse_tree)