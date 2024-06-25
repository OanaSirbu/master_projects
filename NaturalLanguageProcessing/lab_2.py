def generate_left_corner_table(grammar):
    left_corner_table = {}

    for rule in grammar:
        lhs, rhs = rule.split(' -> ')
        rhs_symbols = rhs.split()

        first_symbol = rhs_symbols[0]

        if lhs not in left_corner_table:
            left_corner_table[lhs] = []

        if first_symbol.startswith('"') and first_symbol.endswith('"'): 
            for symbol in rhs_symbols:
                left_corner_table[lhs].append(symbol)
        elif first_symbol not in left_corner_table[lhs]:
            left_corner_table[lhs].append(first_symbol)

    return left_corner_table


grammar = [
    'S -> NP VP',
    'S -> VP',
    'NP -> DT NN',
    'NP -> DT JJ NN',
    'NP -> PRP',
    'VP -> VBP NP',
    'VP -> VBP VP',
    'VP -> VBG NP',
    'VP -> TO VP',
    'VP -> VB',
    'VP -> VB NP',
    'NN -> "show" "book"',
    'PRP -> "I"',
    'VBP -> "am"',
    'VBG -> "watching"',
    'VB -> "show"',
    'DT -> "a" "the"',
    'MD -> "will"'
]

left_corner_table = generate_left_corner_table(grammar)

# print(left_corner_table)

# print("Symbol\tLeft corner")
# for symbol, corners in left_corner_table.items():
#     print(f"{symbol}\t{', '.join(corners)}")

def find_symbols_for_word(word, left_corner_table):
    symbols = []
    for symbol, corners in left_corner_table.items():
        if word in corners:
            symbols.append(symbol)
    return symbols


def find_production(symbol, left_corner_table):
    # Find a production for the current symbol based on the current word
    if symbol in left_corner_table:
        return left_corner_table[symbol]
    return None


def bottom_up_parse(input_string, current_word_index, target, parse_tree, left_corner_table):
    words = input_string.split()
    current_symbols = find_symbols_for_word('"' + words[current_word_index] + '"', left_corner_table) or [target]

    while current_symbols:
        next_symbols = []
        for current_symbol in current_symbols:
            symbols_associated_with_word = find_symbols_for_word(current_symbol, left_corner_table)
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

    return current_word_index, parse_tree


def top_down_predict(input_string, current_index, remaining_symbols, current_non_terminal, parse_tree, left_corner_table):
    if not remaining_symbols:
        # If there are no remaining symbols, return True to indicate successful prediction
        return True
    
    # Take the first word from the remaining input string
    current_word = input_string.split()[current_index]

    # Find the non-terminal in the left corner table corresponding to the current word
    next_non_terminal = None
    for row in left_corner_table:
        if current_word in row and current_non_terminal in row:
            next_non_terminal = row[0]
            break

    if next_non_terminal:
        # Recursively call the function with the next non-terminal and remaining symbols
        if top_down_predict(input_string[len(current_word)+1:], remaining_symbols[1:], next_non_terminal, parse_tree, left_corner_table):
            # If the prediction is successful, add the production to the parse tree
            production = f"{current_non_terminal} -> {next_non_terminal} {' '.join(remaining_symbols)}"
            parse_tree.append(production)
            return True
    else:
        # If no match is found, backtrack
        return False

    # Backtrack if the prediction is unsuccessful
    return False


def parse(input_string, left_corner_table):
    parse_tree = []
    current_word_index = 0
    targets = ['S']  # Initialize targets list with the starting non-terminal
    bottom_up_phase = True

    while True:
        if bottom_up_phase:
            current_word_index, parse_tree = bottom_up_parse(input_string, current_word_index, targets[-1], parse_tree, left_corner_table)
            # Check if the current target has been reached
            if targets[-1] in [production.split(' -> ')[0] for production in parse_tree]:
                # Find the next target from the left corner table
                next_target = find_production(targets[-1], left_corner_table)[1].split()[1]
                # Add the next target to the targets list
                targets.append(next_target)
                bottom_up_phase = False  # Switch to top-down prediction
                current_word_index = 0  # Reset current_word_index for top-down prediction
        else:
            # Perform top-down prediction for the current target
            current_target = targets[-1]
            current_production = find_production(current_target, left_corner_table)[1]
            remaining_symbols = current_production.split()[1:]
            if top_down_predict(input_string, current_word_index, remaining_symbols, current_target, parse_tree, left_corner_table):
                print("Top-down prediction successful for target:", current_target)
                bottom_up_phase = True  # Switch back to bottom-up parsing
            else:
                print("Top-down prediction failed for target:", current_target)
                return []  # Parsing failed

        # Check if parsing is completed
        if not bottom_up_phase and current_word_index >= len(input_string.split()):
            break

    return parse_tree



# def parse(input_string, left_corner_table):
#     parse_tree = []
#     words = input_string.split()
#     current_word_index = 0
#     target = 'S'

#     # Start bottom-up search with non-terminals associated with the first word
#     current_symbols = find_symbols_for_word('"' + words[current_word_index] + '"', left_corner_table) or [target]

#     # Bottom-up search
#     while current_symbols:
#         next_symbols = []
#         for current_symbol in current_symbols:
#             symbols_associated_with_word = find_symbols_for_word(current_symbol, left_corner_table)
#             if symbols_associated_with_word:
#                 # Found symbols associated with the current word
#                 next_symbols.extend(symbols_associated_with_word)
#                 # Update parse tree and current word index
#                 production = f"{current_symbol} -> {' '.join(symbols_associated_with_word)}"
#                 parse_tree.append(production)
#                 if symbols_associated_with_word != [target]:
#                     current_word_index += 1
#                 break
#         else:
#             # Backtrack if no symbols found for the current word
#             parse_tree.pop() if parse_tree else None
#             current_word_index -= 1
#             current_symbols = [production.split(' -> ')[0] for production in parse_tree]
#             continue
        
#         # Check if 'S' is reached
#         if target in symbols_associated_with_word:
#             break
        
#         # Update current symbols for the next iteration
#         current_symbols = symbols_associated_with_word

#     print(parse_tree)

#     # Top-down prediction
#     second_target = find_production(target, left_corner_table)[1]
#     print(second_target)

#     # back to bottom-up


#     return parse_tree



# Test input string
input_string = "I am watching a show"

parse_tree = parse(input_string, left_corner_table)

# Print the parse tree
print(parse_tree)