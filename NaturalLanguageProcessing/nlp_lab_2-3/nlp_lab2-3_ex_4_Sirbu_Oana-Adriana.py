from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser

sfd_parser = '/home/oana/Downloads/stanford-parser-4.2.0/stanford-parser-full-2020-11-17/stanford-parser.jar'
sfd_models = '/home/oana/Downloads/stanford-parser-4.2.0/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'

parser_instance = StanfordParser(path_to_jar=sfd_parser, path_to_models_jar=sfd_models)

my_sentence = "The researcher studies bright supernovas"

parsed_tree = list(parser_instance.raw_parse(my_sentence))
print(parsed_tree) # this is just a check, I got the expected output

depend_parser = StanfordDependencyParser(path_to_jar=sfd_parser, path_to_models_jar=sfd_models)

dep = list(depend_parser.raw_parse(my_sentence))[0]

with open("parser_output.txt", "w") as output_file:
    for triple in dep.triples():
        output_file.write(str(list(triple)) + '\n')

# I will print here the output for an easier verification
# [('studies', 'VBZ'), 'nsubj', ('researcher', 'NN')] [('researcher', 'NN'), 'det', ('The', 'DT')]
# [('studies', 'VBZ'), 'obj', ('supernovas', 'NNS')] [('supernovas', 'NNS'), 'amod', ('bright', 'JJ')]