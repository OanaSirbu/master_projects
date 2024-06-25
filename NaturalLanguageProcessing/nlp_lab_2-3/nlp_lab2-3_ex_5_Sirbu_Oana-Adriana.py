from nltk.parse.stanford import StanfordParser, StanfordDependencyParser


sfd_parser_dir = '/home/oana/Downloads/stanford-parser-4.2.0/stanford-parser-full-2020-11-17/'
model_path = sfd_parser_dir + "stanford-parser-4.2.0-models.jar"
jar_path = sfd_parser_dir + "stanford-parser.jar"


constituency_parser = StanfordParser(path_to_jar=jar_path,path_to_models_jar=model_path)
dependency_parser = StanfordDependencyParser(path_to_jar=jar_path, path_to_models_jar=model_path)


def parse_sentences(input_file, output_file):
    with open(input_file, 'r') as inputfile, open(output_file, 'w') as outputfile:
        sentences = inputfile.readlines()
        for i, sentence in enumerate(sentences, 1):
            sentence = sentence.strip()
            
            constituency_parse = list(constituency_parser.raw_parse(sentence))[0]
            constituency_tree = constituency_parse.__str__()

            dependency_parse = dependency_parser.raw_parse(sentence)
            dependency_tree = [list(parse.triples()) for parse in dependency_parse][0]

            outputfile.write(f"Sentence - number [{i}]\n")
            outputfile.write(f"{sentence}\n")
            outputfile.write("Constituency parsing: \n")
            outputfile.write(f"{constituency_tree} \n")
            outputfile.write("Dependency parsing: \n")
            outputfile.write(f"{dependency_tree} \n")
            outputfile.write("-" * 35 + "\n")


input_file = "nlp_lab_2-3/five_sents_Sirbu_Oana-Adriana.txt"
output_file = "double_parsed_sentences_Sirbu_Oana-Adriana.txt"

parse_sentences(input_file, output_file)
