from json import dump
import sys
import getopt

from genetic_algorithm import GeneticAlgorithm


def main(argv):

    mutation_rate = [.2, .15, .1, .09, .07, .04]
    max_generations = 1_000_000
    from_memory = False

    try:
        opts, args = getopt.getopt(
            argv, 'hm:g:f:', ['mutation_rate=', 'max_generations=', 'from_memory='])
    except getopt.GetoptError:
        print(
            'Usage: python main.py <mutation rate (float)> <maximum generations (int)> <load weights from file [--from_memory]>\nExample => python main.py 0.1 1_000 --from_memory')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'Usage: python main.py <mutation rate (float)> <maximum generations (int)> <load weights from file [--from_memory]>\nExample => python main.py 0.1 1_000 --from_memory')
            sys.exit()
        elif opt in ('-m', '--mutation_rate'):
            mutation_rate = float(arg)
        elif opt in ('-g', '--max_generations'):
            max_generations = int(arg)
        elif opt in ('-f', '--from_memory'):
            from_memory = bool(arg)

    with open('test_mutation.txt', 'a') as out_file:
            out_file.write('mutation rate: '+str(mutation_rate)+'\n')
    ga = GeneticAlgorithm(mutation_rate, max_generations, 10, 50, 10, 500, from_memory)
    ga.run()


if __name__ == '__main__':
    main(sys.argv[1:])
