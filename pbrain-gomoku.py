from genetic_algorithm import genetic_algorithm

def main():
    ga = genetic_algorithm(100)
    ga.initial_population()
    ga.dispay_population_info()

main()