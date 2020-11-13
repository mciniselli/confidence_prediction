import argparse

import sys

import numpy as np

def main():
    # Instantiate argument parser
    parser = argparse.ArgumentParser()    
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="The path of the file in which predictions will be printed.")

    # Generate args
    args = parser.parse_args()

    with open(args.predictions_path) as f:
        inputs = f.readlines()
    inputs = [x.strip() for x in inputs][1:] 

    print(len(inputs))   


    #'PREDICTIONS |_| LABEL |_| PROBABILITIES |_| MEAN |_| MEDIAN |_| MIN |_| PERFECT_PRED'

    num_perfect=0
    num_non_perfect=0

    median_perfect=0
    median_non_perfect=0

    num_non_perfect_greater=0
    num_perfect_lower=0

    i = 0
    while i < len(inputs):
        input = inputs[i]


        input_splitted=input.split("|_|")

        if input_splitted[0] == "ERROR PREDICTION":
            print("SKIPPED")
            i+=1
            continue

        is_perfect=False

        if str(input_splitted[-1])=="True":
            is_perfect=True
            num_perfect+=1
        else:
            num_non_perfect+=1

        if is_perfect:
            median_perfect+=float(input_splitted[4])
            if float(input_splitted[4])<0.6:
                num_perfect_lower+=1

        else:
            median_non_perfect+=float(input_splitted[4])

            if float(input_splitted[4])>0.9:
                num_non_perfect_greater+=1


        i+=1

    print("perfect pred {}".format(num_perfect))

    print("median perfect {} - non perfect {}".format(median_perfect/num_perfect, median_non_perfect/num_non_perfect))

    print("anomalies perfect {} - anomalies non perfect {}".format(num_perfect_lower, num_non_perfect_greater))

if __name__ == "__main__":
    main()