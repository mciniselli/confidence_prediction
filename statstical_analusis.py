import argparse

import sys

import numpy as np

def main():
    # Instantiate argument parser
    parser = argparse.ArgumentParser()    
    parser.add_argument("--file_path", type=str, required=True,
                        help="The path of the file you want to count the tokens for.")

    # Generate args
    args = parser.parse_args()

    with open(args.file_path) as f:
        inputs = f.readlines()
    inputs = [x.strip() for x in inputs][1:] 

    print(len(inputs))   

    num_tokens=list()

    for r in inputs:
        r_splitted=r.split(" ")
        print(r)
        print(len(r_splitted))
        num_tokens.append(len(r_splitted))

    print(np.mean(num_tokens))
    print(np.var(num_tokens))

if __name__ == "__main__":
    main()