import argparse
import torch

from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer
)

import sys

import numpy as np

def main():
    # Instantiate argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The model to use for the prediction.")
    parser.add_argument("--test_set_inputs_path", type=str, required=True,
                        help="The path of the test set containing the inputs to provide to the model.")
    parser.add_argument("--test_set_targets_path", type=str, required=True,
                        help="The path of the test set containing the targets that the model should generate.")    
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="The path of the file in which predictions will be printed.")
    parser.add_argument("--cpu", action='store_true', required=False,
                        help="Use it to force to run on CPU")

    # Generate args
    args = parser.parse_args()

    # Predictions
    predictions_file = open(args.predictions_path, 'w')
    predictions_file.write('PREDICTIONS |_| LABEL |_| PROBABILITIES |_| MEAN |_| MEDIAN |_| MIN |_| PERFECT_PRED' + '\n')

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMaskedLM, RobertaTokenizer

    # Prepare the tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_path)

    # Prepare the model
    model = model_class.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu:
        print("Running on CPU")
        device = "cpu"
    model.to(device)

    # Test set inputs
    with open(args.test_set_inputs_path) as f:
        inputs = f.readlines()
    inputs = [x.strip() for x in inputs]


    # Test set inputs
    with open(args.test_set_targets_path) as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]

    assert(len(inputs)==len(labels))

    print("Length input and labels: {}".format(len(inputs)))

    i = 0
    while i < len(inputs):
        try:
            print(str(i+1) + '  out of ' + str(len(inputs)))
            input = inputs[i]
            label=labels[i]



            indexed_tokens = tokenizer.encode(input)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(device)
            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs[0]

            predicted_sentence = []

            token_weights = []

            probs = torch.nn.functional.softmax(predictions[0], 1)
            top_weights, top_indices = torch.topk(probs, 1, sorted=True)
            for j, pred_idx in enumerate(top_indices):
                # predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
                token_weight = top_weights[j]
                if pred_idx != tokenizer.convert_tokens_to_ids('<z>'):
                    predicted_sentence.append(pred_idx)
                    token_weights.append((token_weight.detach().numpy())[0])
                    # print((token_weight.detach().numpy())[0])
                else:
                    break


            # for token in torch.argmax(predictions[0], 1).cpu().numpy():
            #     if token != tokenizer.convert_tokens_to_ids('<z>'):
            #         predicted_sentence.append(token)
            #     else:
            #         break

            prediction = tokenizer.decode(predicted_sentence)

            print("{}  {}".format(prediction, label))

            perfect_pred=False

            if prediction.replace(" ","")==label.replace(" ","").replace("<z>", ""):
                perfect_pred = True

            line=list()

            line.append(prediction)
            line.append(label.replace("<z>",""))
            line.append(str(token_weights))
            line.append(str(np.mean(token_weights)))
            line.append(str(np.median(token_weights)))
            line.append(str(np.min(token_weights)))
            line.append(str(perfect_pred))

            predictions_file.write("|_|".join(line) + '\n')
        except Exception as e:
            line=list()

            line.append("ERROR PREDICTION")
            line.append("ERROR LABEL")
            line.append("0")
            line.append("0")
            line.append("0")
            line.append("0")
            line.append("0")

            predictions_file.write("|_|".join(line) + '\n')
        i += 1

if __name__ == "__main__":
    main()