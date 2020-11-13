#confidence_prediction

we want to get an **estimate of the confidence of the prediction**. The idea is to find a confidence value for the prediction (a sort of probability) to see if RoBERTa is confident on the prediction or not. We hope that if the confidence is high then the predicted code is correct. If this is working we can run the code on a opensource systems and if we find an if condition with high confidence that is different from the one predicted by RoBERTa then this should be a bug.  
You can find the files in predict_confidence folder

How to get predictions and probabilities running the chosen model:

```
python3 run_on_test_set.py --model_path models/model_1/checkpoint --test_set_inputs_path input/masked_code.txt --test_set_targets_path input/mask_after.txt --predictions_path result_1_fake.txt
```
```
python3 run_on_test_set.py --model_path models/model_2/checkpoint --test_set_inputs_path input/masked_code.txt --test_set_targets_path input/mask_after.txt --predictions_path result_2.txt
```
```
python3 run_on_test_set.py --model_path models/model_3/checkpoint --test_set_inputs_path input/masked_code.txt --test_set_targets_path input/mask_after.txt --predictions_path result_3.txt
```

how to postprocess results to retrieve median and other metrics
```
python3 post_process_result.py --predictions_path result.txt
```