
## Experiment01:

Thanks for providing clarification. Here's a refined version:

---

**Chunking:** The complete essay is first divided into chunks of a specified size, such as `max_words_per_chunk=416`.

**Feature Set:** Each chunk is then processed by an Encoder-based Large Language Model (LLM) for feature extraction. For the first chunk, two features are extracted:

1. The token corresponding to `[CLS]`.
2. The maximum value among all other tokens in the chunk.

For all subsequent chunks, only the maximum value among all tokens is considered.

Finally, the feature extraction process results in two sets of features:

Got it, here's the refined description:

---

The feature extraction process results in two sets of features:

1. **Token corresponding to CLS:** For the first chunk, the feature corresponds to the token representing `[CLS]`.
   
2. **Max of Max for all other chunks:** For all chunks except the first one, the feature is obtained by selecting the maximum value among all tokens in each chunk. Then, from these maximum values across all chunks, the maximum value is selected. This value represents the "Max of Max" feature for the entire essay.

## Regression 
We used LightGBM Regressor with below paramter for the predicting score between 1-6.

# Define parameters for LightGBM

Certainly! Below is the reformatted text with the LightGBM parameters organized into a code block and the metric data presented in a table:

```python
# Define parameters for LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',              # Root Mean Squared Error
    'num_leaves': 112,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 2,
    'max_depth': 9,
    'verbose': 0,
}

```

**Prediction Output:** Please refer file **predict_regressor_01**.

**Metric:**

| Metric                               | Value |
|--------------------------------------|-------|
| Matthews Correlation Coefficient (MCC)| 0.67 |
| Ovell Accuracy                             | 0.61  |
| Accuracy for class 0                 | 0.22  |
| Accuracy for class 1                 | 0.72  |
| Accuracy for class 2                 | 0.68  |
| Accuracy for class 3                 | 0.59  |
| Accuracy for class 4                 | 0.34  |
| Accuracy for class 5                 | 0.06  |


