# Copyright 2023 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy metric."""

import datasets
from sklearn.metrics import average_precision_score

import evaluate


_DESCRIPTION = """
This metric computes the area under the curve (AUC) for the Precision-Recall Curve (PR). summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.

You should use this metric:
- when your data is heavily imbalanced. As mentioned before, it was discussed extensively in this article by Takaya Saito and Marc Rehmsmeier. The intuition is the following: since PR AUC focuses mainly on the positive class (PPV and TPR) it cares less about the frequent negative class.
- when you care more about positive than negative class. If you care more about the positive class and hence PPV and TPR you should go with Precision-Recall curve and PR AUC (average precision).
"""

_KWARGS_DESCRIPTION = """
Args:
- references (array-like of shape (n_samples,) or (n_samples, n_classes)): True binary labels or binary label indicators.
- predictions (array-like of shape (n_samples,) or (n_samples, n_classes)): Model predictions. Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by decision_function on some classifiers).
- average (`str`): Type of average, and is ignored in the binary use case. Defaults to 'macro'. Options are:
    - `'micro'`: Calculate metrics globally by considering each element of the label indicator matrix as a label.
    - `'macro'`: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    - `'weighted'`: Calculate metrics for each label, and find their average, weighted by support (i.e. the number of true instances for each label).
    - `'samples'`: Calculate metrics for each instance, and find their average. Only works with the multilabel use case.
    - `None`:  No average is calculated, and scores for each class are returned. Only works with the multilabels use case.
- pos_label (int, float, bool or str): The label of the positive class. Only applied to binary y_true. For multilabel-indicator y_true, pos_label is fixed to 1.
- sample_weight (array-like of shape (n_samples,)): Sample weights. Defaults to None.
Returns:
    average_precision (`float` or array-like of shape (n_classes,)): Returns `float` of average precision score.
Example:
    Example 1:
        >>> average_precision_score = evaluate.load("pr_auc")
        >>> refs = np.array([0, 0, 1, 1])
        >>> pred_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> results = average_precision_score.compute(references=refs, prediction_scores=pred_scores)
        >>> print(round(results['average_precision'], 2))
        0.83

    Example 2:
        >>> average_precision_score = evaluate.load("pr_auc")
        >>> refs = np.array([0, 0, 1, 1, 2, 2])
        >>> pred_scores = np.array([[0.7, 0.2, 0.1],
        ...                         [0.4, 0.3, 0.3],
        ...                         [0.1, 0.8, 0.1],
        ...                         [0.2, 0.3, 0.5],
        ...                         [0.4, 0.4, 0.2],
        ...                         [0.1, 0.2, 0.7]])
        >>> results = average_precision_score.compute(references=refs, prediction_scores=pred_scores)
        >>> print(round(results['average_precision'], 2))
        0.77
"""

_CITATION = """
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PRAUC(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html"],
        )

    def _compute(
        self,
        references,
        predictions,
        average="macro",
        sample_weight=None,
        pos_label=1,
    ):
        return {
            "average_precision": average_precision_score(
                references,
                predictions,
                average=average,
                sample_weight=sample_weight,
                pos_labels=pos_label,
            )
        }