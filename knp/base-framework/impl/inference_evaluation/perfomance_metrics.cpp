/**
 * @file perfomance_metrics.cpp
 * @brief Functions to calculate model statistics.
 * @kaspersky_support D. Postnikov
 * @date 24.07.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <knp/framework/inference_evaluation/perfomance_metrics.h>


namespace knp::framework::inference_evaluation
{

float get_precision(size_t true_positives, size_t false_positives)
{
    if (true_positives + false_positives == 0) return 0.F;
    return static_cast<float>(true_positives) / (true_positives + false_positives);
}


float get_recall(size_t true_positives, size_t false_negatives)
{
    if (true_positives + false_negatives == 0) return 0.F;
    return static_cast<float>(true_positives) / (true_positives + false_negatives);
}


float get_prevalence(size_t true_positives, size_t false_negatives, size_t false_positives, size_t true_negatives)
{
    const size_t total = true_positives + false_negatives + false_positives + true_negatives;
    if (total == 0) return 0.F;
    return static_cast<float>(true_positives + false_negatives) / total;
}


float get_accuracy(size_t true_positives, size_t false_negatives, size_t false_positives, size_t true_negatives)
{
    const size_t total = true_positives + false_negatives + false_positives + true_negatives;
    if (total == 0) return 0.F;
    return static_cast<float>(true_positives + true_negatives) / total;
}


float get_f_score(float precision, float recall)
{
    if (precision * recall == 0) return 0.F;
    return 2.F * precision * recall / (precision + recall);
}

}  // namespace knp::framework::inference_evaluation
