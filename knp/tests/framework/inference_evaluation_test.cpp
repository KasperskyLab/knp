/**
 * @file inference_evaluation_test.cpp
 * @brief Inference evaluation test.
 * @kaspersky_support D. Postnikov
 * @date 21.08.2025
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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

#include <knp/framework/inference_evaluation/classification.h>

#include <tests_common.h>


class ClassificationTestHelper : public knp::framework::data_processing::classification::Dataset
{
public:
    ClassificationTestHelper()
    {
        steps_required_for_inference_ = 4;
        steps_per_frame_ = 1;
        classes_amount_ = 2;
        data_for_inference_ = {{1, {}}, {1, {}}, {0, {}}, {0, {}}};
    }
};


TEST(InferenceEvaluation, Classification)
{
    ClassificationTestHelper dataset;
    std::vector<knp::core::messaging::SpikeMessage> spikes;
    spikes.push_back({{knp::core::UID(false), 0}, {1, 3}});
    spikes.push_back({{knp::core::UID(false), 1}, {5, 1}});
    spikes.push_back({{knp::core::UID(false), 2}, {2, 0}});
    spikes.push_back({{knp::core::UID(false), 3}, {3, 1}});

    knp::framework::inference_evaluation::classification::InferenceResultForClass::InferenceResultsProcessor processor;
    processor.process_inference_results(spikes, dataset);

    auto const& res = processor.get_inference_results();

    ASSERT_EQ(res[0].get_total_votes(), 2);
    ASSERT_EQ(res[0].get_true_positives(), 1);
    ASSERT_EQ(res[0].get_false_negatives(), 0);
    ASSERT_EQ(res[0].get_false_positives(), 1);
    ASSERT_EQ(res[0].get_true_negatives(), 2);
    ASSERT_EQ(res[1].get_total_votes(), 2);
    ASSERT_EQ(res[1].get_true_positives(), 2);
    ASSERT_EQ(res[1].get_false_negatives(), 0);
    ASSERT_EQ(res[1].get_false_positives(), 0);
    ASSERT_EQ(res[1].get_true_negatives(), 2);

    std::stringstream csv_res;
    processor.write_inference_results_to_stream_as_csv(csv_res);

    ASSERT_EQ(
        csv_res.str(),
        "CLASS,TOTAL_VOTES,TRUE_POSITIVES,FALSE_NEGATIVES,FALSE_POSITIVES,TRUE_NEGATIVES,PRECISION,RECALL,PREVALENCE,"
        "ACCURACY,F_MEASURE\n0,2,1,0,1,2,0.5,0.5,0.25,0.75,0.5\n1,2,2,0,0,2,1,1,0.5,1,1\n");
}
