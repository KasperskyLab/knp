#pragma once

#include <cstdint>

#include "type_traits.h"


/**
 * @brief Namespace for neuron traits.
 */
namespace knp::neuron_traits
{

/**
 * @brief Type that represents a calcium-based astrocyte.
 *
 * @note This type is intended to be used only as a template argument. It does not contain any data members.
 */
struct AstrocyteCell;


/**
 * @brief Name of the population tag that stores the UIDs of the delta-synapse projections that an astrocyte
 * population controls.
 *
 * @details Value type is `std::vector<knp::core::UID>`. If the tag is not set, the astrocyte still tracks calcium,
 * but does not modulate any projection.
 */
constexpr char astrocyte_modulated_projections_tag[] = "astrocyte_modulated_projections";


/**
 * @brief Structure for astrocyte default values.
 */
template <>
struct default_values<AstrocyteCell>
{
    /**
     * @brief Default initial calcium level.
     */
    constexpr static float calcium_ = 0.F;

    /**
     * @brief Default initial value of the calcium rising/decaying flag.
     */
    constexpr static bool rising_ = false;

    /**
     * @brief Default initial number of monitored spikes accumulated since calcium was last at 0.
     */
    constexpr static uint64_t spike_counter_ = 0;

    /**
     * @brief Default initial total number of monitored spikes observed over the whole simulation.
     */
    constexpr static uint64_t total_spikes_ = 0;

    /**
     * @brief Default initial number of times the calcium rise has been activated.
     */
    constexpr static uint64_t activation_count_ = 0;

    /**
     * @brief Default initial number of monitored spikes received so far during the current step.
     */
    constexpr static uint64_t monitored_spikes_this_step_ = 0;

    /**
     * @brief Default calcium level at which calcium stops rising and starts decaying.
     */
    constexpr static float calcium_max_ = 1000.F;

    /**
     * @brief Default amount by which calcium rises or decays on every active step.
     */
    constexpr static float calcium_step_ = 1.F;

    /**
     * @brief Default calcium level above which the astrocyte is considered active.
     */
    constexpr static float modulation_threshold_ = 0.F;

    /**
     * @brief Default number of monitored spikes required to activate the calcium rise.
     */
    constexpr static uint64_t spike_threshold_ = 14;

    /**
     * @brief Default base synapse weight, reserved for future absolute-modulation use.
     */
    constexpr static float base_weight_ = 0.07F;

    /**
     * @brief Default multiplier applied to a controlled synapse's own weight while the astrocyte is active.
     */
    constexpr static float active_weight_multiplier_ = 1000.F;

    /**
     * @brief Default lower clamp for the resulting synapse weight.
     */
    constexpr static float min_weight_ = 0.001F;

    /**
     * @brief Default upper clamp for the resulting synapse weight, reserved for future use.
     */
    constexpr static float max_weight_ = 2.F;
};


/**
 * @brief Structure for astrocyte parameters.
 */
template <>
struct neuron_parameters<AstrocyteCell>
{
    /**
     * @brief Current calcium level.
     */
    float calcium_ = default_values<AstrocyteCell>::calcium_;

    /**
     * @brief `true` while calcium is rising towards `calcium_max_`, `false` while it decays back towards 0.
     */
    bool rising_ = default_values<AstrocyteCell>::rising_;

    /**
     * @brief Number of monitored spikes accumulated since calcium was last at 0.
     */
    uint64_t spike_counter_ = default_values<AstrocyteCell>::spike_counter_;

    /**
     * @brief Total number of monitored spikes observed over the whole simulation.
     */
    uint64_t total_spikes_ = default_values<AstrocyteCell>::total_spikes_;

    /**
     * @brief Number of times the calcium rise has been activated.
     */
    uint64_t activation_count_ = default_values<AstrocyteCell>::activation_count_;

    /**
     * @brief Number of monitored spikes received so far during the current step.
     */
    uint64_t monitored_spikes_this_step_ = default_values<AstrocyteCell>::monitored_spikes_this_step_;

    /**
     * @brief Calcium level at which calcium stops rising and starts decaying.
     */
    float calcium_max_ = default_values<AstrocyteCell>::calcium_max_;

    /**
     * @brief Amount by which calcium rises or decays on every active step.
     */
    float calcium_step_ = default_values<AstrocyteCell>::calcium_step_;

    /**
     * @brief Calcium level above which the astrocyte is considered active and modulates synapse weights.
     */
    float modulation_threshold_ = default_values<AstrocyteCell>::modulation_threshold_;

    /**
     * @brief Number of monitored spikes required to activate the calcium rise.
     */
    uint64_t spike_threshold_ = default_values<AstrocyteCell>::spike_threshold_;

    /**
     * @brief Base synapse weight, reserved for future absolute-modulation use.
     */
    float base_weight_ = default_values<AstrocyteCell>::base_weight_;

    /**
     * @brief Multiplier applied to a controlled synapse's own weight while the astrocyte is active.
     */
    float active_weight_multiplier_ = default_values<AstrocyteCell>::active_weight_multiplier_;

    /**
     * @brief Lower clamp for the resulting synapse weight.
     */
    float min_weight_ = default_values<AstrocyteCell>::min_weight_;

    /**
     * @brief Upper clamp for the resulting synapse weight, reserved for future use.
     */
    float max_weight_ = default_values<AstrocyteCell>::max_weight_;
};

}  // namespace knp::neuron_traits
