#pragma once

#include "astrocyte_impl.h"


namespace knp::backends::cpu::populations::impl
{

/**
 * @brief Calculate pre impact state of single neuron.
 * @param neuron Neuron.
 */
inline void calculate_pre_impact_single_neuron_state_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> &neuron)
{
    astrocyte::calculate_pre_impact_single_neuron_state_impl(neuron);
}


/**
 * @brief Impact neuron.
 * @param neuron Neuron.
 * @param impact Impact message.
 * @param is_forcing Is impact forced.
 */
inline void impact_neuron_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> &neuron,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    astrocyte::impact_neuron_impl(neuron, impact, is_forcing);
}


/**
 * @brief Calculate post impact state of single neuron.
 * @param neuron Neuron.
 * @return Should neuron produce spike or should not.
 */
inline bool calculate_post_impact_single_neuron_state_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> &neuron)
{
    return astrocyte::calculate_post_impact_single_neuron_state_impl(neuron);
}

}  // namespace knp::backends::cpu::populations::impl
