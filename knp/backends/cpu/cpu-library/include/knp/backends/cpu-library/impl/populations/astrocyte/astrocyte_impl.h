#pragma once

#include <knp/core/population.h>
#include <knp/neuron-traits/astrocyte.h>


namespace knp::backends::cpu::populations::impl::astrocyte
{

inline void calculate_pre_impact_single_neuron_state_impl(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> & /* neuron */)
{
}


inline void impact_neuron_impl(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> &neuron,
    const knp::core::messaging::SynapticImpact & /* impact */, bool /* is_forcing */)
{
    ++neuron.monitored_spikes_this_step_;
}


inline bool calculate_post_impact_single_neuron_state_impl(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::AstrocyteCell> &neuron)
{
    const uint64_t spikes_this_step = neuron.monitored_spikes_this_step_;
    neuron.monitored_spikes_this_step_ = 0;

    if (neuron.calcium_ == 0.F)
    {
        neuron.spike_counter_ += spikes_this_step;
        neuron.total_spikes_ += spikes_this_step;

        if (neuron.spike_counter_ >= neuron.spike_threshold_)
        {
            neuron.rising_ = true;
            neuron.calcium_ = neuron.calcium_step_;
            neuron.spike_counter_ = 0;
            ++neuron.activation_count_;
        }
    }
    else if (neuron.rising_)
    {
        neuron.calcium_ += neuron.calcium_step_;
        if (neuron.calcium_ >= neuron.calcium_max_)
        {
            neuron.calcium_ = neuron.calcium_max_;
            neuron.rising_ = false;
        }
    }
    else
    {
        neuron.calcium_ -= neuron.calcium_step_;
        if (neuron.calcium_ < 0.F)
        {
            neuron.calcium_ = 0.F;
            neuron.rising_ = false;
        }
    }

    return neuron.calcium_ > neuron.modulation_threshold_;
}

}  // namespace knp::backends::cpu::populations::impl::astrocyte
