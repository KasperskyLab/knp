% plot_results.m
%
% Visualizes SNAN.exe results: reads the CSV written by run_simulation()
% (default "stimulation_multi_astro_output.csv") and plots all logged
% variables over time -- spike counts per population/group, calcium level,
% and the current weight multiplier (active_weight_multiplier_, x1 at rest,
% xN while active) for each of the three astrocytes, plus activation counts.
%
% Each synapse is scaled by its astrocyte relative to its OWN original
% (random) weight, so the CSV logs a multiplier rather than a single
% projection-wide weight -- different synapses within the same projection
% genuinely have different weights.
%
% This file must be in the same folder as the CSV (by default, the working
% folder SNAN.exe was run from, e.g. build\windows-ninja\bin), or the path
% to the CSV must be set explicitly in the csv_file variable below.
%
% Expected CSV columns (in exactly this order, as written by SNAN.cpp):
%   Step, PreSpikeCount, Post1GroupASpikeCount, Post1GroupBSpikeCount, Post2SpikeCount,
%   Astro1_Ca, Astro1_Multiplier, Astro1_Activations,
%   Astro2_Ca, Astro2_Multiplier, Astro2_Activations,
%   Astro3_Ca, Astro3_Multiplier, Astro3_Activations,
%   PreToPost1A_S0..S5 (6 individual synapses of the pre_to_post1_a projection, modulated by astrocyte 1, x1),
%   Post1LoopB_S0..S3 (4 individual synapses of the post1_loop_b projection, not modulated, weight fixed)

clear; clc;

% ------------------------------------------------------------------
% 1. Load data.
% ------------------------------------------------------------------
csv_file = 'stimulation_multi_astro_output.csv';

if ~isfile(csv_file)
    error('File "%s" not found. Set the correct path in the csv_file variable.', csv_file);
end

data = readtable(csv_file);

step = data.Step;

% ------------------------------------------------------------------
% 2. One figure, 4 plots: spikes, calcium, weight, activation count.
% ------------------------------------------------------------------
figure('Name', 'SNAN: network dynamics with astrocytes', 'Color', 'w');

% --- (1,1) Spike count per population/group per step. ---
subplot(2, 2, 1);
hold on;
plot(step, data.PreSpikeCount, 'DisplayName', 'Pre (3 neurons)');
plot(step, data.Post1GroupASpikeCount, 'DisplayName', 'Post1 group A (2 neurons)');
plot(step, data.Post1GroupBSpikeCount, 'DisplayName', 'Post1 group B (2 neurons)');
plot(step, data.Post2SpikeCount, 'DisplayName', 'Post2 (5 neurons)');
hold off;
xlabel('Simulation step');
ylabel('Spike count per step');
title('Population spike activity');
legend('Location', 'best');
grid on;

% --- (1,2) Calcium level of each astrocyte. ---
subplot(2, 2, 2);
hold on;
plot(step, data.Astro1_Ca, 'DisplayName', 'Astro 1');
plot(step, data.Astro2_Ca, 'DisplayName', 'Astro 2');
plot(step, data.Astro3_Ca, 'DisplayName', 'Astro 3');
hold off;
xlabel('Simulation step');
ylabel('Calcium (Ca)');
title('Astrocyte calcium level');
legend('Location', 'best');
grid on;

% --- (2,1) Current weight multiplier of each astrocyte (x1 at rest, xN while active). ---
subplot(2, 2, 3);
hold on;
plot(step, data.Astro1_Multiplier, 'DisplayName', 'Astro 1 (base multiplier x1)');
plot(step, data.Astro2_Multiplier, 'DisplayName', 'Astro 2 (base multiplier x2)');
plot(step, data.Astro3_Multiplier, 'DisplayName', 'Astro 3 (base multiplier x3)');
hold off;
xlabel('Simulation step');
ylabel('Weight multiplier (active\_weight\_multiplier\_)');
title('Astrocyte modulation strength');
legend('Location', 'best');
grid on;

% --- (2,2) Cumulative activation count (rising_ turned on) for each astrocyte. ---
subplot(2, 2, 4);
hold on;
stairs(step, data.Astro1_Activations, 'DisplayName', 'Astro 1');
stairs(step, data.Astro2_Activations, 'DisplayName', 'Astro 2');
stairs(step, data.Astro3_Activations, 'DisplayName', 'Astro 3');
hold off;
xlabel('Simulation step');
ylabel('Activation count (cumulative)');
title('Astrocyte activations');
legend('Location', 'best');
grid on;

sgtitle('SNAN: 3 populations + 3 independent astrocytes');

% ------------------------------------------------------------------
% 3. Second figure: individual synapse weights -- checks that the random
%    initial weights within one projection really differ and behave
%    differently depending on whether an astrocyte modulates them.
% ------------------------------------------------------------------
figure('Name', 'SNAN: individual synapse weights', 'Color', 'w');

% --- (1,1) 6 synapses of pre_to_post1_a -- modulated by astrocyte 1 (multiplier x1 while active): each
% synapse has its own random original weight, but they are all scaled by the same multiplier, so the lines
% diverge/converge in sync while keeping their relative order. ---
subplot(1, 2, 1);
hold on;
pre_to_post1_a_labels = {'pre0->A0', 'pre0->A1', 'pre1->A0', 'pre1->A1', 'pre2->A0', 'pre2->A1'};
pre_to_post1_a_cols = {'PreToPost1A_S0', 'PreToPost1A_S1', 'PreToPost1A_S2', 'PreToPost1A_S3', 'PreToPost1A_S4', 'PreToPost1A_S5'};
for k = 1:numel(pre_to_post1_a_cols)
    plot(step, data.(pre_to_post1_a_cols{k}), 'DisplayName', pre_to_post1_a_labels{k});
end
hold off;
xlabel('Simulation step');
ylabel('Synapse weight');
title('pre\_to\_post1\_a (modulated by astrocyte 1)');
legend('Location', 'best');
grid on;

% --- (1,2) 4 synapses of post1_loop_b -- not modulated by any astrocyte: weights stay horizontal
% straight lines for the whole simulation, which also shows they start out different. ---
subplot(1, 2, 2);
hold on;
post1_loop_b_labels = {'post1(2)->post1(2)', 'post1(2)->post1(3)', 'post1(3)->post1(2)', 'post1(3)->post1(3)'};
post1_loop_b_cols = {'Post1LoopB_S0', 'Post1LoopB_S1', 'Post1LoopB_S2', 'Post1LoopB_S3'};
for k = 1:numel(post1_loop_b_cols)
    plot(step, data.(post1_loop_b_cols{k}), 'DisplayName', post1_loop_b_labels{k});
end
hold off;
xlabel('Simulation step');
ylabel('Synapse weight');
title('post1\_loop\_b (not modulated, weight fixed)');
legend('Location', 'best');
grid on;

sgtitle('SNAN: random individual synapse weights -- modulated vs fixed projection');

% ------------------------------------------------------------------
% 4. Optional: save the figures to files next to the CSV.
% ------------------------------------------------------------------
% saveas(gcf, 'snan_synapse_weights.png');
% figure(1); saveas(gcf, 'snan_dynamics.png');
