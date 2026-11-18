%> @file plot_results.m
%> @brief Visualize spike activity, astrocyte calcium, and synaptic weights.
%>
%> @details Reads stimulation_multi_astro_output.csv and plots spike counts, astrocyte
%> calcium, weight multipliers, activation counts, and individual synapse weights.
%> Each figure has a Save figure button for PNG, PDF, or MATLAB FIG output.
%>
%> @note Set csv_file to the path of the simulation results.
%> Controlled weights are scaled relative to their initial values. Within each
%> logged projection, source indices vary first, as in the KNP all-to-all connector.
%>
%> @author Yuliya Tsybina
%> @date 2026
%> @license Apache 2.0
%> @copyright © 2026 Lobachevsky University
%>
%> Licensed under the Apache License, Version 2.0 (the "License");
%> you may not use this file except in compliance with the License.
%> You may obtain a copy of the License at
%>
%>     http://www.apache.org/licenses/LICENSE-2.0
%>
%> Unless required by applicable law or agreed to in writing, software
%> distributed under the License is distributed on an "AS IS" BASIS,
%> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%> See the License for the specific language governing permissions and
%> limitations under the License.

clear; clc;

csv_file = 'stimulation_multi_astro_output.csv';
data = load_results(csv_file);
dynamics_figure = plot_network_dynamics(data);
synapse_figure = plot_synapse_weights(data);

%> @brief Read simulation results from a CSV file.
%> @param csv_file Path to the CSV written by neuron_astrocyte_network.
%> @return Table containing simulation results.
function data = load_results(csv_file)
    if ~isfile(csv_file)
        error('File "%s" not found. Set the correct path in the csv_file variable.', csv_file);
    end
    data = readtable(csv_file);
end


%> @brief Plot spike counts and the state of the three astrocytes.
%> @param data Table containing simulation results.
%> @return Figure containing four plots of network dynamics.
function fig = plot_network_dynamics(data)
    fig = figure('Name', 'neuron astrocyte network: network dynamics with astrocytes', 'Color', 'w', ...
                 'Position', [80, 60, 1000, 700]);

    ax = subplot(2, 2, 1, 'Parent', fig);
    columns = {'PreSpikeCount', 'Post1GroupASpikeCount', 'Post1GroupBSpikeCount', 'Post2SpikeCount'};
    labels = {'Pre (3 neurons)', 'Post1 group A (2 neurons)', ...
              'Post1 group B (2 neurons)', 'Post2 (5 neurons)'};
    plot_series(ax, data, columns, labels, @plot);
    format_axes(ax, 'Spike count per step', 'Population spike activity');

    ax = subplot(2, 2, 2, 'Parent', fig);
    columns = {'Astro1_Ca', 'Astro2_Ca', 'Astro3_Ca'};
    labels = {'Astro 1 (Simple)', 'Astro 2 (Simple)', 'Astro 3 (Nonlinear)'};
    plot_series(ax, data, columns, labels, @plot);
    format_axes(ax, 'Calcium (Ca)', 'Astrocyte calcium level');

    ax = subplot(2, 2, 3, 'Parent', fig);
    columns = {'Astro1_Multiplier', 'Astro2_Multiplier', 'Astro3_Multiplier'};
    labels = {'Astro 1 (Simple)', 'Astro 2 (Simple)', ...
              'Astro 3 (Nonlinear)'};
    plot_series(ax, data, columns, labels, @plot);
    format_axes(ax, 'Weight multiplier (active\_weight\_multiplier\_)', 'Astrocyte modulation strength');

    ax = subplot(2, 2, 4, 'Parent', fig);
    columns = {'Astro1_Activations', 'Astro2_Activations', 'Astro3_Activations'};
    labels = {'Astro 1 (Simple)', 'Astro 2 (Simple)', 'Astro 3 (Nonlinear)'};
    plot_series(ax, data, columns, labels, @stairs);
    format_axes(ax, 'Activation count (cumulative)', 'Astrocyte activations');

    sgtitle(fig, 'neuron astrocyte network: 3 populations + 2 simple and 1 nonlinear astrocytes');
    add_save_button(fig, 'neuron_astrocyte_network_dynamics.png');
end


%> @brief Plot individual weights in modulated and unmodulated projections.
%> @param data Table containing simulation results.
%> @return Figure containing two plots of synaptic weights.
function fig = plot_synapse_weights(data)
    fig = figure('Name', 'neuron astrocyte network: individual synapse weights', 'Color', 'w', ...
                 'Position', [80, 60, 1000, 700]);

    ax = subplot(1, 2, 1, 'Parent', fig);
    columns = {'PreToPost1A_S0', 'PreToPost1A_S1', 'PreToPost1A_S2', ...
               'PreToPost1A_S3', 'PreToPost1A_S4', 'PreToPost1A_S5'};
    labels = {'pre0->A0', 'pre1->A0', 'pre2->A0', 'pre0->A1', 'pre1->A1', 'pre2->A1'};
    plot_series(ax, data, columns, labels, @plot);
    format_axes(ax, 'Synapse weight', 'pre\_to\_post1\_a (modulated by astrocyte 1)');

    ax = subplot(1, 2, 2, 'Parent', fig);
    columns = {'Post1LoopB_S0', 'Post1LoopB_S1', 'Post1LoopB_S2', 'Post1LoopB_S3'};
    labels = {'post1(2)->post1(2)', 'post1(3)->post1(2)', ...
              'post1(2)->post1(3)', 'post1(3)->post1(3)'};
    plot_series(ax, data, columns, labels, @plot);
    format_axes(ax, 'Synapse weight', 'post1\_loop\_b (not modulated, weight fixed)');

    sgtitle(fig, 'neuron astrocyte network: random individual synapse weights -- modulated vs fixed projection');
    add_save_button(fig, 'neuron_astrocyte_network_synapse_weights.png');
end


%> @brief Draw named CSV columns against simulation steps.
%> @param ax Axes receiving the plotted series.
%> @param data Table containing simulation results.
%> @param columns Cell array of CSV column names.
%> @param labels Cell array of legend labels in the same order as columns.
%> @param plot_function Plotting function handle, such as plot or stairs.
function plot_series(ax, data, columns, labels, plot_function)
    hold(ax, 'on');
    for k = 1:numel(columns)
        plot_function(ax, data.Step, data.(columns{k}), 'DisplayName', labels{k});
    end
    hold(ax, 'off');
end


%> @brief Apply axis labels, a title, a legend, and a grid to a plot.
%> @param ax Axes to format.
%> @param y_label Label for the vertical axis.
%> @param plot_title Title of the plot.
function format_axes(ax, y_label, plot_title)
    xlabel(ax, 'Simulation step');
    ylabel(ax, y_label);
    title(ax, plot_title);
    legend(ax, 'Location', 'best');
    grid(ax, 'on');
end


%> @brief Reserve space below the plots and add a figure saving button.
%> @param fig Figure receiving the button.
%> @param default_name Suggested output filename.
function add_save_button(fig, default_name)
    axes_list = findall(fig, 'Type', 'axes');
    for k = 1:numel(axes_list)
        ax = axes_list(k);
        ax.Units = 'normalized';
        position = ax.OuterPosition;
        position(2) = 0.08 + 0.84 * position(2);
        position(4) = 0.84 * position(4);
        ax.OuterPosition = position;
    end

    uicontrol(fig, 'Style', 'pushbutton', 'String', 'Save figure...', ...
              'Units', 'normalized', 'Position', [0.75, 0.01, 0.23, 0.05], ...
              'Callback', @(button, ~) save_figure(fig, button, default_name));
end


%> @brief Ask for an output path and save the selected figure.
%> @param fig Figure to save.
%> @param button Save button, hidden during PNG and PDF export.
%> @param default_name Suggested output filename.
function save_figure(fig, button, default_name)
    filters = {'*.png', 'PNG image (*.png)'; ...
               '*.pdf', 'PDF document (*.pdf)'; ...
               '*.fig', 'MATLAB figure (*.fig)'};
    [filename, folder, filter_index] = uiputfile(filters, 'Save figure', default_name);
    if isequal(filename, 0)
        return;
    end

    [~, ~, extension] = fileparts(filename);
    if isempty(extension)
        extensions = {'.png', '.pdf', '.fig'};
        extension = extensions{filter_index};
        filename = [filename, extension];
    end
    output_path = fullfile(folder, filename);

    try
        if strcmpi(extension, '.fig')
            savefig(fig, output_path);
        else
            previous_visibility = button.Visible;
            restore_button = onCleanup(@() set(button, 'Visible', previous_visibility));
            button.Visible = 'off';
            drawnow;
            if strcmpi(extension, '.png')
                print(fig, output_path, '-dpng', '-r150', '-noui');
            elseif strcmpi(extension, '.pdf')
                print(fig, output_path, '-dpdf', '-bestfit', '-noui');
            else
                error('Choose a PNG, PDF, or FIG filename.');
            end
        end
    catch exception
        errordlg(sprintf('Could not save the figure:\n%s', exception.message), 'Save figure');
    end
end
