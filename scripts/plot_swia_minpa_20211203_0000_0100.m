function outputs = plot_swia_minpa_20211203_0000_0100(inputDir, outputDir)
%PLOT_SWIA_MINPA_20211203_0000_0100 Plot local MAVEN SWIA and TW-1 MINPA NV_MSO.
%
% Reads local comparison CSV products for 2021-12-03 00:00-01:00 UTC and
% writes a focused five-panel PNG/PDF/JSON under the project outputs folder.

if nargin < 1 || strlength(string(inputDir)) == 0
    inputDir = 'D:\Data\TW-1\result_codex\comparisons\maven_swia_tw1_minpa_20211203_0000_0100';
end
if nargin < 2 || strlength(string(outputDir)) == 0
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(repoRoot, 'outputs', 'swia_minpa_20211203_0000_0100');
end

inputDir = char(inputDir);
outputDir = char(outputDir);
swiaCsv = fullfile(inputDir, 'maven_swia_hplus_20211203_0000_0100.csv');
minpaCsv = fullfile(inputDir, 'tw1_minpa_hplus_mso_20211203_0000_0100.csv');
summaryJson = fullfile(inputDir, 'swia_minpa_hplus_summary_20211203_0000_0100.json');
assert(isfile(swiaCsv), 'Missing MAVEN SWIA CSV: %s', swiaCsv);
assert(isfile(minpaCsv), 'Missing TW-1 MINPA CSV: %s', minpaCsv);

swia = readtable(swiaCsv, TextType='string');
minpa = readtable(minpaCsv, TextType='string');
swiaTime = datetime(swia.time_utc, InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', TimeZone='UTC');
minpaTime = datetime(minpa.time_utc, InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', TimeZone='UTC');

if ~isfolder(outputDir)
    mkdir(outputDir);
end

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 1050]);
layout = tiledlayout(fig, 5, 1, TileSpacing='compact', Padding='compact');

plotPanel(nexttile(layout), swiaTime, swia.density_cm3, minpaTime, minpa.density_cm3, ...
    'Density (cm^{-3})', 'H^+ density');
plotPanel(nexttile(layout), swiaTime, swia.speed_kms, minpaTime, minpa.speed_kms, ...
    '|V| (km s^{-1})', 'MSO speed');
plotPanel(nexttile(layout), swiaTime, swia.vx_mso_kms, minpaTime, minpa.vx_mso_kms, ...
    'V_X (km s^{-1})', 'MSO V_X');
plotPanel(nexttile(layout), swiaTime, swia.vy_mso_kms, minpaTime, minpa.vy_mso_kms, ...
    'V_Y (km s^{-1})', 'MSO V_Y');
plotPanel(nexttile(layout), swiaTime, swia.vz_mso_kms, minpaTime, minpa.vz_mso_kms, ...
    'V_Z (km s^{-1})', 'MSO V_Z');

title(layout, 'MAVEN SWIA and Tianwen-1 MINPA/NV\_MSO, 2021-12-03 00:00-01:00 UTC', ...
    FontWeight='bold');
xlabel(layout, 'UTC');

pngPath = fullfile(outputDir, 'swia_minpa_density_velocity_mso_20211203_0000_0100.png');
pdfPath = fullfile(outputDir, 'swia_minpa_density_velocity_mso_20211203_0000_0100.pdf');
exportgraphics(fig, pngPath, Resolution=220);
exportgraphics(fig, pdfPath, ContentType='vector');
close(fig);

summary = struct;
summary.interval_utc = struct('start', '2021-12-03T00:00:00Z', 'stop', '2021-12-03T01:00:00Z');
summary.sources = struct( ...
    'maven_swia_csv', swiaCsv, ...
    'tw1_minpa_nv_mso_csv', minpaCsv, ...
    'previous_summary_json', summaryJson);
summary.coordinate_system = 'MSO';
summary.species = 'H+';
summary.maven_swia_records = height(swia);
summary.tw1_minpa_records = height(minpa);
summary.outputs = struct('png', pngPath, 'pdf', pdfPath);
summary.notes = [
    "MAVEN SWIA CSV was derived from local L2 onboardsvymom density and velocity_mso."
    "TW-1 MINPA CSV was derived from local NV_MSO NH_TW1 and VH_TW1_MSO."
    "The spacecraft are not co-located; this is a simultaneous time-series comparison."
];

outJson = fullfile(outputDir, 'swia_minpa_density_velocity_mso_20211203_0000_0100_summary.json');
fid = fopen(outJson, 'w');
assert(fid >= 0, 'Could not write %s', outJson);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(summary, PrettyPrint=true));
clear cleanup;

outputs = summary.outputs;
fprintf('Wrote %s\n', pngPath);
fprintf('Wrote %s\n', pdfPath);
end


function plotPanel(ax, swiaTime, swiaValue, minpaTime, minpaValue, yLabelText, panelTitle)
plot(ax, swiaTime, swiaValue, '-', Color=[0.121, 0.466, 0.705], LineWidth=0.85);
hold(ax, 'on');
plot(ax, minpaTime, minpaValue, '--', Color=[0.839, 0.153, 0.157], LineWidth=0.95);
grid(ax, 'on');
box(ax, 'on');
ylabel(ax, yLabelText);
title(ax, panelTitle, FontWeight='normal', HorizontalAlignment='left');
legend(ax, {'MAVEN SWIA', 'TW-1 MINPA NV\_MSO'}, Location='best');
ax.XLim = [datetime(2021,12,3,0,0,0,TimeZone='UTC'), datetime(2021,12,3,1,0,0,TimeZone='UTC')];
ax.XAxis.TickLabelFormat = 'HH:mm';
end
