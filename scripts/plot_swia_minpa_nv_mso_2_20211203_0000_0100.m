function outputs = plot_swia_minpa_nv_mso_2_20211203_0000_0100(outputDir)
%PLOT_SWIA_MINPA_NV_MSO_2_20211203_0000_0100 Compare SWIA with MINPA NV_MSO_2.
%
% Plots MAVEN SWIA H+ density/velocity and Tianwen-1 MINPA NV_MSO_2 H+
% density/velocity for 2021-12-03 00:00-01:00 UTC.

if nargin < 1 || strlength(string(outputDir)) == 0
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(repoRoot, 'outputs', 'swia_minpa_nv_mso_2_20211203_0000_0100');
end
outputDir = char(outputDir);
if ~isfolder(outputDir)
    mkdir(outputDir);
end

swiaCsv = 'D:\Data\TW-1\result_codex\comparisons\maven_swia_tw1_minpa_20211203_0000_0100\maven_swia_hplus_20211203_0000_0100.csv';
minpaMat = 'D:\Data\TW-1\result\MINPA\NV_MSO_2\NV_20211203.mat';
oldMinpaMat = 'D:\Data\TW-1\result\MINPA\NV_MSO\NV_20211203.mat';
assert(isfile(swiaCsv), 'Missing MAVEN SWIA CSV: %s', swiaCsv);
assert(isfile(minpaMat), 'Missing TW-1 MINPA NV_MSO_2 MAT: %s', minpaMat);

tStart = datetime(2021, 12, 3, 0, 0, 0, TimeZone='UTC');
tStop = datetime(2021, 12, 3, 1, 0, 0, TimeZone='UTC');

swia = readtable(swiaCsv, TextType='string');
swiaTime = datetime(swia.time_utc, InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', TimeZone='UTC');
swiaMask = swiaTime >= tStart & swiaTime <= tStop;
swiaTime = swiaTime(swiaMask);
swiaDensity = swia.density_cm3(swiaMask);
swiaVx = swia.vx_mso_kms(swiaMask);
swiaVy = swia.vy_mso_kms(swiaMask);
swiaVz = swia.vz_mso_kms(swiaMask);
swiaSpeed = swia.speed_kms(swiaMask);

minpa = load(minpaMat, 'NH_TW1', 'VH_TW1_MSO', 'mode_TW1', ...
    'density_scale_TW1', 'energy_width_over_E_TW1', 'Xb_MSO');
minpaTime = datetime(minpa.NH_TW1(:, 1), ConvertFrom='posixtime', TimeZone='UTC');
minpaMask = minpaTime >= tStart & minpaTime <= tStop;
minpaTime = minpaTime(minpaMask);
minpaDensity = minpa.NH_TW1(minpaMask, 2);
minpaVx = minpa.VH_TW1_MSO(minpaMask, 2);
minpaVy = minpa.VH_TW1_MSO(minpaMask, 3);
minpaVz = minpa.VH_TW1_MSO(minpaMask, 4);
minpaSpeed = sqrt(minpaVx.^2 + minpaVy.^2 + minpaVz.^2);
minpaMode = minpa.mode_TW1(minpaMask, 2);
minpaScale = minpa.density_scale_TW1(minpaMask, 2);
minpaWidth = minpa.energy_width_over_E_TW1(minpaMask, 2);

minpaTable = table(string(datetime(minpaTime, Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''')), ...
    posixtime(minpaTime), minpaDensity, minpaVx, minpaVy, minpaVz, minpaSpeed, ...
    minpaMode, minpaScale, minpaWidth, ...
    'VariableNames', {'time_utc', 'time_unix_s', 'density_cm3', 'vx_mso_kms', ...
    'vy_mso_kms', 'vz_mso_kms', 'speed_kms', 'mode', 'density_scale', 'dE_over_E'});
minpaCsv = fullfile(outputDir, 'tw1_minpa_nv_mso_2_hplus_mso_20211203_0000_0100.csv');
writetable(minpaTable, minpaCsv);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1250, 1050]);
layout = tiledlayout(fig, 5, 1, TileSpacing='compact', Padding='compact');

plotPanel(nexttile(layout), swiaTime, swiaDensity, minpaTime, minpaDensity, ...
    'Density (cm^{-3})', 'H^+ density');
plotPanel(nexttile(layout), swiaTime, swiaSpeed, minpaTime, minpaSpeed, ...
    '|V| (km s^{-1})', 'MSO speed');
plotPanel(nexttile(layout), swiaTime, swiaVx, minpaTime, minpaVx, ...
    'V_X (km s^{-1})', 'MSO V_X');
plotPanel(nexttile(layout), swiaTime, swiaVy, minpaTime, minpaVy, ...
    'V_Y (km s^{-1})', 'MSO V_Y');
plotPanel(nexttile(layout), swiaTime, swiaVz, minpaTime, minpaVz, ...
    'V_Z (km s^{-1})', 'MSO V_Z');

title(layout, 'MAVEN SWIA and Tianwen-1 MINPA/NV\_MSO\_2, 2021-12-03 00:00-01:00 UTC', ...
    FontWeight='bold');
xlabel(layout, 'UTC');

pngPath = fullfile(outputDir, 'swia_minpa_nv_mso_2_density_velocity_mso_20211203_0000_0100.png');
pdfPath = fullfile(outputDir, 'swia_minpa_nv_mso_2_density_velocity_mso_20211203_0000_0100.pdf');
exportgraphics(fig, pngPath, Resolution=220);
exportgraphics(fig, pdfPath, ContentType='vector');
close(fig);

summary = struct();
summary.interval_utc = struct('start', '2021-12-03T00:00:00Z', 'stop', '2021-12-03T01:00:00Z');
summary.sources = struct('maven_swia_csv', swiaCsv, 'tw1_minpa_nv_mso_2_mat', minpaMat, ...
    'tw1_minpa_legacy_nv_mso_mat', oldMinpaMat);
summary.coordinate_system = 'MSO';
summary.species = 'H+';
summary.maven_swia_records = numel(swiaDensity);
summary.tw1_minpa_records = numel(minpaDensity);
summary.minpa_modes_in_interval = unique(minpaMode(isfinite(minpaMode)))';
summary.minpa_density_scale_median = median(minpaScale, 'omitnan');
summary.minpa_dE_over_E_median = median(minpaWidth, 'omitnan');
summary.statistics = struct();
summary.statistics.swia_density_median_cm3 = median(swiaDensity, 'omitnan');
summary.statistics.minpa_density_median_cm3 = median(minpaDensity, 'omitnan');
summary.statistics.minpa_over_swia_density_median_ratio = ...
    summary.statistics.minpa_density_median_cm3 / summary.statistics.swia_density_median_cm3;
summary.statistics.swia_speed_median_kms = median(swiaSpeed, 'omitnan');
summary.statistics.minpa_speed_median_kms = median(minpaSpeed, 'omitnan');
summary.statistics.speed_median_diff_minpa_minus_swia_kms = ...
    summary.statistics.minpa_speed_median_kms - summary.statistics.swia_speed_median_kms;
summary.outputs = struct('png', pngPath, 'pdf', pdfPath, 'minpa_csv', minpaCsv);
summary.notes = [
    "MAVEN SWIA CSV was derived from local L2 onboardsvymom density and velocity_mso."
    "TW-1 MINPA is read directly from NV_MSO_2 NH_TW1 and VH_TW1_MSO."
    "NV_MSO_2 density uses mode-aware adjacent-center energy integration width; velocity is inherited from legacy NV_MSO."
    "The spacecraft are not co-located; this is a simultaneous time-series comparison."
];

outJson = fullfile(outputDir, 'swia_minpa_nv_mso_2_density_velocity_mso_20211203_0000_0100_summary.json');
fid = fopen(outJson, 'w');
assert(fid >= 0, 'Could not write %s', outJson);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(summary, PrettyPrint=true));
clear cleanup;

outputs = summary.outputs;
fprintf('Wrote %s\n', pngPath);
fprintf('Wrote %s\n', pdfPath);
fprintf('Wrote %s\n', outJson);
fprintf('Wrote %s\n', minpaCsv);
fprintf('Median density SWIA=%.6g MINPA=%.6g ratio=%.6g\n', ...
    summary.statistics.swia_density_median_cm3, ...
    summary.statistics.minpa_density_median_cm3, ...
    summary.statistics.minpa_over_swia_density_median_ratio);
end

function plotPanel(ax, swiaTime, swiaValue, minpaTime, minpaValue, yLabelText, panelTitle)
plot(ax, swiaTime, swiaValue, '-', Color=[0.121, 0.466, 0.705], LineWidth=0.85);
hold(ax, 'on');
plot(ax, minpaTime, minpaValue, '--', Color=[0.839, 0.153, 0.157], LineWidth=1.0);
grid(ax, 'on');
box(ax, 'on');
ylabel(ax, yLabelText);
title(ax, panelTitle, FontWeight='normal', HorizontalAlignment='left');
legend(ax, {'MAVEN SWIA', 'TW-1 MINPA NV\_MSO\_2'}, Location='best');
ax.XLim = [datetime(2021,12,3,0,0,0,TimeZone='UTC'), datetime(2021,12,3,1,0,0,TimeZone='UTC')];
ax.XAxis.TickLabelFormat = 'HH:mm';
end
