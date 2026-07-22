function summary = plot_tw1_minpa_quality_flag_context_examples(examplesCsv, windowMinutes, outputDir)
%PLOT_TW1_MINPA_QUALITY_FLAG_CONTEXT_EXAMPLES Plot +/- windows around flag examples.
%
% For each row in the quality-flag example CSV, this script plots:
%   1) H+ 1D energy spectrum from day_spe, +/- windowMinutes
%   2) NV_MSO_2 H+ density time series
%   3) NV_MSO_2 H+ speed magnitude time series
%
% Raw ori products are not modified. This is a read-only plotting workflow.

if nargin < 1 || isempty(examplesCsv)
    examplesCsv = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
        'outputs', 'tw1_minpa_quality_flag_audit', ...
        'tw1_minpa_hplus_quality_flag_audit_examples.csv');
end
if nargin < 2 || isempty(windowMinutes)
    windowMinutes = 5;
end
if nargin < 3
    outputDir = '';
end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
paths.daySpeDir = 'D:\Data\TW-1\result\MINPA\day_spe';
paths.nvMso2Dir = 'D:\Data\TW-1\result\MINPA\NV_MSO_2';
if isempty(outputDir)
    paths.outputDir = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_audit', 'context_5min_project_spectrogram');
else
    paths.outputDir = outputDir;
end
if ~exist(paths.outputDir, 'dir')
    mkdir(paths.outputDir);
end

opts = detectImportOptions(examplesCsv);
opts = setvartype(opts, {'category', 'source_file', 'time_utc', 'quality_flag_binary'}, 'string');
samples = readtable(examplesCsv, opts);
flagRows = read_flag_rows_for_examples(examplesCsv);

rows = table();
for i = 1:height(samples)
    sample = samples(i, :);
    try
        out = plot_one_context(sample, paths, windowMinutes, flagRows);
        rows = [rows; out]; %#ok<AGROW>
    catch ME
        warning('Could not plot row %d %s: %s', i, sample.time_utc, ME.message);
    end
end

indexCsv = fullfile(paths.outputDir, 'tw1_minpa_quality_flag_context_5min_index.csv');
if ~isempty(rows)
    writetable(rows, indexCsv);
end

summary = struct();
summary.examples_csv = examplesCsv;
summary.output_dir = paths.outputDir;
summary.index_csv = indexCsv;
summary.window_minutes = windowMinutes;
summary.figures = rows;
fprintf('Wrote %d context figures under %s\n', height(rows), paths.outputDir);
end

function out = plot_one_context(sample, paths, windowMinutes, flagRows)
t0 = double(sample.time_unix_s);
tStart = t0 - windowMinutes * 60;
tStop = t0 + windowMinutes * 60;
days = days_for_window(tStart, tStop);

[specSegments, specTimeCount, specMode] = load_h_spectrum_window(paths.daySpeDir, days, tStart, tStop);
[momT, density, speed] = load_nv_mso2_window(paths.nvMso2Dir, days, tStart, tStop);

centerNum = epoch_to_datenum(t0);
startNum = epoch_to_datenum(tStart);
stopNum = epoch_to_datenum(tStop);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1200, 980]);
layout = tiledlayout(fig, 4, 1, TileSpacing='compact', Padding='compact');

ax1 = nexttile(layout, 1);
if isempty(specSegments)
    text(ax1, 0.5, 0.5, 'No H+ day\_spe data in window', HorizontalAlignment='center');
else
    plot_h_spectrum_segments(ax1, specSegments);
    cb = colorbar(ax1);
    cb.Label.String = 'log_{10}(H^+ day\_spe p)';
end
grid(ax1, 'on');
ylabel(ax1, 'Energy (eV)');
title(ax1, 'H^+ day\_spe spectrogram');
xlim(ax1, [startNum, stopNum]);
hold(ax1, 'on');
xline(ax1, centerNum, 'r-', 'LineWidth', 1.2);
if ~isempty(specMode)
    modeText = sprintf('modes: %s', strjoin(cellstr(unique(string(specMode(isfinite(specMode))))), ','));
    text(ax1, 0.01, 0.95, modeText, Units='normalized', VerticalAlignment='top', BackgroundColor='w');
end
datetick(ax1, 'x', 'HH:MM', 'keeplimits');

ax2 = nexttile(layout, 2);
plot_flag_panel(ax2, flagRows, tStart, tStop, centerNum, startNum, stopNum);

ax3 = nexttile(layout, 3);
plot_time_series(ax3, momT, density, centerNum, startNum, stopNum, 'Density (cm^{-3})', 'H^+ density from NV\_MSO\_2', true);

ax4 = nexttile(layout, 4);
plot_time_series(ax4, momT, speed, centerNum, startNum, stopNum, 'Speed (km/s)', 'H^+ speed magnitude from NV\_MSO\_2', false);
xlabel(ax4, 'UTC');

linkaxes([ax1, ax2, ax3, ax4], 'x');
format_flag_y_axis(ax2);

title(layout, sprintf('%s | UTC %s | flag=%d bits=%s | valid2D=%d', ...
    sample.category, sample.time_utc, sample.quality_flag, sample.quality_flag_binary, ...
    sample.valid_2d_channel_count), FontWeight='bold', Interpreter='none');

safeCat = regexprep(char(sample.category), '[^\w]+', '_');
timeTag = datestr(centerNum, 'yyyymmdd_HHMMSSFFF');
baseName = sprintf('%02d_%s_%s_flag%d_context_pm%dmin', ...
    sample.record_index, safeCat, timeTag, sample.quality_flag, windowMinutes);
png = fullfile(paths.outputDir, [baseName, '.png']);
pdf = fullfile(paths.outputDir, [baseName, '.pdf']);
format_flag_y_axis(ax2);
exportgraphics(fig, png, Resolution=180);
exportgraphics(fig, pdf, ContentType='vector');
close(fig);

out = table(sample.category, sample.time_utc, sample.quality_flag, sample.quality_flag_binary, ...
    sample.valid_2d_channel_count, string(png), string(pdf), specTimeCount, numel(unique(momT)), ...
    'VariableNames', {'category', 'time_utc', 'quality_flag', 'quality_flag_binary', ...
    'valid_2d_channel_count', 'png', 'pdf', 'spectrum_time_count', 'moment_time_count'});
end

function flagRows = read_flag_rows_for_examples(examplesCsv)
[folder, name, ~] = fileparts(examplesCsv);
candidateCsv = strings(0, 1);
if endsWith(string(name), "_examples")
    candidateCsv(end + 1) = fullfile(folder, [regexprep(name, '_examples$', '_records'), '.csv']);
end
candidateCsv(end + 1) = fullfile(folder, [strrep(name, 'examples', 'records'), '.csv']);
candidateCsv(end + 1) = fullfile(folder, 'tw1_minpa_hplus_quality_flag_audit_records.csv');

auditCsv = "";
for i = 1:numel(candidateCsv)
    if isfile(candidateCsv(i))
        auditCsv = candidateCsv(i);
        break
    end
end

if strlength(auditCsv) == 0
    flagRows = table();
    return
end
opts = detectImportOptions(auditCsv);
opts = setvartype(opts, {'source_file', 'time_utc', 'quality_flag_binary'}, 'string');
flagRows = readtable(auditCsv, opts);
end

function plot_flag_panel(ax, flagRows, tStart, tStop, centerNum, startNum, stopNum)
if isempty(flagRows)
    text(ax, 0.5, 0.5, 'No audit flag table available', HorizontalAlignment='center');
    xlim(ax, [startNum, stopNum]);
    xline(ax, centerNum, 'r-', 'LineWidth', 1.2);
    format_flag_y_axis(ax);
    return
end
mask = flagRows.time_unix_s >= tStart & flagRows.time_unix_s <= tStop;
rows = flagRows(mask, :);
if isempty(rows)
    text(ax, 0.5, 0.5, 'No flag records in window', HorizontalAlignment='center');
    xlim(ax, [startNum, stopNum]);
    xline(ax, centerNum, 'r-', 'LineWidth', 1.2);
    format_flag_y_axis(ax);
    return
end
t = epoch_to_datenum(double(rows.time_unix_s(:)));
bits = [rows.flag_bit1_sparse_caution, rows.flag_bit2_sparse_invalid, ...
    rows.flag_bit3_uv_contamination, rows.flag_bit4_evenodd_error, ...
    rows.flag_bit5_high_channel];
hold(ax, 'on');
for b = 1:5
    y0 = b - 0.35;
    y1 = b + 0.35;
    on = logical(bits(:, b));
    for i = 1:numel(t)
        if ~on(i)
            continue
        end
        if numel(t) == 1
            halfWidth = 4.1 / 86400;
        elseif i == 1
            halfWidth = (t(2) - t(1)) / 2;
        elseif i == numel(t)
            halfWidth = (t(end) - t(end - 1)) / 2;
        else
            halfWidth = (t(i + 1) - t(i - 1)) / 4;
        end
        patch(ax, [t(i)-halfWidth, t(i)+halfWidth, t(i)+halfWidth, t(i)-halfWidth], ...
            [y0, y0, y1, y1], [0.85 0.15 0.12], EdgeColor='none', FaceAlpha=0.9);
    end
end
set(ax, 'FontSize', 9);
set(ax, 'YDir', 'reverse');
xlim(ax, [startNum, stopNum]);
grid(ax, 'on');
title(ax, 'Quality flag bits: 1=5-10, 2=<5, 3=UV, 4=odd/even, 5=>1e10');
xline(ax, centerNum, 'r-', 'LineWidth', 1.2);
datetick(ax, 'x', 'HH:MM', 'keeplimits');
format_flag_y_axis(ax);
end

function format_flag_y_axis(ax)
set(ax, 'YLim', [0, 6], 'YLimMode', 'manual', ...
    'YTick', 0:6, 'YTickMode', 'manual', ...
    'YTickLabel', {'0', '1', '2', '3', '4', '5', '6'}, ...
    'YTickLabelMode', 'manual');
ylabel(ax, 'Bit');
set(ax, 'YDir', 'reverse');
end

function plot_time_series(ax, t, y, centerNum, startNum, stopNum, ylab, ttl, useLog)
if isempty(t)
    text(ax, 0.5, 0.5, 'No NV\_MSO\_2 data in window', HorizontalAlignment='center');
else
    dt = epoch_to_datenum(t);
    pos = y(isfinite(y) & y > 0);
    if useLog && ~isempty(pos)
        semilogy(ax, dt, max(y, min(pos)), '-o', MarkerSize=3, LineWidth=0.9);
    else
        plot(ax, dt, y, '-o', MarkerSize=3, LineWidth=0.9);
    end
end
grid(ax, 'on');
ylabel(ax, ylab);
title(ax, ttl);
xlim(ax, [startNum, stopNum]);
hold(ax, 'on');
xline(ax, centerNum, 'r-', 'LineWidth', 1.2);
datetick(ax, 'x', 'HH:MM', 'keeplimits');
end

function outDays = days_for_window(tStart, tStop)
d0 = datetime(tStart, ConvertFrom='posixtime', TimeZone='UTC');
d1 = datetime(tStop, ConvertFrom='posixtime', TimeZone='UTC');
d0 = dateshift(d0, 'start', 'day');
d1 = dateshift(d1, 'start', 'day');
n = round(datenum(d1) - datenum(d0));
outDays = strings(n + 1, 1);
for k = 0:n
    outDays(k + 1) = string(datestr(d0 + caldays(k), 'yyyymmdd'));
end
end

function [segments, timeCount, allMode] = load_h_spectrum_window(daySpeDir, days, tStart, tStop)
segments = struct('t', {}, 'energy', {}, 'p', {}, 'mode', {}, 'pLabel', {});
timeCount = 0;
allMode = [];
for d = 1:numel(days)
    path = fullfile(daySpeDir, ['Ion_spe_', char(days(d)), '.mat']);
    if ~isfile(path)
        continue
    end
    s = load(path, 'H_spe_num');
    if ~isfield(s, 'H_spe_num')
        continue
    end
    fn = fieldnames(s.H_spe_num);
    for k = 1:numel(fn)
        seg = s.H_spe_num.(fn{k});
        if ~isfield(seg, 't') || ~isfield(seg, 'p') || ~isfield(seg, 'f')
            continue
        end
        t = double(seg.t(:));
        mask = t >= tStart & t <= tStop;
        if ~any(mask)
            continue
        end
        p = double(seg.p(mask, :));
        e = double(seg.f(:));
        idx = numel(segments) + 1;
        segments(idx).t = t(mask);
        segments(idx).energy = e;
        segments(idx).p = p;
        segments(idx).mode = double(seg.mod);
        if isfield(seg, 'p_label')
            segments(idx).pLabel = seg.p_label;
        else
            segments(idx).pLabel = {};
        end
        timeCount = timeCount + nnz(mask);
        if isfield(seg, 'mod')
            allMode = [allMode; double(seg.mod) * ones(nnz(mask), 1)]; %#ok<AGROW>
        end
    end
end
end

function plot_h_spectrum_segments(ax, segments)
hold(ax, 'on');
for k = 1:numel(segments)
    t = segments(k).t(:);
    e = segments(k).energy(:);
    p = segments(k).p;
    if isempty(t) || isempty(e) || isempty(p)
        continue
    end
    xEdges = time_edges(epoch_to_datenum(t));
    yEdges = energy_edges(e);
    c = log10(max(double(p), 1));
    surface(ax, ...
        repmat(xEdges(:)', numel(yEdges), 1), ...
        repmat(yEdges(:), 1, numel(xEdges)), ...
        zeros(numel(yEdges), numel(xEdges)), ...
        pad_for_surface(c), ...
        EdgeColor='none', FaceColor='flat');
end
set(ax, 'YScale', 'log');
colormap(ax, 'jet');
clim(ax, [4, 7]);
end

function c = pad_for_surface(c0)
c = nan(size(c0, 2) + 1, size(c0, 1) + 1);
c(1:end-1, 1:end-1) = c0';
end

function xEdges = time_edges(x)
x = x(:);
if numel(x) == 1
    halfWidth = 4.1 / 86400;
    xEdges = [x - halfWidth, x + halfWidth];
    return
end
mid = (x(1:end-1) + x(2:end)) / 2;
xEdges = [x(1) - (mid(1) - x(1)); mid; x(end) + (x(end) - mid(end))]';
end

function eEdges = energy_edges(e)
e = e(:);
if numel(e) == 1
    eEdges = e .* [0.925; 1.075];
    return
end
logE = log10(e);
mid = (logE(1:end-1) + logE(2:end)) / 2;
logEdges = [logE(1) - (mid(1) - logE(1)); mid; logE(end) + (logE(end) - mid(end))];
eEdges = 10 .^ logEdges;
end

function dn = epoch_to_datenum(t)
dn = datenum(datetime(t, ConvertFrom='posixtime', TimeZone='UTC'));
end

function [t, density, speed] = load_nv_mso2_window(nvDir, days, tStart, tStop)
t = [];
density = [];
speed = [];
for d = 1:numel(days)
    path = fullfile(nvDir, ['NV_', char(days(d)), '.mat']);
    if ~isfile(path)
        continue
    end
    s = load(path, 'NH_TW1', 'VH_TW1_MSO');
    if ~isfield(s, 'NH_TW1') || ~isfield(s, 'VH_TW1_MSO')
        continue
    end
    tt = double(s.NH_TW1(:, 1));
    mask = tt >= tStart & tt <= tStop;
    if ~any(mask)
        continue
    end
    vv = double(s.VH_TW1_MSO(mask, 2:4));
    t = [t; tt(mask)]; %#ok<AGROW>
    density = [density; double(s.NH_TW1(mask, 2))]; %#ok<AGROW>
    speed = [speed; sqrt(sum(vv.^2, 2))]; %#ok<AGROW>
end
[t, order] = sort(t);
density = density(order);
speed = speed(order);
end
