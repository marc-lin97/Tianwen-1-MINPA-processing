function summary = find_tw1_minpa_high_channel_candidates(dateList, maxFiles, threshold, outputCsv)
%FIND_TW1_MINPA_HIGH_CHANNEL_CANDIDATES Quickly find H+ max 2D DEF outliers.
%
% This read-only helper scans MINPA ori files and writes records whose H+
% energy-angle DEF exceeds threshold. It is intentionally narrower than the
% full quality-flag audit so rare bit-5 examples can be located quickly.

if nargin < 1 || isempty(dateList)
    dateList = {};
end
if nargin < 2 || isempty(maxFiles)
    maxFiles = inf;
end
if nargin < 3 || isempty(threshold)
    threshold = 1e10;
end
if nargin < 4 || isempty(outputCsv)
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputCsv = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_audit', ...
        'tw1_minpa_hplus_high_channel_candidates.csv');
end
if ischar(dateList) || isstring(dateList)
    dateList = cellstr(dateList);
end

oriDir = 'D:\Data\TW-1\result\MINPA\ori';
files = select_ori_files(oriDir, dateList);
if isfinite(maxFiles)
    files = files(1:min(numel(files), maxFiles));
end

rows = table();
for f = 1:numel(files)
    filePath = fullfile(files(f).folder, files(f).name);
    try
        one = scan_one_file(filePath, threshold);
    catch ME
        warning('Failed %s: %s', filePath, ME.message);
        continue
    end
    if ~isempty(one)
        rows = [rows; one]; %#ok<AGROW>
    end
    fprintf('High-channel scan %d/%d %s candidates=%d\n', f, numel(files), files(f).name, height(rows));
end

[outDir, ~, ~] = fileparts(outputCsv);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
if ~isempty(rows)
    rows = sortrows(rows, 'max_def_2d', 'descend');
    writetable(rows, outputCsv);
end

summary = struct();
summary.output_csv = outputCsv;
summary.files_scanned = numel(files);
summary.candidates = rows;
summary.threshold = threshold;
fprintf('Wrote %d candidates to %s\n', height(rows), outputCsv);
end

function rows = scan_one_file(filePath, threshold)
mode = parse_mode(filePath);
if isnan(mode) || mode == 13
    rows = table();
    return
end
energy = minpa_energy(mode);
masses = minpa_masses(mode);
nAngle = minpa_angle_count(mode);
if isempty(energy) || isempty(masses) || isempty(nAngle)
    rows = table();
    return
end
groups = species_groups(masses, mode);
if isempty(groups)
    rows = table();
    return
end

s = load(filePath, 'tw1_MINPA_data');
[times, rawRows, recordIndex] = read_times_and_counts(s.tw1_MINPA_data, mode);
valid = isfinite(times) & all(isfinite(rawRows), 2);
times = times(valid);
rawRows = rawRows(valid, :);
recordIndex = recordIndex(valid);
if isempty(times)
    rows = table();
    return
end

massLen = numel(masses);
n = numel(times);
dpfFlat = zeros(n, numel(energy) * nAngle);
for g = groups(:)'
    dpfFlat = dpfFlat + rawRows(:, g:massLen:end);
end
weights = repelem(energy(:)', nAngle);
defFlat = dpfFlat .* weights;
[maxDef, idx] = max(defFlat, [], 2, 'omitnan');
hit = isfinite(maxDef) & maxDef > threshold;
if ~any(hit)
    rows = table();
    return
end
energyIdx = ceil(double(idx(hit)) ./ nAngle);
angleIdx = mod(double(idx(hit)) - 1, nAngle) + 1;
timeUtc = strings(nnz(hit), 1);
hitTimes = times(hit);
for i = 1:numel(hitTimes)
    timeUtc(i) = string(datetime(hitTimes(i), ConvertFrom='posixtime', TimeZone='UTC', Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
end

rows = table(repmat(string(filePath), nnz(hit), 1), recordIndex(hit), hitTimes, timeUtc, ...
    repmat(mode, nnz(hit), 1), maxDef(hit), energy(energyIdx(:)), angleIdx(:), ...
    'VariableNames', {'source_file', 'record_index', 'time_unix_s', 'time_utc', ...
    'mode', 'max_def_2d', 'max_def_energy_eV', 'max_def_angle_index'});
end

function files = select_ori_files(oriDir, dateList)
files = dir(fullfile(oriDir, '*.mat'));
if isempty(dateList)
    return
end
wanted = string(dateList);
keep = false(size(files));
for k = 1:numel(files)
    toks = regexp(files(k).name, '_(\d{8})\d{6}_(\d{8})\d{6}_', 'tokens', 'once');
    if isempty(toks)
        continue
    end
    keep(k) = any(wanted == string(toks{1})) || any(wanted == string(toks{2}));
end
files = files(keep);
end

function mode = parse_mode(filePath)
[~, name, ext] = fileparts(filePath);
tok = regexp([name ext], 'MINPA-MOD(\d+)-', 'tokens', 'once');
if isempty(tok)
    mode = NaN;
else
    mode = str2double(tok{1});
end
end

function [times, rawRows, recordIndex] = read_times_and_counts(a, mode)
utc = squeeze(struct2cell(a(1).value));
times0 = nan(numel(utc), 1);
for i = 1:numel(utc)
    txt = string(utc{i});
    if strlength(txt) > 20
        try
            times0(i) = posixtime(datetime(txt, InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''', TimeZone='UTC'));
        catch
            times0(i) = posixtime(datetime(txt, InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', TimeZone='UTC'));
        end
    end
end

if mode == 12
    tempValue = a(42).value.t;
    rawRows = nan(numel(times0) * 2, numel(tempValue) / 2);
    times = sort([times0 + 4.1 / 4; times0 + 4.1 * 3 / 4]);
    recordIndex = repelem((1:numel(times0))', 2);
    for i = 1:numel(times0)
        row = double(a(42).value(i).t);
        rawRows(2 * i - 1, :) = row(1:end/2);
        rawRows(2 * i, :) = row(end/2+1:end);
    end
else
    tempValue = a(42).value.t;
    rawRows = nan(numel(times0), numel(tempValue));
    times = times0;
    recordIndex = (1:numel(times0))';
    for i = 1:numel(times0)
        row = double(a(42).value(i).t);
        if numel(row) == size(rawRows, 2)
            rawRows(i, :) = row;
        end
    end
end
end

function groups = species_groups(masses, mode)
groups = find(abs(masses - 1) < 1e-6);
if isempty(groups) && mode == 12
    groups = [1, 2];
end
end

function n = minpa_angle_count(mode)
if mode >= 1 && mode <= 6
    n = 64;
elseif mode >= 7 && mode <= 11
    n = 256;
elseif mode == 12
    n = 16;
else
    n = [];
end
end

function e = minpa_energy(mode)
switch mode
    case {1, 2}
        e = [2.81,3.548928,4.482167,5.660813,7.149401,9.029433,11.40385,14.40264,18.19001,22.97332,29.01447,36.64422,46.28032,58.45036,73.82067,93.23282,117.7497,148.7135,187.8198,237.2095,299.587,378.3675,477.8644,603.5253,762.2305,962.6694,1215.816,1535.532,1939.321,2449.292,3093.366,3906.809,4934.157,6231.661,7870.361,9939.98,12553.83,15855.03,20024.33,25290];
    case {3, 4, 5, 7, 8}
        e = [2.81,3.246924,3.751784,4.335145,5.009211,5.788088,6.688071,7.727991,8.929607,10.31806,11.9224,13.77621,15.91825,18.39336,21.25332,24.55798,28.37647,32.7887,37.88697,43.77797,50.58496,58.45036,67.53873,78.04025,90.17464,104.1958,120.3971,139.1175,160.7487,185.7433,214.6243,247.996,286.5566,331.113,382.5974,442.087,510.8266,590.2544,682.0324,788.0808,910.6185,1052.21,1215.816,1404.862,1623.303,1875.708,2167.36,2504.36,2893.76,3343.707,3863.617,4464.366,5158.525,5960.618,6887.428,7958.346,9195.78,10625.62,12277.79,14186.84,16392.74,18941.63,21886.84,25290];
    case 6
        e = [2.81,3.410666,4.139729,5.024638,6.098705,7.402364,8.984693,10.90526,13.23637,16.06578,19.5,23.66831,28.72765,34.86848,42.32196,51.3687,62.34928,75.67706,91.85379,111.4885,135.3202,164.2463,199.3556,241.9698,293.6933,356.4731,432.6728,525.161,637.4194,773.6741,939.0547,1139.787,1383.428,1679.149,2038.084,2473.745,3002.533,3644.355,4423.372,5368.912,6516.57,7909.553,9600.298,11652.46,14143.29,17166.56,20836.08,25290];
    case {9, 10, 11}
        e = [44.96,49.67818,54.89149,60.65189,67.0168,74.04965,81.82054,90.40692,99.89437,110.3775,121.9606,134.7594,148.9013,164.5272,181.793,200.8706,221.9503,245.2421,270.9783,299.4152,330.8363,365.5548,403.9167,446.3044,493.1403,544.8912,602.073,665.2555,735.0686,812.2079,897.4423,991.6213,1095.684,1210.667,1337.716,1478.098,1633.212,1804.604,1993.982,2203.234,2434.445,2689.919,2972.204,3284.112,3628.752,4009.559,4430.329,4895.255,5408.971,5976.597,6603.791,7296.804,8062.543,8908.639,9843.526,10876.52,12017.92,13279.1,14672.63,16212.4,17913.76,19793.66,21870.84,24166];
    case 12
        e = [2.81,3.267539,3.799578,4.418245,5.137647,5.974187,6.946936,8.078073,9.393388,10.92287,12.70139,14.7695,17.17435,19.97077,23.22251,27.00373,31.40062,36.51344,42.45875,49.37211,57.41115,66.75914,77.62922,90.26922,104.9673,122.0587,141.9329,165.0432,191.9164,223.1653,259.5023,301.7558,350.8893,408.023,474.4595,551.7135,641.5465,746.0065,867.4753,1008.722,1172.968,1363.957,1586.043,1844.292,2144.589,2493.783,2899.834,3372];
    otherwise
        e = [];
end
end

function m = minpa_masses(mode)
if mode >= 1 && mode <= 6
    m = [1, 2, 4, 16, 38, 32, 44, 65];
elseif mode == 7 || mode == 8
    m = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 22, 28, 32, 44, 69];
elseif mode >= 9 && mode <= 11
    m = 1;
elseif mode == 12
    m = [0.76,1.25,1.75,2.25,2.75,3.25,3.75,4.26,5.66,6.34,7.65,8.35,9.62,10.38,11.61,12.39,13.55,14.47,15.56,16.45,17.45,18.58,20.98,23.04,26.97,28.92,30.94,33.17,40.07,48.12,64.68,68.95];
else
    m = [];
end
end
