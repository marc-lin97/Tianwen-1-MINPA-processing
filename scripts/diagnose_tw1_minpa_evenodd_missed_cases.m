function summary = diagnose_tw1_minpa_evenodd_missed_cases(outputDir)
%DIAGNOSE_TW1_MINPA_EVENODD_MISSED_CASES Inspect named bit-4 missed cases.
%
% This is a read-only diagnostic. It loads selected TW-1 MINPA ori records,
% recomputes H+ 1D DEF spectra, and compares full-energy and low-energy
% odd/even peak-valley metrics.

if nargin < 1 || isempty(outputDir)
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(repoRoot, 'outputs', 'tw1_minpa_evenodd_missed_case_diagnostics');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
recordsCsv = fullfile(repoRoot, 'outputs', ...
    'tw1_minpa_quality_flag_monthly3_mode1412_peak1e5_valley50', ...
    'tw1_minpa_hplus_quality_flag_audit_records.csv');

opts = detectImportOptions(recordsCsv);
opts = setvartype(opts, {'source_file', 'time_utc', 'quality_flag_binary'}, 'string');
records = readtable(recordsCsv, opts);

targets = target_windows();
rows = table();
spectraRows = table();

for i = 1:height(targets)
    target = targets(i, :);
    mask = records.time_unix_s >= target.t_start & records.time_unix_s <= target.t_stop;
    if target.mode > 0
        mask = mask & records.mode == target.mode;
    end
    windowRows = records(mask, :);
    if isempty(windowRows)
        warning('No audit rows for target %s', target.case_id);
        continue
    end
    [~, order] = sort(windowRows.time_unix_s);
    windowRows = windowRows(order, :);
    for k = 1:height(windowRows)
        r = windowRows(k, :);
        try
            [energy, def1d, def2d] = load_h_spectrum_at_record(r.source_file, r.record_index, r.mode);
        catch ME
            warning('Could not load spectrum %s record %g: %s', r.source_file, r.record_index, ME.message);
            continue
        end
        diag = metrics_by_energy_bands(energy, def1d, def2d);
        rows = [rows; record_summary_row(target, r, diag)]; %#ok<AGROW>
        spectraRows = [spectraRows; spectrum_rows(target, r, energy, def1d, diag.full.isPeak, diag.full.isValley)]; %#ok<AGROW>
    end
end

summaryCsv = fullfile(outputDir, 'tw1_minpa_evenodd_missed_case_metric_summary.csv');
spectraCsv = fullfile(outputDir, 'tw1_minpa_evenodd_missed_case_spectra.csv');
writetable(rows, summaryCsv);
writetable(spectraRows, spectraCsv);

summary = struct('summary_csv', summaryCsv, 'spectra_csv', spectraCsv, ...
    'output_dir', outputDir, 'rows', rows);
fprintf('Wrote %d diagnostic rows to %s\n', height(rows), summaryCsv);
end

function targets = target_windows()
caseId = [
    "case_20220716_first_bit4_neighbor"
    "case_20220616_example_next"
    "case_20250916_example_next"
    "case_20230731_example_next2"
    "case_20240101_0911_0915"
    "case_20241001_0508_0516"
    "case_20230930_last_bit4_next"
    ];
startUtc = [
    "2022-07-16T16:49:20Z"
    "2022-06-16T23:38:45Z"
    "2025-09-16T20:03:45Z"
    "2023-07-31T01:34:40Z"
    "2024-01-01T09:11:00Z"
    "2024-10-01T05:08:00Z"
    "2023-09-30T17:14:45Z"
    ];
stopUtc = [
    "2022-07-16T16:50:15Z"
    "2022-06-16T23:39:40Z"
    "2025-09-16T20:04:40Z"
    "2023-07-31T01:35:45Z"
    "2024-01-01T09:16:00Z"
    "2024-10-01T05:16:00Z"
    "2023-09-30T17:16:45Z"
    ];
mode = [4; 1; 1; 1; 1; 1; 1];
notes = [
    "20220716 UV example, first bit4 region and previous/next records"
    "20220616 bit4 example and next record"
    "20250916 bit4 example and next record"
    "20230731 UV example and next two records"
    "20240101 09:11-09:15 interval"
    "20241001 05:08-05:16 interval"
    "20230930 final bit4 in example and next record"
    ];
t0 = arrayfun(@(s) posixtime(datetime(s, InputFormat="yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone='UTC')), startUtc);
t1 = arrayfun(@(s) posixtime(datetime(s, InputFormat="yyyy-MM-dd'T'HH:mm:ss'Z'", TimeZone='UTC')), stopUtc);
targets = table(caseId, t0(:), t1(:), mode, notes, ...
    'VariableNames', {'case_id', 't_start', 't_stop', 'mode', 'notes'});
end

function row = record_summary_row(target, r, diag)
row = table( ...
    target.case_id, target.notes, r.source_file, r.record_index, r.time_utc, r.time_unix_s, r.mode, ...
    r.quality_flag, r.quality_flag_binary, r.flag_bit3_uv_contamination, r.flag_bit4_evenodd_error, ...
    r.alternating_extrema_fraction, r.alternating_extrema_count, r.alternating_extrema_total, ...
    r.alternating_phase_fraction, r.alternating_phase_count, r.alternating_phase_total, ...
    diag.low.nEnergy, diag.low.energyMin, diag.low.energyMax, diag.low.extremaFraction, diag.low.extremaCount, diag.low.extremaTotal, ...
    diag.low.phaseFraction, diag.low.phaseCount, diag.low.phaseTotal, diag.low.dominantPattern, ...
    diag.mid.nEnergy, diag.mid.energyMin, diag.mid.energyMax, diag.mid.extremaFraction, diag.mid.extremaCount, diag.mid.extremaTotal, ...
    diag.mid.phaseFraction, diag.mid.phaseCount, diag.mid.phaseTotal, diag.mid.dominantPattern, ...
    diag.bestWindow.energyMin, diag.bestWindow.energyMax, diag.bestWindow.nEnergy, diag.bestWindow.extremaFraction, ...
    diag.bestWindow.extremaCount, diag.bestWindow.extremaTotal, diag.bestWindow.phaseFraction, diag.bestWindow.phaseCount, ...
    diag.bestWindow.phaseTotal, diag.bestWindow.dominantPattern, ...
    diag.twoDLow.anglePassCount, diag.twoDLow.angleTotal, diag.twoDLow.anglePassFraction, ...
    diag.twoDLow.extremaFraction, diag.twoDLow.extremaCount, diag.twoDLow.extremaTotal, ...
    diag.twoDLow.phaseFraction, diag.twoDLow.phaseCount, diag.twoDLow.phaseTotal, ...
    diag.twoDLow.bestAngleIndex, diag.twoDLow.bestAngleExtremaFraction, diag.twoDLow.bestAnglePhaseFraction, ...
    diag.twoDFull.anglePassCount, diag.twoDFull.angleTotal, diag.twoDFull.anglePassFraction, ...
    diag.twoDFull.extremaFraction, diag.twoDFull.extremaCount, diag.twoDFull.extremaTotal, ...
    diag.twoDFull.phaseFraction, diag.twoDFull.phaseCount, diag.twoDFull.phaseTotal, ...
    diag.twoDFull.bestAngleIndex, diag.twoDFull.bestAngleExtremaFraction, diag.twoDFull.bestAnglePhaseFraction, ...
    'VariableNames', {'case_id', 'notes', 'source_file', 'record_index', 'time_utc', 'time_unix_s', 'mode', ...
    'quality_flag', 'quality_flag_binary', 'flag_bit3_uv_contamination', 'flag_bit4_evenodd_error', ...
    'full_extrema_fraction_audit', 'full_extrema_count_audit', 'full_extrema_total_audit', ...
    'full_phase_fraction_audit', 'full_phase_count_audit', 'full_phase_total_audit', ...
    'low_n_energy', 'low_energy_min_eV', 'low_energy_max_eV', 'low_extrema_fraction', 'low_extrema_count', 'low_extrema_total', ...
    'low_phase_fraction', 'low_phase_count', 'low_phase_total', 'low_dominant_pattern', ...
    'mid_n_energy', 'mid_energy_min_eV', 'mid_energy_max_eV', 'mid_extrema_fraction', 'mid_extrema_count', 'mid_extrema_total', ...
    'mid_phase_fraction', 'mid_phase_count', 'mid_phase_total', 'mid_dominant_pattern', ...
    'best_window_energy_min_eV', 'best_window_energy_max_eV', 'best_window_n_energy', 'best_window_extrema_fraction', ...
    'best_window_extrema_count', 'best_window_extrema_total', 'best_window_phase_fraction', 'best_window_phase_count', ...
    'best_window_phase_total', 'best_window_dominant_pattern', ...
    'low2d_angle_pass_count', 'low2d_angle_total', 'low2d_angle_pass_fraction', ...
    'low2d_extrema_fraction', 'low2d_extrema_count', 'low2d_extrema_total', ...
    'low2d_phase_fraction', 'low2d_phase_count', 'low2d_phase_total', ...
    'low2d_best_angle_index', 'low2d_best_angle_extrema_fraction', 'low2d_best_angle_phase_fraction', ...
    'full2d_angle_pass_count', 'full2d_angle_total', 'full2d_angle_pass_fraction', ...
    'full2d_extrema_fraction', 'full2d_extrema_count', 'full2d_extrema_total', ...
    'full2d_phase_fraction', 'full2d_phase_count', 'full2d_phase_total', ...
    'full2d_best_angle_index', 'full2d_best_angle_extrema_fraction', 'full2d_best_angle_phase_fraction'});
end

function out = spectrum_rows(target, r, energy, def1d, isPeakFull, isValleyFull)
n = numel(energy);
caseId = repmat(target.case_id, n, 1);
timeUtc = repmat(r.time_utc, n, 1);
recordIndex = repmat(r.record_index, n, 1);
mode = repmat(r.mode, n, 1);
qualityFlag = repmat(r.quality_flag, n, 1);
bit4 = repmat(r.flag_bit4_evenodd_error, n, 1);
energyIndex = (1:n)';
isPeak = false(n, 1);
isValley = false(n, 1);
isPeak(2:end-1) = isPeakFull;
isValley(2:end-1) = isValleyFull;
out = table(caseId, timeUtc, recordIndex, mode, qualityFlag, bit4, energyIndex, energy(:), def1d(:), isPeak, isValley, ...
    'VariableNames', {'case_id', 'time_utc', 'record_index', 'mode', 'quality_flag', 'flag_bit4_evenodd_error', ...
    'energy_index', 'energy_eV', 'def1d', 'is_peak_full', 'is_valley_full'});
end

function [energy, def1d, def2d] = load_h_spectrum_at_record(filePath, recordIndex, mode)
geom = minpa_geometry(mode);
groups = species_groups(geom.masses, 'H', mode);
s = load(filePath, 'tw1_MINPA_data');
[~, rawRows, recordIndexAll] = read_times_and_counts(s.tw1_MINPA_data, mode);
idx = find(recordIndexAll == recordIndex, 1, 'first');
if isempty(idx)
    error('record_index %g not found', recordIndex);
end
[~, def2d, def1d] = extract_species_h_cube(rawRows(idx, :), geom, groups);
energy = geom.energy(:);
end

function diag = metrics_by_energy_bands(energy, def1d, def2d)
diag.full = band_metrics(energy, def1d, true(size(energy)));
lowMask = energy <= 1000;
midMask = energy > 1000;
diag.low = band_metrics(energy, def1d, lowMask);
diag.mid = band_metrics(energy, def1d, midMask);
diag.bestWindow = best_sliding_window(energy, def1d);
diag.twoDLow = two_d_band_metrics(energy, def2d, lowMask);
diag.twoDFull = two_d_band_metrics(energy, def2d, true(size(energy)));
end

function best = best_sliding_window(energy, def1d)
best = empty_band_result();
minEnergyBins = 10;
for i = 1:(numel(energy) - minEnergyBins + 1)
    for j = (i + minEnergyBins - 1):numel(energy)
        mask = false(size(energy));
        mask(i:j) = true;
        one = band_metrics(energy, def1d, mask);
        if one.extremaTotal < 8 || ~isfinite(one.phaseFraction)
            continue
        end
        score = one.extremaFraction + 0.2 * one.phaseFraction;
        bestScore = best.extremaFraction + 0.2 * best.phaseFraction;
        if ~isfinite(bestScore) || score > bestScore
            best = one;
        end
    end
end
end

function out = band_metrics(energy, def1d, mask)
y = double(def1d(:));
e = double(energy(:));
idx = find(mask(:));
out = empty_band_result();
if numel(idx) < 3
    return
end
y = y(idx);
e = e(idx);
[extFrac, extCount, extTotal, isPeak, isValley] = alternating_extrema_fraction(y);
[phaseFrac, phaseCount, phaseTotal, pattern] = alternating_phase_fraction(y, isPeak, isValley);
out.nEnergy = numel(e);
out.energyMin = min(e);
out.energyMax = max(e);
out.extremaFraction = extFrac;
out.extremaCount = extCount;
out.extremaTotal = extTotal;
out.phaseFraction = phaseFrac;
out.phaseCount = phaseCount;
out.phaseTotal = phaseTotal;
out.dominantPattern = pattern;
out.isPeak = isPeak;
out.isValley = isValley;
end

function out = empty_band_result()
out = struct('nEnergy', 0, 'energyMin', NaN, 'energyMax', NaN, ...
    'extremaFraction', NaN, 'extremaCount', 0, 'extremaTotal', 0, ...
    'phaseFraction', NaN, 'phaseCount', 0, 'phaseTotal', 0, ...
    'dominantPattern', "", 'isPeak', [], 'isValley', []);
end

function [fraction, count, total, isPeak, isValley] = alternating_extrema_fraction(def1d)
peakDefMin = 1e5;
valleyNeighborFractionMax = 0.5;
y = double(def1d(:));
left = y(1:end-2);
mid = y(2:end-1);
right = y(3:end);
usable = isfinite(left) & isfinite(mid) & isfinite(right);
isPeak = mid > left & mid > right & mid >= peakDefMin;
isValley = mid < valleyNeighborFractionMax .* left & mid < valleyNeighborFractionMax .* right;
count = nnz(usable & (isPeak | isValley));
total = nnz(usable);
if total > 0
    fraction = count / total;
else
    fraction = NaN;
end
end

function out = two_d_band_metrics(energy, def2d, mask)
idx = find(mask(:));
out = struct('anglePassCount', 0, 'angleTotal', 0, 'anglePassFraction', NaN, ...
    'extremaFraction', NaN, 'extremaCount', 0, 'extremaTotal', 0, ...
    'phaseFraction', NaN, 'phaseCount', 0, 'phaseTotal', 0, ...
    'bestAngleIndex', NaN, 'bestAngleExtremaFraction', NaN, 'bestAnglePhaseFraction', NaN);
if numel(idx) < 3 || isempty(def2d)
    return
end
z = double(def2d(idx, :));
angleTotal = size(z, 2);
anglePass = false(angleTotal, 1);
angleExt = nan(angleTotal, 1);
anglePhase = nan(angleTotal, 1);
sumExtCount = 0;
sumExtTotal = 0;
sumPhaseCount = 0;
sumPhaseTotal = 0;
bestScore = -Inf;
for a = 1:angleTotal
    y = z(:, a);
    [extFrac, extCount, extTotal, isPeak, isValley] = alternating_extrema_fraction(y);
    [phaseFrac, phaseCount, phaseTotal] = alternating_phase_fraction(y, isPeak, isValley);
    angleExt(a) = extFrac;
    anglePhase(a) = phaseFrac;
    if isfinite(extFrac) && isfinite(phaseFrac) ...
            && extFrac >= 0.525 && phaseFrac >= 0.55
        anglePass(a) = true;
    end
    sumExtCount = sumExtCount + extCount;
    sumExtTotal = sumExtTotal + extTotal;
    sumPhaseCount = sumPhaseCount + phaseCount;
    sumPhaseTotal = sumPhaseTotal + phaseTotal;
    score = extFrac + 0.2 * phaseFrac;
    if isfinite(score) && score > bestScore
        bestScore = score;
        out.bestAngleIndex = a;
        out.bestAngleExtremaFraction = extFrac;
        out.bestAnglePhaseFraction = phaseFrac;
    end
end
out.anglePassCount = nnz(anglePass);
out.angleTotal = angleTotal;
out.anglePassFraction = out.anglePassCount / max(out.angleTotal, 1);
out.extremaCount = sumExtCount;
out.extremaTotal = sumExtTotal;
if sumExtTotal > 0
    out.extremaFraction = sumExtCount / sumExtTotal;
end
out.phaseCount = sumPhaseCount;
out.phaseTotal = sumPhaseTotal;
if sumPhaseTotal > 0
    out.phaseFraction = sumPhaseCount / sumPhaseTotal;
end
end

function [fraction, count, total, pattern] = alternating_phase_fraction(def1d, isPeak, isValley)
y = double(def1d(:));
usable = isfinite(y(1:end-2)) & isfinite(y(2:end-1)) & isfinite(y(3:end));
interiorIndex = (2:numel(y)-1)';
oddInterior = mod(interiorIndex, 2) == 1;
patternOddPeak = (isPeak & oddInterior) | (isValley & ~oddInterior);
patternEvenPeak = (isPeak & ~oddInterior) | (isValley & oddInterior);
countOddPeak = nnz(usable & patternOddPeak);
countEvenPeak = nnz(usable & patternEvenPeak);
if countOddPeak >= countEvenPeak
    count = countOddPeak;
    pattern = "odd_peak_even_valley";
else
    count = countEvenPeak;
    pattern = "even_peak_odd_valley";
end
total = nnz(usable & (isPeak | isValley));
if total > 0
    fraction = count / total;
else
    fraction = NaN;
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
validTime = isfinite(times);
times = times(validTime);
rawRows = rawRows(validTime, :);
recordIndex = recordIndex(validTime);
end

function [dpf2d, def2d, def1d] = extract_species_h_cube(raw, geom, groups)
nEnergy = numel(geom.energy);
nAngle = numel(geom.omega);
dpf2d = zeros(nEnergy, nAngle);
for g = groups(:)'
    oneMass = raw(g:geom.massLen:end);
    if numel(oneMass) ~= nEnergy * nAngle
        dpf2d(:) = NaN;
        break
    end
    dpf2d = dpf2d + reshape(oneMass, [nAngle, nEnergy])';
end
def2d = dpf2d .* geom.energy(:);
omegaSum = sum(geom.omega, 'omitnan');
if isfinite(omegaSum) && omegaSum > 0
    def1d = sum(def2d .* geom.omega(:)', 2, 'omitnan') ./ omegaSum;
else
    def1d = nan(nEnergy, 1);
end
end

function geom = minpa_geometry(mode)
energy = minpa_energy(mode);
masses = minpa_masses(mode);
pEdges = pitch_edges(mode);
if isempty(pEdges)
    geom = struct('energy', [], 'masses', [], 'massLen', 0, 'omega', []);
    return
end
if mode == 12
    nAzimuth = 1;
else
    nAzimuth = 16;
end
azEdges = linspace(0, 360, nAzimuth + 1); %#ok<NASGU>
dphi = deg2rad(360 / nAzimuth);
omegaPitch = (cosd(pEdges(1:end-1) + 90) - cosd(pEdges(2:end) + 90)) * dphi;
geom.energy = energy(:);
geom.masses = masses(:);
geom.massLen = numel(masses);
geom.omega = repelem(omegaPitch(:), nAzimuth);
end

function groups = species_groups(masses, species, mode)
switch species
    case 'H'
        target = 1;
        fallback = [1, 2];
    otherwise
        target = NaN;
        fallback = [];
end
groups = find(abs(masses - target) < 1e-6);
if isempty(groups) && mode == 12
    groups = fallback;
end
end

function e = minpa_energy(mode)
switch mode
    case {1, 2}
        e = [2.81,3.548928,4.482167,5.660813,7.149401,9.029433,11.40385,14.40264,18.19001,22.97332,29.01447,36.64422,46.28032,58.45036,73.82067,93.23282,117.7497,148.7135,187.8198,237.2095,299.587,378.3675,477.8644,603.5253,762.2305,962.6694,1215.816,1535.532,1939.321,2449.292,3093.366,3906.809,4934.157,6231.661,7870.361,9939.98,12553.83,15855.03,20024.33,25290];
    case {3, 4, 5, 7, 8}
        e = [2.81,3.246924,3.751784,4.335145,5.009211,5.788088,6.688071,7.727991,8.929607,10.31806,11.9224,13.77621,15.91825,18.39336,21.25332,24.55798,28.37647,32.7887,37.88697,43.77797,50.58496,58.45036,67.53873,78.04025,90.17464,104.1958,120.3971,139.1175,160.7487,185.7433,214.6243,247.996,286.5566,331.113,382.5974,442.087,510.8266,590.2544,682.0324,788.0808,910.6185,1052.21,1215.816,1404.862,1623.303,1875.708,2167.36,2504.36,2893.76,3343.707,3863.617,4464.366,5158.525,5960.618,6887.428,7958.346,9195.78,10625.62,12277.79,14186.84,16392.74,18941.63,21886.84,25290];
    case 12
        e = [2.81,3.267539,3.799578,4.418245,5.137647,5.974187,6.946936,8.078073,9.393388,10.92287,12.70139,14.7695,17.17435,19.97077,23.22251,27.00373,31.40062,36.51344,42.45875,49.37211,57.41115,66.75914,77.62922,90.26922,104.9673,122.0587,141.9329,165.0432,191.9164,223.1653,259.5023,301.7558,350.8893,408.023,474.4595,551.7135,641.5465,746.0065,867.4753,1008.722,1172.968,1363.957,1586.043,1844.292,2144.589,2493.783,2899.834,3372];
    otherwise
        e = [];
end
end

function m = minpa_masses(mode)
if mode >= 1 && mode <= 6
    m = [1, 2, 4, 16, 38, 32, 44, 65];
elseif mode == 12
    m = [0.76,1.25,1.75,2.25,2.75,3.25,3.75,4.26,5.66,6.34,7.65,8.35,9.62,10.38,11.61,12.39,13.55,14.47,15.56,16.45,17.45,18.58,20.98,23.04,26.97,28.92,30.94,33.17,40.07,48.12,64.68,68.95];
else
    m = 1;
end
end

function p = pitch_edges(mode)
if mode >= 1 && mode <= 6
    p = linspace(0, 90, 5);
elseif mode >= 7 && mode <= 11
    p = linspace(0, 90, 17);
elseif mode == 12
    p = [67.4, 90];
else
    p = [];
end
end
