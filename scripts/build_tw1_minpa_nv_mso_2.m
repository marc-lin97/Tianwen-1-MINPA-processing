function summary = build_tw1_minpa_nv_mso_2(dateList)
%BUILD_TW1_MINPA_NV_MSO_2 Build MINPA NV_MSO_2 products from legacy NV_MSO.
%
% The legacy NV_MSO product is preserved. This derivative product keeps the
% legacy MSO velocities unchanged, rescales densities from fixed dE/E=0.15 to
% a mode-aware numerical energy integration width derived from adjacent energy
% centers, and appends the spacecraft +X_b axis in MSO coordinates.

if nargin < 1 || isempty(dateList)
    dateList = {};
end
if ischar(dateList) || isstring(dateList)
    dateList = cellstr(dateList);
end

paths.nvMsoDir = 'D:\Data\TW-1\result\MINPA\NV_MSO';
paths.daySpeDir = 'D:\Data\TW-1\result\MINPA\day_spe';
paths.momagDir = 'D:\Data\TW-1\result\MOMAG\C\01Hz_all';
paths.outDir = 'D:\Data\TW-1\result\MINPA\NV_MSO_2';
if ~exist(paths.outDir, 'dir')
    mkdir(paths.outDir);
end

oldDeOverE = 0.15;
matchTolS = 1e-3;
attTolS = 5.0;

widthByMode = minpa_energy_width_by_mode();
scaleByMode = widthByMode;
modeKeys = fieldnames(scaleByMode);
for k = 1:numel(modeKeys)
    key = modeKeys{k};
    scaleByMode.(key) = widthByMode.(key) / oldDeOverE;
end

files = dir(fullfile(paths.nvMsoDir, 'NV_*.mat'));
if ~isempty(dateList)
    keep = false(size(files));
    wanted = string(dateList);
    for k = 1:numel(files)
        day = regexp(files(k).name, 'NV_(\d{8})\.mat', 'tokens', 'once');
        keep(k) = ~isempty(day) && any(wanted == string(day{1}));
    end
    files = files(keep);
end

summary = struct();
summary.output_dir = paths.outDir;
summary.old_dE_over_E = oldDeOverE;
summary.mode_energy_width_over_E = widthByMode;
summary.mode_density_scale = scaleByMode;
summary.files = struct('day', {}, 'records', {}, 'unmatched_mode_records', {}, ...
    'unmatched_attitude_records', {}, 'output', {});

for f = 1:numel(files)
    tok = regexp(files(f).name, 'NV_(\d{8})\.mat', 'tokens', 'once');
    if isempty(tok)
        continue
    end
    day = tok{1};
    inPath = fullfile(files(f).folder, files(f).name);
    daySpePath = fullfile(paths.daySpeDir, ['Ion_spe_', day, '.mat']);
    momagPath = fullfile(paths.momagDir, ['Bss', day, '.mat']);
    outPath = fullfile(paths.outDir, files(f).name);

    if ~exist(daySpePath, 'file')
        warning('Missing day_spe for %s: %s', day, daySpePath);
        continue
    end

    data = load(inPath);
    [modeTime, modeValue] = read_day_spe_modes(daySpePath);
    t = data.NH_TW1(:, 1);
    [modeVec, unmatchedMode] = match_modes_to_times(t, modeTime, modeValue, matchTolS);
    scaleVec = scale_for_modes(modeVec, scaleByMode);

    data.NH_TW1(:, 2) = data.NH_TW1(:, 2) .* scaleVec;
    data.NO_TW1(:, 2) = data.NO_TW1(:, 2) .* scaleVec;
    data.NO2_TW1(:, 2) = data.NO2_TW1(:, 2) .* scaleVec;

    [Xb_MSO, unmatchedAtt] = compute_xb_mso(t, momagPath, attTolS);
    mode_TW1 = [t, modeVec];
    density_scale_TW1 = [t, scaleVec];
    energy_width_over_E_TW1 = [t, scaleVec * oldDeOverE];
    NV_MSO_2_info = build_info(day, inPath, daySpePath, momagPath, oldDeOverE, ...
        widthByMode, scaleByMode, matchTolS, attTolS);

    NH_TW1 = data.NH_TW1;
    NO_TW1 = data.NO_TW1;
    NO2_TW1 = data.NO2_TW1;
    VH_TW1_MSO = data.VH_TW1_MSO;
    VO_TW1_MSO = data.VO_TW1_MSO;
    VO2_TW1_MSO = data.VO2_TW1_MSO;

    save(outPath, 'NH_TW1', 'NO_TW1', 'NO2_TW1', ...
        'VH_TW1_MSO', 'VO_TW1_MSO', 'VO2_TW1_MSO', ...
        'Xb_MSO', 'mode_TW1', 'density_scale_TW1', ...
        'energy_width_over_E_TW1', 'NV_MSO_2_info');

    idx = numel(summary.files) + 1;
    summary.files(idx).day = day;
    summary.files(idx).records = numel(t);
    summary.files(idx).unmatched_mode_records = nnz(unmatchedMode);
    summary.files(idx).unmatched_attitude_records = nnz(unmatchedAtt);
    summary.files(idx).output = outPath;
    fprintf('Wrote %s records=%d unmatchedMode=%d unmatchedAtt=%d\n', ...
        outPath, numel(t), nnz(unmatchedMode), nnz(unmatchedAtt));
end

summaryPath = fullfile(paths.outDir, 'NV_MSO_2_build_summary.mat');
save(summaryPath, 'summary');
end

function out = build_info(day, inPath, daySpePath, momagPath, oldDeOverE, widthByMode, scaleByMode, matchTolS, attTolS)
out = struct();
out.day = day;
out.source_nv_mso = inPath;
out.source_day_spe = daySpePath;
out.source_momag = momagPath;
out.density_processing = ['Legacy NV_MSO densities rescaled from fixed dE/E=0.15 to ', ...
    'mode-aware numerical full-bin dE/E from adjacent logarithmic energy centers.'];
out.velocity_processing = 'Legacy NV_MSO velocity components are preserved unchanged.';
out.xb_mso_processing = 'Xb_MSO = R_MSO_from_body * [1;0;0] using nearest MOMAG Roll/Pitch/Yaw within 5 s.';
out.old_dE_over_E = oldDeOverE;
out.mode_energy_width_over_E = widthByMode;
out.mode_density_scale = scaleByMode;
out.mode_match_tolerance_s = matchTolS;
out.attitude_match_tolerance_s = attTolS;
out.coordinate_system = 'MSO';
out.units = struct('density', 'cm^-3', 'velocity', 'km/s', 'Xb_MSO', 'unit vector');
end

function [modeTime, modeValue] = read_day_spe_modes(daySpePath)
s = load(daySpePath, 'H_spe_num');
fn = fieldnames(s.H_spe_num);
modeTime = [];
modeValue = [];
for k = 1:numel(fn)
    seg = s.H_spe_num.(fn{k});
    if ~isfield(seg, 't') || ~isfield(seg, 'mod')
        continue
    end
    tt = double(seg.t(:));
    mm = double(seg.mod) * ones(size(tt));
    modeTime = [modeTime; tt]; %#ok<AGROW>
    modeValue = [modeValue; mm]; %#ok<AGROW>
end
[modeTime, order] = sort(modeTime);
modeValue = modeValue(order);
end

function [modeVec, unmatched] = match_modes_to_times(t, modeTime, modeValue, tolS)
modeVec = NaN(size(t));
unmatched = true(size(t));
for i = 1:numel(t)
    [dt, idx] = min(abs(modeTime - t(i)));
    if ~isempty(idx) && isfinite(dt) && dt <= tolS
        modeVec(i) = modeValue(idx);
        unmatched(i) = false;
    end
end
end

function scaleVec = scale_for_modes(modeVec, scaleByMode)
scaleVec = NaN(size(modeVec));
for i = 1:numel(modeVec)
    if ~isfinite(modeVec(i))
        continue
    end
    key = sprintf('mode%d', round(modeVec(i)));
    if isfield(scaleByMode, key)
        scaleVec(i) = scaleByMode.(key);
    end
end
end

function [Xb_MSO, unmatched] = compute_xb_mso(t, momagPath, attTolS)
Xb_MSO = [t, NaN(numel(t), 3)];
unmatched = true(size(t));
if ~exist(momagPath, 'file')
    return
end
m = load(momagPath, 'Pitch_TW1', 'Roll_TW1', 'Yaw_TW1');
if ~isfield(m, 'Pitch_TW1') || ~isfield(m, 'Roll_TW1') || ~isfield(m, 'Yaw_TW1')
    return
end
tAtt = m.Pitch_TW1(:, 1);
for i = 1:numel(t)
    [dt, idx] = min(abs(tAtt - t(i)));
    if isempty(idx) || ~isfinite(dt) || dt >= attTolS
        continue
    end
    roll = -m.Roll_TW1(idx, 2);
    pitch = -m.Pitch_TW1(idx, 2);
    yaw = -m.Yaw_TW1(idx, 2);
    R = rotz_deg(yaw) * roty_deg(pitch) * rotx_deg(roll);
    xb = R * [1; 0; 0];
    Xb_MSO(i, 2:4) = xb(:)';
    unmatched(i) = false;
end
end

function R = rotx_deg(a)
c = cosd(a);
s = sind(a);
R = [1 0 0; 0 c -s; 0 s c];
end

function R = roty_deg(a)
c = cosd(a);
s = sind(a);
R = [c 0 s; 0 1 0; -s 0 c];
end

function R = rotz_deg(a)
c = cosd(a);
s = sind(a);
R = [c -s 0; s c 0; 0 0 1];
end

function widths = minpa_energy_width_by_mode()
energy = minpa_energy_tables();
widths = struct();
keys = fieldnames(energy);
for k = 1:numel(keys)
    key = keys{k};
    e = energy.(key);
    r = e(2:end) ./ e(1:end-1);
    deOverE = sqrt(r) - 1 ./ sqrt(r);
    widths.(key) = median(deOverE, 'omitnan');
end
end

function energy = minpa_energy_tables()
energy = struct();
energy.mode1 = [2.81,3.548928,4.482167,5.660813,7.149401,9.029433,11.40385,14.40264,18.19001,22.97332,29.01447,36.64422,46.28032,58.45036,73.82067,93.23282,117.7497,148.7135,187.8198,237.2095,299.587,378.3675,477.8644,603.5253,762.2305,962.6694,1215.816,1535.532,1939.321,2449.292,3093.366,3906.809,4934.157,6231.661,7870.361,9939.98,12553.83,15855.03,20024.33,25290]';
energy.mode2 = energy.mode1;
energy.mode3 = [2.81,3.246924,3.751784,4.335145,5.009211,5.788088,6.688071,7.727991,8.929607,10.31806,11.9224,13.77621,15.91825,18.39336,21.25332,24.55798,28.37647,32.7887,37.88697,43.77797,50.58496,58.45036,67.53873,78.04025,90.17464,104.1958,120.3971,139.1175,160.7487,185.7433,214.6243,247.996,286.5566,331.113,382.5974,442.087,510.8266,590.2544,682.0324,788.0808,910.6185,1052.21,1215.816,1404.862,1623.303,1875.708,2167.36,2504.36,2893.76,3343.707,3863.617,4464.366,5158.525,5960.618,6887.428,7958.346,9195.78,10625.62,12277.79,14186.84,16392.74,18941.63,21886.84,25290]';
energy.mode4 = energy.mode3;
energy.mode5 = energy.mode3;
energy.mode6 = [2.81,3.410666,4.139729,5.024638,6.098705,7.402364,8.984693,10.90526,13.23637,16.06578,19.5,23.66831,28.72765,34.86848,42.32196,51.3687,62.34928,75.67706,91.85379,111.4885,135.3202,164.2463,199.3556,241.9698,293.6933,356.4731,432.6728,525.161,637.4194,773.6741,939.0547,1139.787,1383.428,1679.149,2038.084,2473.745,3002.533,3644.355,4423.372,5368.912,6516.57,7909.553,9600.298,11652.46,14143.29,17166.56,20836.08,25290]';
energy.mode7 = energy.mode3;
energy.mode8 = energy.mode3;
energy.mode9 = [44.96,49.67818,54.89149,60.65189,67.0168,74.04965,81.82054,90.40692,99.89437,110.3775,121.9606,134.7594,148.9013,164.5272,181.793,200.8706,221.9503,245.2421,270.9783,299.4152,330.8363,365.5548,403.9167,446.3044,493.1403,544.8912,602.073,665.2555,735.0686,812.2079,897.4423,991.6213,1095.684,1210.667,1337.716,1478.098,1633.212,1804.604,1993.982,2203.234,2434.445,2689.919,2972.204,3284.112,3628.752,4009.559,4430.329,4895.255,5408.971,5976.597,6603.791,7296.804,8062.543,8908.639,9843.526,10876.52,12017.92,13279.1,14672.63,16212.4,17913.76,19793.66,21870.84,24166]';
energy.mode10 = energy.mode9;
energy.mode11 = energy.mode9;
energy.mode12 = [2.81,3.267539,3.799578,4.418245,5.137647,5.974187,6.946936,8.078073,9.393388,10.92287,12.70139,14.7695,17.17435,19.97077,23.22251,27.00373,31.40062,36.51344,42.45875,49.37211,57.41115,66.75914,77.62922,90.26922,104.9673,122.0587,141.9329,165.0432,191.9164,223.1653,259.5023,301.7558,350.8893,408.023,474.4595,551.7135,641.5465,746.0065,867.4753,1008.722,1172.968,1363.957,1586.043,1844.292,2144.589,2493.783,2899.834,3372]';
end
