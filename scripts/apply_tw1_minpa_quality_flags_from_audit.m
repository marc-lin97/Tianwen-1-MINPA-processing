function summary = apply_tw1_minpa_quality_flags_from_audit(auditCsv, dateList, dryRun)
%APPLY_TW1_MINPA_QUALITY_FLAGS_FROM_AUDIT Add MINPA quality flags to products.
%
% The audit CSV must be produced by audit_tw1_minpa_quality_flags with
% stopAfterSamples=false for production use. This script matches flags by UTC
% seconds and appends the H+ quality flag to NV_MSO_2 and day_spe products.
%
% Variables added to NV_MSO_2:
%   quality_flag_TW1              [time_unix_s, uint32 bitmask]
%   quality_flag_bit_TW1          [time_unix_s, bit1, bit2, bit3, bit4, bit5]
%   quality_flag_binary_TW1       string vector in problem-order bits
%   MINPA_quality_flag_info       metadata struct
%
% Fields added to each *_spe_num segment when its time vector can be matched:
%   quality_flag
%   quality_flag_bit
%   quality_flag_binary
%
% dryRun defaults to true. Set dryRun=false only after validating audit plots.

if nargin < 1 || isempty(auditCsv)
    auditCsv = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
        'outputs', 'tw1_minpa_quality_flag_audit', ...
        'tw1_minpa_hplus_quality_flag_audit_records.csv');
end
if nargin < 2 || isempty(dateList)
    dateList = {};
end
if nargin < 3 || isempty(dryRun)
    dryRun = true;
end
if ischar(dateList) || isstring(dateList)
    dateList = cellstr(dateList);
end

paths.nvMso2Dir = 'D:\Data\TW-1\result\MINPA\NV_MSO_2';
paths.daySpeDir = 'D:\Data\TW-1\result\MINPA\day_spe';

opts = detectImportOptions(auditCsv);
opts = setvartype(opts, {'quality_flag_binary'}, 'string');
rows = readtable(auditCsv, opts);
rows.day = extractBetween(string(rows.time_utc), 1, 10);
rows.day = erase(rows.day, "-");
rows = sortrows(rows, {'day', 'time_unix_s'});

if ~isempty(dateList)
    wanted = string(dateList);
    rows = rows(ismember(rows.day, wanted), :);
end

days = unique(rows.day);
summary = struct();
summary.audit_csv = auditCsv;
summary.dry_run = dryRun;
summary.nv_mso2_dir = paths.nvMso2Dir;
summary.day_spe_dir = paths.daySpeDir;
summary.days = struct('day', {}, 'records', {}, 'nv_matched', {}, ...
    'nv_unmatched', {}, 'spectra_segments_updated', {}, 'spectra_unmatched', {});

for d = 1:numel(days)
    day = char(days(d));
    dayRows = rows(rows.day == days(d), :);
    [daySummary, flagByTime] = update_nv_mso2_day(paths.nvMso2Dir, day, dayRows, dryRun);
    speSummary = update_day_spe_day(paths.daySpeDir, day, flagByTime, dryRun);

    idx = numel(summary.days) + 1;
    summary.days(idx).day = day;
    summary.days(idx).records = height(dayRows);
    summary.days(idx).nv_matched = daySummary.matched;
    summary.days(idx).nv_unmatched = daySummary.unmatched;
    summary.days(idx).spectra_segments_updated = speSummary.segments_updated;
    summary.days(idx).spectra_unmatched = speSummary.unmatched_records;
    fprintf('%s records=%d nvMatched=%d nvUnmatched=%d spectraSegments=%d spectraUnmatched=%d dryRun=%d\n', ...
        day, height(dayRows), daySummary.matched, daySummary.unmatched, ...
        speSummary.segments_updated, speSummary.unmatched_records, dryRun);
end
end

function [out, flagByTime] = update_nv_mso2_day(nvDir, day, rows, dryRun)
tolS = 1e-3;
nvPath = fullfile(nvDir, ['NV_', day, '.mat']);
out = struct('matched', 0, 'unmatched', height(rows));
flagByTime = make_flag_map(rows);
if ~isfile(nvPath)
    warning('Missing NV_MSO_2 day: %s', nvPath);
    return
end

s = load(nvPath);
t = s.NH_TW1(:, 1);
[flag, bits, binary, matched] = flags_for_times(t, rows, tolS);
quality_flag_TW1 = [t, double(flag)];
quality_flag_bit_TW1 = [t, double(bits)];
quality_flag_binary_TW1 = binary;
MINPA_quality_flag_info = quality_flag_info(rows);

out.matched = nnz(matched);
out.unmatched = nnz(~matched);
if dryRun
    return
end
save(nvPath, 'quality_flag_TW1', 'quality_flag_bit_TW1', ...
    'quality_flag_binary_TW1', 'MINPA_quality_flag_info', '-append');
end

function out = update_day_spe_day(daySpeDir, day, flagByTime, dryRun)
tolS = 1e-3;
spePath = fullfile(daySpeDir, ['Ion_spe_', day, '.mat']);
out = struct('segments_updated', 0, 'unmatched_records', 0);
if ~isfile(spePath)
    warning('Missing day_spe day: %s', spePath);
    return
end
s = load(spePath);
topFields = fieldnames(s);
for f = 1:numel(topFields)
    topName = topFields{f};
    if ~endsWith(topName, '_spe_num') || ~isstruct(s.(topName))
        continue
    end
    segNames = fieldnames(s.(topName));
    for k = 1:numel(segNames)
        segName = segNames{k};
        seg = s.(topName).(segName);
        if ~isfield(seg, 't')
            continue
        end
        t = double(seg.t(:));
        [flag, bits, binary, matched] = flags_for_times_from_map(t, flagByTime, tolS);
        seg.quality_flag = flag;
        seg.quality_flag_bit = bits;
        seg.quality_flag_binary = binary;
        s.(topName).(segName) = seg;
        out.segments_updated = out.segments_updated + 1;
        out.unmatched_records = out.unmatched_records + nnz(~matched);
    end
end
if dryRun
    return
end
vars = fieldnames(s);
for k = 1:numel(vars)
    eval([vars{k}, ' = s.(vars{k});']); %#ok<EVLDOT>
end
save(spePath, vars{:}, '-append');
end

function m = make_flag_map(rows)
m.time = double(rows.time_unix_s(:));
m.flag = uint32(rows.quality_flag(:));
m.bits = logical([rows.flag_bit1_sparse_caution, rows.flag_bit2_sparse_invalid, ...
    rows.flag_bit3_uv_contamination, rows.flag_bit4_evenodd_error, ...
    rows.flag_bit5_high_channel]);
m.binary = string(rows.quality_flag_binary(:));
end

function [flag, bits, binary, matched] = flags_for_times(t, rows, tolS)
m = make_flag_map(rows);
[flag, bits, binary, matched] = flags_for_times_from_map(t, m, tolS);
end

function [flag, bits, binary, matched] = flags_for_times_from_map(t, m, tolS)
flag = zeros(numel(t), 1, 'uint32');
bits = false(numel(t), 5);
binary = strings(numel(t), 1);
matched = false(numel(t), 1);
for i = 1:numel(t)
    [dt, idx] = min(abs(m.time - t(i)));
    if isempty(idx) || ~isfinite(dt) || dt > tolS
        binary(i) = "99999";
        continue
    end
    flag(i) = m.flag(idx);
    bits(i, :) = m.bits(idx, :);
    binary(i) = m.binary(idx);
    matched(i) = true;
end
end

function info = quality_flag_info(rows)
info = struct();
info.species = 'H+';
info.flag_encoding = ['Five-bit uint32 bitmask. Bit order follows problem ', ...
    '1-5: bit1 valid 2D channels 5-10; bit2 valid 2D channels <5; ', ...
    'bit3 >80% 1D DEF >1e5; bit4 peaks require DEF >=1e5, valleys ', ...
    'must be <50% of both neighbors, >=52.5% finite 1D DEF interior channels are valid peaks/valleys, ', ...
    'and >=55% of those extrema fit one odd/even phase; ', ...
    'bit5 any 2D DEF >1e10.'];
info.flag_binary_missing = '99999';
info.thresholds = struct('valid_raw_channel_min', 0, ...
    'valid_raw_channel_max_exclusive', 1e8, ...
    'uv_def_threshold', 1e5, ...
    'uv_fraction_threshold', 0.8, ...
    'alternating_peak_def_min', 1e5, ...
    'alternating_valley_neighbor_fraction_max', 0.5, ...
    'alternating_extrema_fraction_threshold', 0.525, ...
    'alternating_phase_fraction_threshold', 0.55, ...
    'alternating_energy_bands_eV', [0, 30; 30, 300; 300, Inf; 0, 50; 50, 500; 500, 5000], ...
    'evenodd_bit4_rule', 'all energies OR 0-30 eV OR 30-300 eV OR 300 eV-max energy OR 0-50 eV OR 50-500 eV OR 500-5000 eV', ...
    'high_channel_def_threshold', 1e10);
info.source_audit_records = sprintf('records=%d, first=%s, last=%s', ...
    height(rows), string(rows.time_utc(1)), string(rows.time_utc(end)));
end
