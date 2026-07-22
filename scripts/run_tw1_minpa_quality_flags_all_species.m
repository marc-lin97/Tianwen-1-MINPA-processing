function summary = run_tw1_minpa_quality_flags_all_species(dateList, dryRun, maxDays, overwrite, dataRoot, outputRoot)
%RUN_TW1_MINPA_QUALITY_FLAGS_ALL_SPECIES Compute and attach H+/O+/O2+ flags.
%
% Raw MINPA ori files are read-only. Processing is day-based and restartable.
% Each ori file is loaded once and all three species flags are computed from
% the same raw records. Derived NV_MSO_2 and day_spe MAT files are replaced
% only after a complete temporary MAT file has been written and checked.
%
% Example validation run:
%   run_tw1_minpa_quality_flags_all_species({'20250916'}, true, 1, true)
%
% Full production run:
%   run_tw1_minpa_quality_flags_all_species({}, false, inf, false)

if nargin < 1 || isempty(dateList)
    dateList = {};
end
if nargin < 2 || isempty(dryRun)
    dryRun = true;
end
if nargin < 3 || isempty(maxDays)
    maxDays = inf;
end
if nargin < 4 || isempty(overwrite)
    overwrite = false;
end
if nargin < 5 || isempty(dataRoot)
    dataRoot = getenv('TW1_MINPA_RESULT_ROOT');
    if isempty(dataRoot)
        dataRoot = 'D:\Data\TW-1\result\MINPA';
    end
end
repoRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 6 || isempty(outputRoot)
    outputRoot = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flags_all_species');
end
if ischar(dateList) || isstring(dateList)
    dateList = cellstr(dateList);
end
fullCollection = isempty(dateList);

cfg = tw1_minpa_quality_flag_config();
paths = struct();
paths.data_root = dataRoot;
paths.ori_dir = fullfile(dataRoot, 'ori');
paths.nv_mso2_dir = fullfile(dataRoot, 'NV_MSO_2');
paths.day_spe_dir = fullfile(dataRoot, 'day_spe');
paths.output_root = outputRoot;
paths.ori_checkpoint_dir = fullfile(outputRoot, 'ori_flag_checkpoints');
if ~exist(paths.output_root, 'dir')
    mkdir(paths.output_root);
end
if ~exist(paths.ori_checkpoint_dir, 'dir')
    mkdir(paths.ori_checkpoint_dir);
end

days = select_days(paths.nv_mso2_dir, dateList);
if isfinite(maxDays)
    days = days(1:min(numel(days), maxDays));
end
oriCatalog = build_ori_catalog(paths.ori_dir);
energyByMode = read_global_mode_energies(paths.day_spe_dir, unique(oriCatalog.mode));

summary = struct();
summary.algorithm_version = cfg.algorithm_version;
summary.started_utc = utc_now_string();
summary.finished_utc = '';
summary.dry_run = logical(dryRun);
summary.overwrite = logical(overwrite);
summary.paths = paths;
summary.days_requested = days;
summary.days = empty_day_summary();
if fullCollection
    summary.source_stage = precompute_all_source_flags( ...
        oriCatalog, energyByMode, paths, cfg, overwrite);
else
    summary.source_stage = struct('scope', 'selected products only', ...
        'files_total', 0, 'computed', 0, 'reused', 0, 'failed', 0);
end

for d = 1:numel(days)
    day = char(days(d));
    fprintf('=== %s (%d/%d) dryRun=%d overwrite=%d ===\n', ...
        day, d, numel(days), dryRun, overwrite);
    try
        daySummary = process_day(day, paths, oriCatalog, energyByMode, ...
            cfg, dryRun, overwrite);
    catch ME
        daySummary = failed_day_summary(day, ME);
        warning('Failed %s: %s', day, ME.message);
    end
    summary.days(end + 1) = daySummary; %#ok<AGROW>
    write_root_summary(summary, paths.output_root);
end

summary.finished_utc = utc_now_string();
write_root_summary(summary, paths.output_root);
fprintf('Finished %d days; outputs under %s\n', numel(days), paths.output_root);
end

function daySummary = process_day(day, paths, oriCatalog, energyByMode, cfg, dryRun, overwrite)
dayDir = fullfile(paths.output_root, day);
if ~exist(dayDir, 'dir')
    mkdir(dayDir);
end
checkpointPath = fullfile(dayDir, ['quality_flags_', day, '.mat']);
nvPath = fullfile(paths.nv_mso2_dir, ['NV_', day, '.mat']);
spePath = fullfile(paths.day_spe_dir, ['Ion_spe_', day, '.mat']);
if ~isfile(nvPath) || ~isfile(spePath)
    error('Missing derived product for %s (NV=%d, day_spe=%d).', ...
        day, isfile(nvPath), isfile(spePath));
end

if ~dryRun && ~overwrite && products_are_current(nvPath, spePath, cfg.algorithm_version)
    if isfile(checkpointPath)
        prior = load(checkpointPath, 'dayInfo');
    else
        prior = struct();
    end
    if isfield(prior, 'dayInfo') && isfield(prior.dayInfo, 'summary')
        daySummary = prior.dayInfo.summary;
        daySummary.status = 'skipped_complete';
        fprintf('%s products already contain %s; skipping.\n', day, cfg.algorithm_version);
        return
    end
end

targetTimes = collect_product_times(nvPath, spePath);
dayFiles = select_source_files_for_times(oriCatalog, targetTimes, 10);
if isempty(dayFiles)
    error('No ori files overlap the actual product UTC values in %s.', day);
end
[recordFlags, computeInfo] = compute_day_flags(dayFiles, energyByMode, ...
    cfg, paths, overwrite);
dayInfo = struct();
dayInfo.algorithm_version = cfg.algorithm_version;
dayInfo.day = day;
dayInfo.generated_utc = utc_now_string();
dayInfo.product_time_min_utc = unix_to_utc(min(targetTimes));
dayInfo.product_time_max_utc = unix_to_utc(max(targetTimes));
dayInfo.source_files = dayFiles.file_path;
dayInfo.source_file_size_bytes = dayFiles.size_bytes;
dayInfo.source_file_modified_datenum = dayFiles.modified_datenum;
dayInfo.raw_files_scanned = height(dayFiles);
dayInfo.raw_files_computed = computeInfo.computed;
dayInfo.raw_files_reused = computeInfo.reused;
dayInfo.raw_files_total = height(dayFiles);
dayInfo.records = numel(recordFlags.time_unix_s);
dayInfo.duplicate_records_removed = computeInfo.duplicates_removed;
dayInfo.duplicate_flag_conflicts = computeInfo.duplicate_flag_conflicts;
dayInfo.product_written = false;
dayInfo.summary = make_day_summary(day, recordFlags, computeInfo, false);
payload = struct('recordFlags', recordFlags, 'dayInfo', dayInfo, 'cfg', cfg);
atomic_save_struct(checkpointPath, payload, {'recordFlags', 'dayInfo', 'cfg'});

[nvStats, speStats] = update_products(nvPath, spePath, recordFlags, ...
    dayInfo, cfg, dryRun);
daySummary = make_day_summary(day, recordFlags, dayInfo, ~dryRun);
daySummary.nv_unmatched_H = nvStats.H.unmatched;
daySummary.nv_unmatched_O = nvStats.O.unmatched;
daySummary.nv_unmatched_O2 = nvStats.O2.unmatched;
daySummary.spe_unmatched_H = speStats.H.unmatched;
daySummary.spe_unmatched_O = speStats.O.unmatched;
daySummary.spe_unmatched_O2 = speStats.O2.unmatched;
daySummary.product_written = ~dryRun;
daySummary.status = ternary(dryRun, 'validated_dry_run', 'complete');

dayInfo.product_written = ~dryRun;
dayInfo.completed_utc = utc_now_string();
dayInfo.nv_stats = nvStats;
dayInfo.spe_stats = speStats;
dayInfo.summary = daySummary;
payload = struct('recordFlags', recordFlags, 'dayInfo', dayInfo, 'cfg', cfg);
atomic_save_struct(checkpointPath, payload, {'recordFlags', 'dayInfo', 'cfg'});

fprintf(['%s records=%d files=%d flag(H/O/O2)=%d/%d/%d ', ...
    'NVunmatched=%d/%d/%d SPEunmatched=%d/%d/%d written=%d\n'], ...
    day, daySummary.records, daySummary.raw_files, ...
    daySummary.flagged_H, daySummary.flagged_O, daySummary.flagged_O2, ...
    daySummary.nv_unmatched_H, daySummary.nv_unmatched_O, daySummary.nv_unmatched_O2, ...
    daySummary.spe_unmatched_H, daySummary.spe_unmatched_O, daySummary.spe_unmatched_O2, ...
    daySummary.product_written);
end

function [recordFlags, info] = compute_day_flags(dayFiles, energyByMode, cfg, paths, overwrite)
recordFlags = initialize_record_flags(cfg.species);
computed = 0;
reused = 0;
for f = 1:height(dayFiles)
    [oneFlags, sourceInfo, wasComputed] = load_or_compute_source_flags( ...
        dayFiles(f, :), energyByMode, paths, cfg, overwrite);
    computed = computed + wasComputed;
    reused = reused + ~wasComputed;
    n = numel(oneFlags.time_unix_s);
    if n == 0
        continue
    end
    recordFlags.time_unix_s = [recordFlags.time_unix_s; oneFlags.time_unix_s]; %#ok<AGROW>
    recordFlags.mode = [recordFlags.mode; oneFlags.mode]; %#ok<AGROW>
    for k = 1:numel(cfg.species)
        species = cfg.species{k};
        recordFlags.(species).flag = [recordFlags.(species).flag; oneFlags.(species).flag]; %#ok<AGROW>
        recordFlags.(species).available = [recordFlags.(species).available; oneFlags.(species).available]; %#ok<AGROW>
    end
    fprintf('  %s mode=%d records=%d source=%s\n', ...
        dayFiles.file_name(f), sourceInfo.mode, n, ternary(wasComputed, 'computed', 'checkpoint'));
end

[~, order] = sortrows([recordFlags.time_unix_s, recordFlags.mode], [1, 2]);
recordFlags.time_unix_s = recordFlags.time_unix_s(order);
recordFlags.mode = recordFlags.mode(order);
for k = 1:numel(cfg.species)
    species = cfg.species{k};
    recordFlags.(species).flag = recordFlags.(species).flag(order);
    recordFlags.(species).available = recordFlags.(species).available(order);
end

timeKey = int64(round(recordFlags.time_unix_s * 1e6));
duplicate = [false; diff(timeKey) == 0 & diff(recordFlags.mode) == 0];
conflict = false(size(duplicate));
for k = 1:numel(cfg.species)
    species = cfg.species{k};
    values = recordFlags.(species).flag;
    availability = recordFlags.(species).available;
    conflict = conflict | (duplicate & [false; values(2:end) ~= values(1:end-1)]) ...
        | (duplicate & [false; availability(2:end) ~= availability(1:end-1)]);
end
keep = ~duplicate;
recordFlags.time_unix_s = recordFlags.time_unix_s(keep);
recordFlags.mode = recordFlags.mode(keep);
for k = 1:numel(cfg.species)
    species = cfg.species{k};
    recordFlags.(species).flag = recordFlags.(species).flag(keep);
    recordFlags.(species).available = recordFlags.(species).available(keep);
end

info = struct();
info.raw_files = height(dayFiles);
info.computed = computed;
info.reused = reused;
info.duplicates_removed = nnz(duplicate);
info.duplicate_flag_conflicts = nnz(conflict);
if info.duplicate_flag_conflicts > 0
    warning('%d duplicate timestamps had conflicting flags; the first was retained.', ...
        info.duplicate_flag_conflicts);
end
end

function recordFlags = initialize_record_flags(speciesList)
recordFlags = struct('time_unix_s', zeros(0, 1), 'mode', zeros(0, 1));
for k = 1:numel(speciesList)
    species = speciesList{k};
    recordFlags.(species) = struct('flag', zeros(0, 1, 'uint32'), ...
        'available', false(0, 1));
end
end

function energyByMode = read_mode_energies(spePath)
s = load(spePath, 'H_spe_num');
if ~isfield(s, 'H_spe_num')
    error('Missing H_spe_num in %s.', spePath);
end
energyByMode = struct();
segments = fieldnames(s.H_spe_num);
for k = 1:numel(segments)
    seg = s.H_spe_num.(segments{k});
    if ~isfield(seg, 'mod') || ~isfield(seg, 'f')
        continue
    end
    mode = round(double(seg.mod));
    energy = double(seg.f(:));
    field = sprintf('mode%d', mode);
    if isfield(energyByMode, field)
        prior = energyByMode.(field);
        if numel(prior) ~= numel(energy) ...
                || any(abs(prior - energy) > max(1e-9, 1e-10 * abs(prior)))
            error('Inconsistent energy centers for mode %d in %s.', mode, spePath);
        end
    else
        energyByMode.(field) = energy;
    end
end
end

function energyByMode = read_global_mode_energies(daySpeDir, expectedModes)
expectedModes = unique(expectedModes(isfinite(expectedModes) & expectedModes ~= 13));
files = dir(fullfile(daySpeDir, 'Ion_spe_*.mat'));
energyByMode = struct();
foundModes = zeros(0, 1);
for f = 1:numel(files)
    path = fullfile(files(f).folder, files(f).name);
    try
        one = read_mode_energies(path);
    catch
        continue
    end
    fields = fieldnames(one);
    for k = 1:numel(fields)
        field = fields{k};
        mode = sscanf(field, 'mode%d');
        if isempty(mode)
            continue
        end
        energy = one.(field);
        if isfield(energyByMode, field)
            prior = energyByMode.(field);
            if numel(prior) ~= numel(energy) ...
                    || any(abs(prior - energy) > max(1e-9, 1e-10 * abs(prior)))
                error('Inconsistent global energy centers for mode %d in %s.', mode, path);
            end
        else
            energyByMode.(field) = energy;
            foundModes(end + 1, 1) = mode; %#ok<AGROW>
            fprintf('Energy centers for mode %d from %s (%d bins).\n', ...
                mode, files(f).name, numel(energy));
        end
    end
    if all(ismember(expectedModes, unique(foundModes)))
        break
    end
end
missing = expectedModes(~ismember(expectedModes, unique(foundModes)));
if ~isempty(missing)
    error('Could not find day_spe energy centers for modes: %s.', num2str(missing'));
end
end

function [recordFlags, sourceInfo, wasComputed] = load_or_compute_source_flags(fileRow, energyByMode, paths, cfg, overwrite)
mode = fileRow.mode(1);
modeDir = fullfile(paths.ori_checkpoint_dir, sprintf('mode%d', mode));
if ~exist(modeDir, 'dir')
    mkdir(modeDir);
end
[~, stem] = fileparts(char(fileRow.file_name(1)));
checkpointPath = fullfile(modeDir, [stem, '_quality_flags.mat']);
wasComputed = true;

if isfile(checkpointPath) && ~overwrite
    prior = load(checkpointPath, 'recordFlags', 'sourceInfo', 'cfg');
    if isfield(prior, 'recordFlags') && isfield(prior, 'sourceInfo') ...
            && isfield(prior.sourceInfo, 'algorithm_version') ...
            && strcmp(prior.sourceInfo.algorithm_version, cfg.algorithm_version) ...
            && prior.sourceInfo.source_size_bytes == fileRow.size_bytes(1) ...
            && abs(prior.sourceInfo.source_modified_datenum - fileRow.modified_datenum(1)) < 1e-9
        recordFlags = prior.recordFlags;
        sourceInfo = prior.sourceInfo;
        wasComputed = false;
        return
    end
end

energyField = sprintf('mode%d', mode);
if ~isfield(energyByMode, energyField)
    error('No center-energy vector is available for mode %d.', mode);
end
one = tw1_minpa_quality_flag_all_species_core( ...
    char(fileRow.file_path(1)), energyByMode.(energyField), []);
n = numel(one.time_unix_s);
recordFlags = initialize_record_flags(cfg.species);
recordFlags.time_unix_s = one.time_unix_s;
recordFlags.mode = repmat(mode, n, 1);
for k = 1:numel(cfg.species)
    species = cfg.species{k};
    recordFlags.(species).flag = one.species.(species).flag;
    recordFlags.(species).available = one.species.(species).available;
end

sourceInfo = struct();
sourceInfo.algorithm_version = cfg.algorithm_version;
sourceInfo.source_file = char(fileRow.file_path(1));
sourceInfo.source_size_bytes = fileRow.size_bytes(1);
sourceInfo.source_modified_datenum = fileRow.modified_datenum(1);
sourceInfo.mode = mode;
sourceInfo.records = n;
sourceInfo.time_min_utc = unix_to_utc(min_or_nan(one.time_unix_s));
sourceInfo.time_max_utc = unix_to_utc(max_or_nan(one.time_unix_s));
sourceInfo.flagged_H = nnz(recordFlags.H.flag ~= 0 & recordFlags.H.available);
sourceInfo.flagged_O = nnz(recordFlags.O.flag ~= 0 & recordFlags.O.available);
sourceInfo.flagged_O2 = nnz(recordFlags.O2.flag ~= 0 & recordFlags.O2.available);
sourceInfo.generated_utc = utc_now_string();
payload = struct('recordFlags', recordFlags, 'sourceInfo', sourceInfo, 'cfg', cfg);
atomic_save_struct(checkpointPath, payload, {'recordFlags', 'sourceInfo', 'cfg'});
end

function stage = precompute_all_source_flags(oriCatalog, energyByMode, paths, cfg, overwrite)
n = height(oriCatalog);
status = strings(n, 1);
records = zeros(n, 1);
message = strings(n, 1);
computed = 0;
reused = 0;
failed = 0;
manifestPath = fullfile(paths.output_root, 'ori_flag_checkpoint_manifest.csv');
for f = 1:n
    try
        [~, info, wasComputed] = load_or_compute_source_flags( ...
            oriCatalog(f, :), energyByMode, paths, cfg, overwrite);
        records(f) = info.records;
        if wasComputed
            status(f) = "computed";
            computed = computed + 1;
        else
            status(f) = "reused";
            reused = reused + 1;
        end
    catch ME
        status(f) = "failed";
        message(f) = string(ME.message);
        failed = failed + 1;
        warning('Source flag failure %s: %s', oriCatalog.file_name(f), ME.message);
    end
    if mod(f, 100) == 0 || f == n
        manifest = table(oriCatalog.file_name, oriCatalog.file_path, oriCatalog.mode, ...
            oriCatalog.size_bytes, oriCatalog.modified_datenum, status, records, message, ...
            'VariableNames', {'file_name', 'file_path', 'mode', 'size_bytes', ...
            'modified_datenum', 'status', 'records', 'message'});
        atomic_write_table(manifestPath, manifest);
        fprintf('Source checkpoints %d/%d computed=%d reused=%d failed=%d\n', ...
            f, n, computed, reused, failed);
    end
end
stage = struct('scope', 'all ori files', 'files_total', n, ...
    'computed', computed, 'reused', reused, 'failed', failed, ...
    'manifest', manifestPath);
end

function times = collect_product_times(nvPath, spePath)
nv = load(nvPath, 'NH_TW1');
if ~isfield(nv, 'NH_TW1')
    error('Missing NH_TW1 in %s.', nvPath);
end
times = double(nv.NH_TW1(:, 1));
spe = load(spePath);
topFields = {'H_spe_num', 'O_spe_num', 'O2_spe_num'};
for f = 1:numel(topFields)
    top = topFields{f};
    if ~isfield(spe, top) || ~isstruct(spe.(top))
        continue
    end
    segments = fieldnames(spe.(top));
    for k = 1:numel(segments)
        seg = spe.(top).(segments{k});
        if isfield(seg, 't')
            times = [times; double(seg.t(:))]; %#ok<AGROW>
        end
    end
end
times = sort(unique(times(isfinite(times))));
end

function selected = select_source_files_for_times(catalog, targetTimes, paddingS)
targetTimes = sort(unique(targetTimes(isfinite(targetTimes))));
keep = false(height(catalog), 1);
if isempty(targetTimes)
    selected = catalog(keep, :);
    return
end
breaks = [1; find(diff(targetTimes) > 3600) + 1; numel(targetTimes) + 1];
for k = 1:(numel(breaks) - 1)
    cluster = targetTimes(breaks(k):(breaks(k + 1) - 1));
    startTime = cluster(1) - paddingS;
    stopTime = cluster(end) + paddingS;
    keep = keep | (catalog.start_unix_s <= stopTime ...
        & catalog.stop_unix_s >= startTime & catalog.mode ~= 13);
end
selected = catalog(keep, :);
end

function tf = products_are_current(nvPath, spePath, algorithmVersion)
tf = false;
try
    nvNames = string({whos('-file', nvPath).name});
    speNames = string({whos('-file', spePath).name});
    if ~ismember("MINPA_quality_flag_info", nvNames) ...
            || ~ismember("MINPA_quality_flag_info", speNames)
        return
    end
    nv = load(nvPath, 'MINPA_quality_flag_info');
    spe = load(spePath, 'MINPA_quality_flag_info');
    requiredNv = {'quality_flag_H_TW1', 'quality_flag_O_TW1', 'quality_flag_O2_TW1', ...
        'quality_flag_bit_H_TW1', 'quality_flag_bit_O_TW1', 'quality_flag_bit_O2_TW1'};
    tf = isfield(nv, 'MINPA_quality_flag_info') ...
        && isfield(spe, 'MINPA_quality_flag_info') ...
        && isfield(nv.MINPA_quality_flag_info, 'algorithm_version') ...
        && isfield(spe.MINPA_quality_flag_info, 'algorithm_version') ...
        && strcmp(nv.MINPA_quality_flag_info.algorithm_version, algorithmVersion) ...
        && strcmp(spe.MINPA_quality_flag_info.algorithm_version, algorithmVersion) ...
        && all(ismember(string(requiredNv), nvNames));
catch
    tf = false;
end
end

function [nvStats, speStats] = update_products(nvPath, spePath, recordFlags, dayInfo, cfg, dryRun)
nv = load(nvPath);
spe = load(spePath);
if ~isfield(nv, 'NH_TW1')
    error('Missing NH_TW1 time base in %s.', nvPath);
end
nvTimes = double(nv.NH_TW1(:, 1));
if isfield(nv, 'mode_TW1') && size(nv.mode_TW1, 1) == numel(nvTimes)
    nvModes = double(nv.mode_TW1(:, 2));
else
    nvModes = nan(size(nvTimes));
end
nvStats = struct();

for k = 1:numel(cfg.species)
    species = cfg.species{k};
    mapped = map_species_flags(recordFlags, species, nvTimes, nvModes, cfg);
    nv.(sprintf('quality_flag_%s_TW1', species)) = [nvTimes, double(mapped.flag)];
    nv.(sprintf('quality_flag_bit_%s_TW1', species)) = [nvTimes, double(mapped.bits)];
    nv.(sprintf('quality_flag_binary_%s_TW1', species)) = mapped.binary;
    nv.(sprintf('quality_flag_available_%s_TW1', species)) = [nvTimes, double(mapped.available)];
    nvStats.(species) = struct('records', numel(nvTimes), ...
        'matched', nnz(mapped.matched), 'unmatched', nnz(~mapped.matched), ...
        'available', nnz(mapped.available));
end

info = product_metadata(dayInfo, cfg);
nv.MINPA_quality_flag_info = info;

topNames = struct('H', 'H_spe_num', 'O', 'O_spe_num', 'O2', 'O2_spe_num');
speStats = struct();
for k = 1:numel(cfg.species)
    species = cfg.species{k};
    topName = topNames.(species);
    stats = struct('records', 0, 'matched', 0, 'unmatched', 0, ...
        'available', 0, 'segments', 0);
    if ~isfield(spe, topName) || ~isstruct(spe.(topName))
        warning('Missing %s in %s.', topName, spePath);
        speStats.(species) = stats;
        continue
    end
    segmentNames = fieldnames(spe.(topName));
    for j = 1:numel(segmentNames)
        segmentName = segmentNames{j};
        seg = spe.(topName).(segmentName);
        if ~isfield(seg, 't')
            continue
        end
        times = double(seg.t(:));
        if isfield(seg, 'mod')
            targetModes = repmat(double(seg.mod), numel(times), 1);
        else
            targetModes = nan(size(times));
        end
        mapped = map_species_flags(recordFlags, species, times, targetModes, cfg);
        seg.quality_flag = mapped.flag;
        seg.quality_flag_bit = mapped.bits;
        seg.quality_flag_binary = mapped.binary;
        seg.quality_flag_available = mapped.available;
        seg.quality_flag_species = cfg.species_labels{k};
        spe.(topName).(segmentName) = seg;
        stats.records = stats.records + numel(times);
        stats.matched = stats.matched + nnz(mapped.matched);
        stats.unmatched = stats.unmatched + nnz(~mapped.matched);
        stats.available = stats.available + nnz(mapped.available);
        stats.segments = stats.segments + 1;
    end
    speStats.(species) = stats;
end
spe.MINPA_quality_flag_info = info;

if dryRun
    return
end
atomic_save_struct(nvPath, nv, { ...
    'quality_flag_H_TW1', 'quality_flag_O_TW1', 'quality_flag_O2_TW1', ...
    'quality_flag_bit_H_TW1', 'quality_flag_bit_O_TW1', 'quality_flag_bit_O2_TW1', ...
    'MINPA_quality_flag_info'});
atomic_save_struct(spePath, spe, {'H_spe_num', 'O_spe_num', 'O2_spe_num', ...
    'MINPA_quality_flag_info'});
end

function mapped = map_species_flags(recordFlags, species, targetTimes, targetModes, cfg)
sourceTimes = double(recordFlags.time_unix_s(:));
targetTimes = double(targetTimes(:));
targetModes = double(targetModes(:));
nTarget = numel(targetTimes);
mapped = struct();
mapped.flag = zeros(nTarget, 1, 'uint32');
mapped.bits = false(nTarget, 5);
mapped.binary = repmat(string(cfg.missing_binary), nTarget, 1);
mapped.available = false(nTarget, 1);
mapped.matched = false(nTarget, 1);
if isempty(sourceTimes) || isempty(targetTimes)
    return
end

idx = ones(nTarget, 1);
matched = false(nTarget, 1);
finiteModes = unique(targetModes(isfinite(targetModes)));
for mode = finiteModes'
    targetMask = targetModes == mode;
    sourceIndex = find(recordFlags.mode == mode);
    [idx(targetMask), matched(targetMask)] = nearest_source_indices( ...
        sourceTimes, sourceIndex, targetTimes(targetMask), cfg.time_match_tolerance_s);
end
unknownMode = ~isfinite(targetModes);
if any(unknownMode)
    [~, firstPerTime] = unique(int64(round(sourceTimes * 1e6)), 'stable');
    [idx(unknownMode), matched(unknownMode)] = nearest_source_indices( ...
        sourceTimes, firstPerTime, targetTimes(unknownMode), cfg.time_match_tolerance_s);
end
available = false(nTarget, 1);
available(matched) = recordFlags.(species).available(idx(matched));
usable = matched & available;
mapped.flag(usable) = recordFlags.(species).flag(idx(usable));
for bitIndex = 1:5
    mapped.bits(:, bitIndex) = logical(bitget(mapped.flag, bitIndex));
end
mapped.binary(usable) = binary_from_flags(mapped.flag(usable));
mapped.available = available;
mapped.matched = matched;
end

function [globalIndex, matched] = nearest_source_indices(sourceTimes, sourceIndex, targetTimes, toleranceS)
nTarget = numel(targetTimes);
globalIndex = ones(nTarget, 1);
matched = false(nTarget, 1);
if isempty(sourceIndex) || nTarget == 0
    return
end
localTimes = sourceTimes(sourceIndex);
if numel(localTimes) == 1
    localIndex = ones(nTarget, 1);
else
    origin = localTimes(1);
    localIndex = round(interp1(localTimes - origin, (1:numel(localTimes))', ...
        targetTimes - origin, 'nearest', 'extrap'));
    localIndex = max(1, min(numel(localTimes), localIndex));
end
globalIndex = sourceIndex(localIndex);
matched = abs(sourceTimes(globalIndex) - targetTimes) <= toleranceS;
end

function binary = binary_from_flags(flag)
lookup = strings(32, 1);
for value = 0:31
    bits = bitget(uint32(value), 1:5);
    lookup(value + 1) = string(sprintf('%d%d%d%d%d', bits));
end
binary = lookup(double(flag) + 1);
end

function info = product_metadata(dayInfo, cfg)
info = struct();
info.algorithm_version = cfg.algorithm_version;
info.generated_utc = utc_now_string();
info.species = cfg.species_labels;
info.raw_input_policy = 'D:\Data\TW-1\result\MINPA\ori is read-only';
info.source_files = dayInfo.source_files;
info.flag_encoding = ['Five-bit uint32 bitmask in problem order. ', ...
    'bit1=valid raw 2D channels 5-10; bit2=<5; ', ...
    'bit3=>80% 1D DEF above 1e5; bit4=odd/even peak-valley error ', ...
    'in any of seven energy intervals; bit5=any 2D DEF above 1e10.'];
info.bit4_energy_intervals_eV = cfg.thresholds.alternating_energy_bands_eV;
info.bit4_energy_interval_labels = cfg.thresholds.alternating_energy_band_labels;
info.thresholds = cfg.thresholds;
info.time_match_tolerance_s = cfg.time_match_tolerance_s;
info.missing_binary = cfg.missing_binary;
info.missing_semantics = ['quality_flag_available=false and binary=99999; ', ...
    'numeric flag is stored as 0 only as a placeholder and is not a valid quality judgment.'];
info.nv_fields = { ...
    'quality_flag_H_TW1', 'quality_flag_O_TW1', 'quality_flag_O2_TW1', ...
    'quality_flag_bit_H_TW1', 'quality_flag_bit_O_TW1', 'quality_flag_bit_O2_TW1', ...
    'quality_flag_binary_H_TW1', 'quality_flag_binary_O_TW1', 'quality_flag_binary_O2_TW1', ...
    'quality_flag_available_H_TW1', 'quality_flag_available_O_TW1', 'quality_flag_available_O2_TW1'};
end

function catalog = build_ori_catalog(oriDir)
files = dir(fullfile(oriDir, '*.mat'));
fileName = strings(0, 1);
filePath = strings(0, 1);
startUnix = zeros(0, 1);
stopUnix = zeros(0, 1);
mode = zeros(0, 1);
sizeBytes = zeros(0, 1);
modifiedDatenum = zeros(0, 1);
for k = 1:numel(files)
    tokTime = regexp(files(k).name, '_(\d{14})_(\d{14})_', 'tokens', 'once');
    tokMode = regexp(files(k).name, 'MINPA-MOD(\d+)-', 'tokens', 'once');
    if isempty(tokTime) || isempty(tokMode)
        continue
    end
    startTime = datetime(tokTime{1}, InputFormat='yyyyMMddHHmmss', TimeZone='UTC');
    stopTime = datetime(tokTime{2}, InputFormat='yyyyMMddHHmmss', TimeZone='UTC');
    fileName(end + 1, 1) = string(files(k).name); %#ok<AGROW>
    filePath(end + 1, 1) = string(fullfile(files(k).folder, files(k).name)); %#ok<AGROW>
    startUnix(end + 1, 1) = posixtime(startTime); %#ok<AGROW>
    stopUnix(end + 1, 1) = posixtime(stopTime); %#ok<AGROW>
    mode(end + 1, 1) = str2double(tokMode{1}); %#ok<AGROW>
    sizeBytes(end + 1, 1) = files(k).bytes; %#ok<AGROW>
    modifiedDatenum(end + 1, 1) = files(k).datenum; %#ok<AGROW>
end
catalog = table(fileName, filePath, startUnix, stopUnix, mode, sizeBytes, modifiedDatenum, ...
    'VariableNames', {'file_name', 'file_path', 'start_unix_s', 'stop_unix_s', ...
    'mode', 'size_bytes', 'modified_datenum'});
catalog = sortrows(catalog, {'start_unix_s', 'file_name'});
end

function days = select_days(nvDir, dateList)
if isempty(dateList)
    files = dir(fullfile(nvDir, 'NV_*.mat'));
    days = strings(0, 1);
    for k = 1:numel(files)
        tok = regexp(files(k).name, '^NV_(\d{8})\.mat$', 'tokens', 'once');
        if ~isempty(tok)
            days(end + 1, 1) = string(tok{1}); %#ok<AGROW>
        end
    end
    days = sort(unique(days));
else
    days = sort(unique(string(dateList(:))));
end
end

function [startUnix, stopUnix] = day_bounds(day)
startTime = datetime(day, InputFormat='yyyyMMdd', TimeZone='UTC');
startUnix = posixtime(startTime);
stopUnix = posixtime(startTime + days(1));
end

function daySummary = make_day_summary(day, recordFlags, info, productWritten)
daySummary = blank_day_summary();
daySummary.day = day;
daySummary.status = 'flags_computed';
daySummary.records = numel(recordFlags.time_unix_s);
if isfield(info, 'raw_files_scanned')
    daySummary.raw_files = info.raw_files_scanned;
elseif isfield(info, 'raw_files')
    daySummary.raw_files = info.raw_files;
else
    daySummary.raw_files = NaN;
end
if isfield(info, 'duplicates_removed')
    daySummary.duplicates_removed = info.duplicates_removed;
elseif isfield(info, 'duplicate_records_removed')
    daySummary.duplicates_removed = info.duplicate_records_removed;
end
if isfield(info, 'duplicate_flag_conflicts')
    daySummary.duplicate_flag_conflicts = info.duplicate_flag_conflicts;
end
daySummary.flagged_H = nnz(recordFlags.H.flag ~= 0 & recordFlags.H.available);
daySummary.flagged_O = nnz(recordFlags.O.flag ~= 0 & recordFlags.O.available);
daySummary.flagged_O2 = nnz(recordFlags.O2.flag ~= 0 & recordFlags.O2.available);
daySummary.product_written = logical(productWritten);
end

function out = empty_day_summary()
out = repmat(blank_day_summary(), 0, 1);
end

function out = blank_day_summary()
out = struct('day', '', 'status', '', 'records', 0, 'raw_files', 0, ...
    'duplicates_removed', 0, 'duplicate_flag_conflicts', 0, ...
    'flagged_H', 0, 'flagged_O', 0, 'flagged_O2', 0, ...
    'nv_unmatched_H', 0, 'nv_unmatched_O', 0, 'nv_unmatched_O2', 0, ...
    'spe_unmatched_H', 0, 'spe_unmatched_O', 0, 'spe_unmatched_O2', 0, ...
    'product_written', false, 'message', '');
end

function out = failed_day_summary(day, ME)
out = blank_day_summary();
out.day = day;
out.status = 'failed';
out.message = getReport(ME, 'extended', 'hyperlinks', 'off');
end

function write_root_summary(summary, outputRoot)
payload = struct('summary', summary);
atomic_save_struct(fullfile(outputRoot, 'quality_flag_all_species_summary.mat'), ...
    payload, {'summary'});
if ~isempty(summary.days)
    tbl = struct2table(summary.days, 'AsArray', true);
    csvPath = fullfile(outputRoot, 'quality_flag_all_species_manifest.csv');
    atomic_write_table(csvPath, tbl);
end
end

function atomic_write_table(finalPath, tbl)
folder = fileparts(finalPath);
if ~exist(folder, 'dir')
    mkdir(folder);
end
tempPath = [tempname(folder), '.csv'];
cleanup = onCleanup(@() delete_if_exists(tempPath));
writetable(tbl, tempPath);
if ~isfile(tempPath) || dir(tempPath).bytes == 0
    error('Temporary table write failed: %s.', tempPath);
end
replace_file_with_retry(tempPath, finalPath);
clear cleanup
end

function atomic_save_struct(finalPath, payload, requiredVariables)
folder = fileparts(finalPath);
if ~exist(folder, 'dir')
    mkdir(folder);
end
tempPath = [tempname(folder), '.mat'];
cleanup = onCleanup(@() delete_if_exists(tempPath));
save(tempPath, '-struct', 'payload');
if ~isfile(tempPath) || dir(tempPath).bytes == 0
    error('Temporary MAT write failed: %s.', tempPath);
end
names = string({whos('-file', tempPath).name});
if ~all(ismember(string(requiredVariables), names))
    error('Temporary MAT schema validation failed for %s.', finalPath);
end
replace_file_with_retry(tempPath, finalPath);
clear cleanup
end

function replace_file_with_retry(tempPath, finalPath)
lastMessage = '';
for attempt = 1:30
    try
        [ok, message] = movefile(tempPath, finalPath, 'f');
    catch ME
        ok = false;
        message = ME.message;
    end
    if ok
        return
    end
    lastMessage = message;
    pause(min(0.1 * attempt, 1));
end
error('Could not replace %s after retries: %s', finalPath, lastMessage);
end

function delete_if_exists(path)
if isfile(path)
    delete(path);
end
end

function value = ternary(condition, trueValue, falseValue)
if condition
    value = trueValue;
else
    value = falseValue;
end
end

function value = utc_now_string()
value = char(datetime('now', TimeZone='UTC', Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
end

function value = unix_to_utc(timeUnix)
if isempty(timeUnix) || ~isfinite(timeUnix)
    value = '';
else
    value = char(datetime(timeUnix, ConvertFrom='posixtime', TimeZone='UTC', ...
        Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
end
end

function value = min_or_nan(x)
if isempty(x)
    value = NaN;
else
    value = min(x, [], 'omitnan');
end
end

function value = max_or_nan(x)
if isempty(x)
    value = NaN;
else
    value = max(x, [], 'omitnan');
end
end
