function summary = run_tw1_minpa_quality_flags_all(dateList, dryRun, maxDays)
%RUN_TW1_MINPA_QUALITY_FLAGS_ALL Compute and attach MINPA quality flags by day.
%
% This driver processes days independently to avoid keeping all MINPA records
% in memory. It first computes H+ flags from read-only ori files, then calls
% apply_tw1_minpa_quality_flags_from_audit for that day.
%
% Defaults are conservative:
%   dateList = {}  -> all days found in NV_MSO_2
%   dryRun   = true -> do not modify NV_MSO_2 or day_spe
%   maxDays  = inf
%
% Example dry-run for one day:
%   run_tw1_minpa_quality_flags_all({'20211203'}, true, 1)
%
% Example production run after audit approval:
%   run_tw1_minpa_quality_flags_all({}, false, inf)

if nargin < 1 || isempty(dateList)
    dateList = {};
end
if nargin < 2 || isempty(dryRun)
    dryRun = true;
end
if nargin < 3 || isempty(maxDays)
    maxDays = inf;
end
if ischar(dateList) || isstring(dateList)
    dateList = cellstr(dateList);
end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
paths.nvMso2Dir = 'D:\Data\TW-1\result\MINPA\NV_MSO_2';
paths.outputRoot = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flags_full');
if ~exist(paths.outputRoot, 'dir')
    mkdir(paths.outputRoot);
end

days = select_days(paths.nvMso2Dir, dateList);
if isfinite(maxDays)
    days = days(1:min(numel(days), maxDays));
end

summary = struct();
summary.output_root = paths.outputRoot;
summary.dry_run = dryRun;
summary.days_requested = days;
summary.days = struct('day', {}, 'status', {}, 'records', {}, ...
    'audit_csv', {}, 'apply_summary', {}, 'message', {});

for d = 1:numel(days)
    day = char(days(d));
    dayOut = fullfile(paths.outputRoot, day);
    if ~exist(dayOut, 'dir')
        mkdir(dayOut);
    end
    fprintf('=== %s dryRun=%d ===\n', day, dryRun);
    dayStatus = 'ok';
    message = '';
    auditSummary = struct();
    applySummary = struct();
    try
        auditSummary = audit_tw1_minpa_quality_flags({day}, inf, false, false, dayOut);
        applySummary = apply_tw1_minpa_quality_flags_from_audit(auditSummary.audit_csv, {day}, dryRun);
    catch ME
        dayStatus = 'failed';
        message = getReport(ME, 'extended', 'hyperlinks', 'off');
        warning('Failed %s: %s', day, ME.message);
    end

    idx = numel(summary.days) + 1;
    summary.days(idx).day = day;
    summary.days(idx).status = dayStatus;
    if isfield(auditSummary, 'records_scanned')
        summary.days(idx).records = auditSummary.records_scanned;
    else
        summary.days(idx).records = NaN;
    end
    if isfield(auditSummary, 'audit_csv')
        summary.days(idx).audit_csv = auditSummary.audit_csv;
    else
        summary.days(idx).audit_csv = '';
    end
    summary.days(idx).apply_summary = applySummary;
    summary.days(idx).message = message;

    daySummaryPath = fullfile(dayOut, 'quality_flag_day_summary.mat');
    save(daySummaryPath, 'auditSummary', 'applySummary', 'dayStatus', 'message');
    rootSummaryPath = fullfile(paths.outputRoot, 'quality_flag_all_summary.mat');
    save(rootSummaryPath, 'summary');
end

fprintf('Wrote summary under %s\n', paths.outputRoot);
end

function days = select_days(nvMso2Dir, dateList)
if isempty(dateList)
    files = dir(fullfile(nvMso2Dir, 'NV_*.mat'));
    days = strings(numel(files), 1);
    keep = false(numel(files), 1);
    for k = 1:numel(files)
        tok = regexp(files(k).name, 'NV_(\d{8})\.mat', 'tokens', 'once');
        if isempty(tok)
            continue
        end
        days(k) = string(tok{1});
        keep(k) = true;
    end
    days = sort(days(keep));
else
    days = sort(string(dateList(:)));
end
end
