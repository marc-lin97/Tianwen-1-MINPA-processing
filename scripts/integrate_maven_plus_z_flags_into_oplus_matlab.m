function integrate_maven_plus_z_flags_into_oplus_matlab(rootDir, workers, overwrite, dryRun)
%INTEGRATE_MAVEN_PLUS_Z_FLAGS_INTO_OPLUS_MATLAB Append +Z_MSE FOV flags to O+ MAT files.
%
% Usage from MATLAB batch:
%   addpath('C:\Users\Administrator\Documents\-E_highE_O+\scripts');
%   integrate_maven_plus_z_flags_into_oplus_matlab('E:\Data\highE\MAVEN', 8, false, false);
%
% For each date, this function checks that:
%   maven_static_highE_Oplus_YYYYMMDD.mat epoch_unix_s
% matches:
%   maven_static_plus_z_fov_flag_YYYYMMDD.mat epoch_unix_s
% Then it appends the flag variables to the Oplus file with save(..., '-append').

if nargin < 1 || isempty(rootDir)
    rootDir = 'E:\Data\highE\MAVEN';
end
if nargin < 2 || isempty(workers)
    workers = 0;
end
if nargin < 3 || isempty(overwrite)
    overwrite = false;
end
if nargin < 4 || isempty(dryRun)
    dryRun = true;
end

flagFields = { ...
    'mse_plus_z_in_static_fov_flag', ...
    'mse_plus_z_static_theta_deg', ...
    'mse_plus_z_static_x', ...
    'mse_plus_z_static_y', ...
    'mse_plus_z_static_z', ...
    'static_fov_theta_min_deg', ...
    'static_fov_theta_max_deg'};

jobs = discover_jobs(rootDir);
fprintf('Discovered %d Oplus products under %s\n', numel(jobs), rootDir);

if workers > 0
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= workers
        if ~isempty(pool)
            delete(pool);
        end
        parpool('local', workers);
    end
end

results = repmat(empty_result(), numel(jobs), 1);
if workers > 0
    parfor i = 1:numel(jobs)
        results(i) = integrate_one(jobs(i), flagFields, overwrite, dryRun);
    end
else
    for i = 1:numel(jobs)
        results(i) = integrate_one(jobs(i), flagFields, overwrite, dryRun);
    end
end

summary = summarize_results(results);
disp(summary.counts);

logDir = fullfile(rootDir, 'logs');
if ~exist(logDir, 'dir')
    mkdir(logDir);
end
stamp = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyyMMdd''T''HHmmss''Z'''));
summaryPath = fullfile(logDir, ['maven_plus_z_flag_integration_matlab_' stamp '.json']);
fid = fopen(summaryPath, 'w');
if fid < 0
    error('Could not open summary path for writing: %s', summaryPath);
end
fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
fclose(fid);
fprintf('Summary written: %s\n', summaryPath);
end


function jobs = discover_jobs(rootDir)
yearDirs = dir(rootDir);
jobs = struct('date', {}, 'oplusPath', {}, 'flagPath', {});
for i = 1:numel(yearDirs)
    item = yearDirs(i);
    if ~item.isdir || isempty(regexp(item.name, '^\d{4}$', 'once'))
        continue;
    end
    yearPath = fullfile(rootDir, item.name);
    files = dir(fullfile(yearPath, 'maven_static_highE_Oplus_*.mat'));
    for j = 1:numel(files)
        tok = regexp(files(j).name, 'maven_static_highE_Oplus_(\d{8})\.mat$', 'tokens', 'once');
        if isempty(tok)
            continue;
        end
        dateToken = tok{1};
        jobs(end+1).date = dateToken; %#ok<AGROW>
        jobs(end).oplusPath = fullfile(yearPath, files(j).name);
        jobs(end).flagPath = fullfile(yearPath, ['maven_static_plus_z_fov_flag_' dateToken '.mat']);
    end
end
[~, order] = sort({jobs.date});
jobs = jobs(order);
end


function result = integrate_one(job, flagFields, overwrite, dryRun)
result = empty_result();
result.date = job.date;
result.oplusPath = job.oplusPath;
result.flagPath = job.flagPath;

if ~exist(job.oplusPath, 'file')
    result.status = 'missing_oplus';
    return;
end
if ~exist(job.flagPath, 'file')
    result.status = 'missing_flag';
    return;
end

try
    oInfo = whos('-file', job.oplusPath);
    oNames = {oInfo.name};
    if all(ismember(flagFields, oNames)) && ~overwrite
        result.status = 'already_has_fields';
        return;
    end

    oEpoch = load(job.oplusPath, 'epoch_unix_s');
    f = load(job.flagPath, 'epoch_unix_s', flagFields{:});
catch ME
    result.status = 'read_failed';
    result.message = ME.message;
    return;
end

if ~isfield(oEpoch, 'epoch_unix_s') || ~isfield(f, 'epoch_unix_s')
    result.status = 'missing_epoch';
    return;
end

left = double(oEpoch.epoch_unix_s(:));
right = double(f.epoch_unix_s(:));
result.rows = numel(left);
if numel(left) ~= numel(right) || any(left ~= right)
    result.status = 'epoch_mismatch';
    result.flagRows = numel(right);
    return;
end

for k = 1:numel(flagFields)
    if ~isfield(f, flagFields{k})
        result.status = 'missing_flag_field';
        result.message = flagFields{k};
        return;
    end
end

payload = struct();
for k = 1:numel(flagFields)
    payload.(flagFields{k}) = f.(flagFields{k});
end

if dryRun
    result.status = 'would_write';
    return;
end

try
    save(job.oplusPath, '-struct', 'payload', '-append');
    result.status = 'written';
catch ME
    result.status = 'write_failed';
    result.message = ME.message;
end
end


function result = empty_result()
result = struct( ...
    'date', '', ...
    'status', '', ...
    'rows', 0, ...
    'flagRows', 0, ...
    'oplusPath', '', ...
    'flagPath', '', ...
    'message', '');
end


function summary = summarize_results(results)
statuses = {results.status};
[uniqueStatuses, ~, idx] = unique(statuses);
counts = struct();
for i = 1:numel(uniqueStatuses)
    name = matlab.lang.makeValidName(uniqueStatuses{i});
    counts.(name) = sum(idx == i);
end
summary = struct();
summary.created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
summary.counts = counts;
summary.total = numel(results);
summary.results = results;
end
