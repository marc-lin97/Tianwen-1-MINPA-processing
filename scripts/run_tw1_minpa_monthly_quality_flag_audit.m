function summary = run_tw1_minpa_monthly_quality_flag_audit(outputDir)
%RUN_TW1_MINPA_MONTHLY_QUALITY_FLAG_AUDIT Audit one MINPA day per month.
%
% This driver keeps the strict quality-flag algorithm in
% audit_tw1_minpa_quality_flags unchanged. It selects one available ori day
% per month, computes H+ quality flags for all records on those days, selects
% up to five representative examples for each issue bit, and plots +/-5 min
% context figures using the project day_spe spectrogram style.

if nargin < 1 || isempty(outputDir)
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_monthly_sample_strict');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

oriDir = 'D:\Data\TW-1\result\MINPA\ori';
dateTable = select_one_available_day_per_month(oriDir);
dateCsv = fullfile(outputDir, 'tw1_minpa_monthly_sample_dates.csv');
writetable(dateTable, dateCsv);

dateList = cellstr(string(dateTable.sample_date));
auditSummary = audit_tw1_minpa_quality_flags(dateList, inf, false, false, outputDir);

recordsCsv = auditSummary.audit_csv;
examplesCsv = fullfile(outputDir, 'tw1_minpa_hplus_quality_flag_issue_examples.csv');
[examples, counts] = select_issue_examples(recordsCsv, examplesCsv, 5);

contextDir = fullfile(outputDir, 'issue_classes_1_5_context_5min');
plotSummary = plot_tw1_minpa_quality_flag_context_examples(examplesCsv, 5, contextDir);

summary = struct();
summary.output_dir = outputDir;
summary.monthly_dates_csv = dateCsv;
summary.monthly_dates = dateTable;
summary.audit_summary = auditSummary;
summary.records_csv = recordsCsv;
summary.examples_csv = examplesCsv;
summary.example_counts = counts;
summary.context_dir = contextDir;
summary.plot_summary = plotSummary;
summary.notes = [
    "Sample date selection is deterministic: the middle available start date in each yyyyMM group."
    "Raw ori, day_spe, and NV_MSO_2 products are read-only in this workflow."
    "Example selection is severity-ranked within each bit, with a >=5 minute separation when enough candidates exist."
];

summaryMat = fullfile(outputDir, 'tw1_minpa_monthly_quality_flag_audit_summary.mat');
save(summaryMat, 'summary');
summary.summary_mat = summaryMat;

fprintf('Monthly sample dates: %s\n', dateCsv);
fprintf('Issue examples: %s\n', examplesCsv);
fprintf('Context figures: %s\n', contextDir);
end

function dateTable = select_one_available_day_per_month(oriDir)
files = dir(fullfile(oriDir, '*.mat'));
starts = strings(0, 1);
for i = 1:numel(files)
    toks = regexp(files(i).name, '_(\d{8})\d{6}_(\d{8})\d{6}_', 'tokens', 'once');
    if isempty(toks)
        continue
    end
    starts(end + 1, 1) = string(toks{1}); %#ok<AGROW>
end
starts = unique(starts);
months = unique(extractBefore(starts, 7));

sampleMonth = strings(numel(months), 1);
sampleDate = strings(numel(months), 1);
availableDateCount = zeros(numel(months), 1);
for m = 1:numel(months)
    monthDays = sort(starts(startsWith(starts, months(m))));
    availableDateCount(m) = numel(monthDays);
    sampleMonth(m) = months(m);
    sampleDate(m) = monthDays(ceil(numel(monthDays) / 2));
end

dateTable = table(sampleMonth, sampleDate, availableDateCount, ...
    'VariableNames', {'sample_month', 'sample_date', 'available_start_date_count'});
end

function [examples, counts] = select_issue_examples(recordsCsv, examplesCsv, maxPerClass)
opts = detectImportOptions(recordsCsv);
stringVars = intersect({'source_file', 'time_utc', 'quality_flag_binary'}, opts.VariableNames);
if ~isempty(stringVars)
    opts = setvartype(opts, stringVars, 'string');
end
records = readtable(recordsCsv, opts);
records = add_evenodd_selection_scores(records);

classes = [
    struct('name', "sparse_caution", 'bitColumn', "flag_bit1_sparse_caution", 'severityColumn', "valid_2d_channel_count", 'direction', "ascend")
    struct('name', "sparse_invalid", 'bitColumn', "flag_bit2_sparse_invalid", 'severityColumn', "valid_2d_channel_count", 'direction', "ascend")
    struct('name', "uv_contamination", 'bitColumn', "flag_bit3_uv_contamination", 'severityColumn', "uv_fraction", 'direction', "descend")
    struct('name', "evenodd_error", 'bitColumn', "flag_bit4_evenodd_error", 'severityColumn', "evenodd_selection_extrema_fraction", 'direction', "descend")
    struct('name', "high_channel", 'bitColumn', "flag_bit5_high_channel", 'severityColumn', "max_def_2d", 'direction', "descend")
];

examples = table();
className = strings(numel(classes), 1);
candidateCount = zeros(numel(classes), 1);
selectedCount = zeros(numel(classes), 1);
for c = 1:numel(classes)
    cls = classes(c);
    className(c) = cls.name;
    mask = logical(records.(cls.bitColumn));
    candidates = records(mask, :);
    candidateCount(c) = height(candidates);
    if isempty(candidates)
        continue
    end
    candidates = add_selection_columns(candidates, cls.name, cls.severityColumn);
    candidates = sortrows(candidates, {'selection_score', 'time_unix_s'}, {char(cls.direction), 'ascend'});
    chosen = choose_with_min_separation(candidates, maxPerClass, 5 * 60);
    selectedCount(c) = height(chosen);
    examples = [examples; chosen]; %#ok<AGROW>
end

counts = table(className, candidateCount, selectedCount, ...
    'VariableNames', {'category', 'candidate_count', 'selected_count'});
countCsv = fullfile(fileparts(examplesCsv), 'tw1_minpa_hplus_quality_flag_issue_example_counts.csv');
writetable(counts, countCsv);

if ~isempty(examples)
    front = {'category'};
    remaining = setdiff(examples.Properties.VariableNames, front, 'stable');
    examples = examples(:, [front, remaining]);
    examples.selection_rank = (1:height(examples))';
end
writetable(examples, examplesCsv);
end

function records = add_evenodd_selection_scores(records)
extCols = {'alternating_extrema_fraction', ...
    'alternating_extrema_fraction_0_30eV', ...
    'alternating_extrema_fraction_30_300eV', ...
    'alternating_extrema_fraction_300_maxeV', ...
    'alternating_extrema_fraction_0_50eV', ...
    'alternating_extrema_fraction_50_500eV', ...
    'alternating_extrema_fraction_500_5000eV'};
records.evenodd_selection_extrema_fraction = max_existing_columns(records, extCols);
end

function score = max_existing_columns(records, columns)
score = nan(height(records), 1);
for i = 1:numel(columns)
    if any(strcmp(records.Properties.VariableNames, columns{i}))
        score = max([score, double(records.(columns{i}))], [], 2, 'omitnan');
    end
end
end

function candidates = add_selection_columns(candidates, category, severityColumn)
candidates.category = repmat(string(category), height(candidates), 1);
score = candidates.(severityColumn);
if islogical(score)
    score = double(score);
end
candidates.selection_score = double(score);
end

function chosen = choose_with_min_separation(candidates, maxRows, minSeparationSeconds)
chosen = candidates([], :);
if height(candidates) <= maxRows
    chosen = candidates;
    return
end

for i = 1:height(candidates)
    if height(chosen) >= maxRows
        break
    end
    t = double(candidates.time_unix_s(i));
    if isempty(chosen) || all(abs(double(chosen.time_unix_s) - t) >= minSeparationSeconds)
        chosen = [chosen; candidates(i, :)]; %#ok<AGROW>
    end
end

if height(chosen) < maxRows
    for i = 1:height(candidates)
        if height(chosen) >= maxRows
            break
        end
        t = double(candidates.time_unix_s(i));
        if isempty(chosen) || all(abs(double(chosen.time_unix_s) - t) > 0)
            already = chosen.time_unix_s == candidates.time_unix_s(i) ...
                & chosen.record_index == candidates.record_index(i) ...
                & chosen.source_file == candidates.source_file(i);
            if ~any(already)
                chosen = [chosen; candidates(i, :)]; %#ok<AGROW>
            end
        end
    end
end
end
