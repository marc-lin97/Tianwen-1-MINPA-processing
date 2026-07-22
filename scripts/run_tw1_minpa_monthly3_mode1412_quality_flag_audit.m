function summary = run_tw1_minpa_monthly3_mode1412_quality_flag_audit(outputDir)
%RUN_TW1_MINPA_MONTHLY3_MODE1412_QUALITY_FLAG_AUDIT Audit 3 days/month.
%
% Selects up to three available days per month from MINPA ori mode 1, 4, and
% 12 products, computes H+ quality flags, selects representative issue
% examples, and plots +/-5 minute context figures. Raw products are read-only.

if nargin < 1 || isempty(outputDir)
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_monthly3_mode1412');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

oriDir = 'D:\Data\TW-1\result\MINPA\ori';
allowedModes = [1, 4, 12];
dateTable = select_three_available_days_per_month(oriDir, allowedModes);
dateCsv = fullfile(outputDir, 'tw1_minpa_monthly3_mode1412_sample_dates.csv');
writetable(dateTable, dateCsv);

dateList = cellstr(string(unique(dateTable.sample_date, 'stable')));
auditSummary = audit_tw1_minpa_quality_flags(dateList, inf, false, false, outputDir, allowedModes);

recordsCsv = auditSummary.audit_csv;
examplesCsv = fullfile(outputDir, 'tw1_minpa_hplus_quality_flag_issue_examples.csv');
[examples, counts] = select_issue_examples(recordsCsv, examplesCsv);

contextDir = fullfile(outputDir, 'issue_classes_1_5_context_5min');
plotSummary = plot_tw1_minpa_quality_flag_context_examples(examplesCsv, 5, contextDir);

summary = struct();
summary.output_dir = outputDir;
summary.allowed_modes = allowedModes;
summary.monthly_dates_csv = dateCsv;
summary.monthly_dates = dateTable;
summary.audit_summary = auditSummary;
summary.records_csv = recordsCsv;
summary.examples_csv = examplesCsv;
summary.example_counts = counts;
summary.context_dir = contextDir;
summary.plot_summary = plotSummary;
summary.notes = [
    "Mode filter: only MINPA modes 1, 4, and 12 are audited."
    "Sample date selection is deterministic: first/middle/last available start date in each yyyyMM group, after mode filtering."
    "Requested examples: bit1=10, bit2=10, bit3=20, bit4=20, bit5=10; fewer are selected only when fewer candidates exist."
    "Raw ori, day_spe, and NV_MSO_2 products are read-only in this workflow."
];

summaryMat = fullfile(outputDir, 'tw1_minpa_monthly3_mode1412_quality_flag_audit_summary.mat');
save(summaryMat, 'summary');
summary.summary_mat = summaryMat;

fprintf('Monthly sample dates: %s\n', dateCsv);
fprintf('Issue examples: %s\n', examplesCsv);
fprintf('Context figures: %s\n', contextDir);
end

function dateTable = select_three_available_days_per_month(oriDir, allowedModes)
files = dir(fullfile(oriDir, '*.mat'));
starts = strings(0, 1);
startModes = [];
for i = 1:numel(files)
    mode = parse_mode_from_name(files(i).name);
    if ~any(mode == allowedModes)
        continue
    end
    toks = regexp(files(i).name, '_(\d{8})\d{6}_(\d{8})\d{6}_', 'tokens', 'once');
    if isempty(toks)
        continue
    end
    starts(end + 1, 1) = string(toks{1}); %#ok<AGROW>
    startModes(end + 1, 1) = mode; %#ok<AGROW>
end

allDays = unique(starts);
months = unique(extractBefore(allDays, 7));
sampleMonth = strings(0, 1);
sampleDate = strings(0, 1);
sampleIndexInMonth = zeros(0, 1);
availableDateCount = zeros(0, 1);
modesPresent = strings(0, 1);

for m = 1:numel(months)
    monthDays = sort(allDays(startsWith(allDays, months(m))));
    nDays = numel(monthDays);
    if nDays == 0
        continue
    end
    nSelect = min(3, nDays);
    if nSelect == 1
        idx = 1;
    else
        idx = unique(round(linspace(1, nDays, nSelect)), 'stable');
        while numel(idx) < nSelect
            missing = setdiff(1:nDays, idx, 'stable');
            idx(end + 1) = missing(ceil(numel(missing) / 2)); %#ok<AGROW>
            idx = sort(idx);
        end
    end
    for j = 1:numel(idx)
        day = monthDays(idx(j));
        sampleMonth(end + 1, 1) = months(m); %#ok<AGROW>
        sampleDate(end + 1, 1) = day; %#ok<AGROW>
        sampleIndexInMonth(end + 1, 1) = j; %#ok<AGROW>
        availableDateCount(end + 1, 1) = nDays; %#ok<AGROW>
        dayModes = unique(startModes(starts == day));
        modesPresent(end + 1, 1) = strjoin(string(dayModes(:))', ','); %#ok<AGROW>
    end
end

dateTable = table(sampleMonth, sampleDate, sampleIndexInMonth, availableDateCount, modesPresent, ...
    'VariableNames', {'sample_month', 'sample_date', 'sample_index_in_month', ...
    'available_start_date_count', 'modes_present_on_start_date'});
end

function mode = parse_mode_from_name(name)
tok = regexp(name, 'MINPA-MOD(\d+)-', 'tokens', 'once');
if isempty(tok)
    mode = NaN;
else
    mode = str2double(tok{1});
end
end

function [examples, counts] = select_issue_examples(recordsCsv, examplesCsv)
opts = detectImportOptions(recordsCsv);
stringVars = intersect({'source_file', 'time_utc', 'quality_flag_binary'}, opts.VariableNames);
if ~isempty(stringVars)
    opts = setvartype(opts, stringVars, 'string');
end
records = readtable(recordsCsv, opts);
records = add_evenodd_selection_scores(records);

classes = [
    struct('name', "sparse_caution", 'bitColumn', "flag_bit1_sparse_caution", 'severityColumn', "valid_2d_channel_count", 'secondaryColumn', "", 'direction', "ascend", 'maxCount', 10)
    struct('name', "sparse_invalid", 'bitColumn', "flag_bit2_sparse_invalid", 'severityColumn', "valid_2d_channel_count", 'secondaryColumn', "", 'direction', "ascend", 'maxCount', 10)
    struct('name', "uv_contamination", 'bitColumn', "flag_bit3_uv_contamination", 'severityColumn', "uv_fraction", 'secondaryColumn', "", 'direction', "descend", 'maxCount', 20)
    struct('name', "evenodd_error", 'bitColumn', "flag_bit4_evenodd_error", 'severityColumn', "evenodd_selection_extrema_fraction", 'secondaryColumn', "evenodd_selection_phase_fraction", 'direction', "descend", 'maxCount', 20)
    struct('name', "high_channel", 'bitColumn', "flag_bit5_high_channel", 'severityColumn', "max_def_2d", 'secondaryColumn', "", 'direction', "descend", 'maxCount', 10)
];

examples = table();
className = strings(numel(classes), 1);
candidateCount = zeros(numel(classes), 1);
selectedCount = zeros(numel(classes), 1);
requestedCount = zeros(numel(classes), 1);
for c = 1:numel(classes)
    cls = classes(c);
    className(c) = cls.name;
    requestedCount(c) = cls.maxCount;
    mask = logical(records.(cls.bitColumn));
    candidates = records(mask, :);
    candidateCount(c) = height(candidates);
    if isempty(candidates)
        continue
    end
    candidates = add_selection_columns(candidates, cls.name, cls.severityColumn, cls.secondaryColumn);
    if cls.direction == "ascend"
        candidates = sortrows(candidates, {'selection_score', 'selection_secondary_score', 'time_unix_s'}, {'ascend', 'descend', 'ascend'});
    else
        candidates = sortrows(candidates, {'selection_score', 'selection_secondary_score', 'time_unix_s'}, {'descend', 'descend', 'ascend'});
    end
    chosen = choose_with_min_separation(candidates, cls.maxCount, 5 * 60);
    selectedCount(c) = height(chosen);
    examples = [examples; chosen]; %#ok<AGROW>
end

counts = table(className, requestedCount, candidateCount, selectedCount, ...
    'VariableNames', {'category', 'requested_count', 'candidate_count', 'selected_count'});
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
phaseCols = {'alternating_phase_fraction', ...
    'alternating_phase_fraction_0_30eV', ...
    'alternating_phase_fraction_30_300eV', ...
    'alternating_phase_fraction_300_maxeV', ...
    'alternating_phase_fraction_0_50eV', ...
    'alternating_phase_fraction_50_500eV', ...
    'alternating_phase_fraction_500_5000eV'};
records.evenodd_selection_extrema_fraction = max_existing_columns(records, extCols);
records.evenodd_selection_phase_fraction = max_existing_columns(records, phaseCols);
end

function score = max_existing_columns(records, columns)
score = nan(height(records), 1);
for i = 1:numel(columns)
    if any(strcmp(records.Properties.VariableNames, columns{i}))
        score = max([score, double(records.(columns{i}))], [], 2, 'omitnan');
    end
end
end

function candidates = add_selection_columns(candidates, category, severityColumn, secondaryColumn)
candidates.category = repmat(string(category), height(candidates), 1);
score = candidates.(severityColumn);
if islogical(score)
    score = double(score);
end
candidates.selection_score = double(score);
if strlength(string(secondaryColumn)) > 0 && any(strcmp(candidates.Properties.VariableNames, char(secondaryColumn)))
    secondaryScore = candidates.(secondaryColumn);
    if islogical(secondaryScore)
        secondaryScore = double(secondaryScore);
    end
    candidates.selection_secondary_score = double(secondaryScore);
else
    candidates.selection_secondary_score = nan(height(candidates), 1);
end
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
        already = chosen.time_unix_s == candidates.time_unix_s(i) ...
            & chosen.record_index == candidates.record_index(i) ...
            & chosen.source_file == candidates.source_file(i);
        if ~any(already)
            chosen = [chosen; candidates(i, :)]; %#ok<AGROW>
        end
    end
end
end
