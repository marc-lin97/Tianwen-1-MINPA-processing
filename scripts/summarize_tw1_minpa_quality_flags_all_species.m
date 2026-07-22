function summary = summarize_tw1_minpa_quality_flags_all_species(outputRoot)
%SUMMARIZE_TW1_MINPA_QUALITY_FLAGS_ALL_SPECIES Aggregate production checks.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(outputRoot)
    outputRoot = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flags_all_species');
end
cfg = tw1_minpa_quality_flag_config();
checkpointRoot = fullfile(outputRoot, 'ori_flag_checkpoints');
files = dir(fullfile(checkpointRoot, '**', '*_quality_flags.mat'));

nSpecies = numel(cfg.species);
totalRecords = zeros(nSpecies, 1);
availableRecords = zeros(nSpecies, 1);
flaggedRecords = zeros(nSpecies, 1);
bitCounts = zeros(nSpecies, 5);
combinationCounts = zeros(nSpecies, 32);
rawChanged = 0;
loadFailures = 0;
modeValues = [1, 4, 7, 12];
modeFileCounts = zeros(size(modeValues));
modeRecordCounts = zeros(size(modeValues));

for f = 1:numel(files)
    path = fullfile(files(f).folder, files(f).name);
    try
        s = load(path, 'recordFlags', 'sourceInfo');
        n = numel(s.recordFlags.time_unix_s);
        modeIndex = find(modeValues == s.sourceInfo.mode, 1);
        if ~isempty(modeIndex)
            modeFileCounts(modeIndex) = modeFileCounts(modeIndex) + 1;
            modeRecordCounts(modeIndex) = modeRecordCounts(modeIndex) + n;
        end
        raw = dir(s.sourceInfo.source_file);
        if isempty(raw) || raw.bytes ~= s.sourceInfo.source_size_bytes ...
                || abs(raw.datenum - s.sourceInfo.source_modified_datenum) >= 1e-9
            rawChanged = rawChanged + 1;
        end
        for k = 1:nSpecies
            species = cfg.species{k};
            flag = s.recordFlags.(species).flag;
            available = s.recordFlags.(species).available;
            totalRecords(k) = totalRecords(k) + numel(flag);
            availableRecords(k) = availableRecords(k) + nnz(available);
            flaggedRecords(k) = flaggedRecords(k) + nnz(available & flag ~= 0);
            for bitIndex = 1:5
                bitCounts(k, bitIndex) = bitCounts(k, bitIndex) ...
                    + nnz(available & bitget(flag, bitIndex));
            end
            combinationCounts(k, :) = combinationCounts(k, :) ...
                + accumarray(double(flag(available)) + 1, 1, [32, 1])';
        end
    catch ME
        loadFailures = loadFailures + 1;
        warning('Could not summarize %s: %s', path, ME.message);
    end
    if mod(f, 1000) == 0
        fprintf('Summarized %d/%d source checkpoints.\n', f, numel(files));
    end
end

speciesLabel = string(cfg.species_labels(:));
speciesSummary = table(speciesLabel, totalRecords, availableRecords, flaggedRecords, ...
    bitCounts(:, 1), bitCounts(:, 2), bitCounts(:, 3), bitCounts(:, 4), bitCounts(:, 5), ...
    'VariableNames', {'species', 'records', 'available_records', 'flagged_records', ...
    'bit1_sparse_caution', 'bit2_sparse_invalid', 'bit3_uv_contamination', ...
    'bit4_evenodd_error', 'bit5_high_channel'});
writetable(speciesSummary, fullfile(outputRoot, 'quality_flag_species_summary.csv'));

species = strings(nSpecies * 32, 1);
flagValue = zeros(nSpecies * 32, 1);
count = zeros(nSpecies * 32, 1);
row = 0;
for k = 1:nSpecies
    for value = 0:31
        row = row + 1;
        species(row) = speciesLabel(k);
        flagValue(row) = value;
        count(row) = combinationCounts(k, value + 1);
    end
end
combinationTable = table(species, flagValue, count);
writetable(combinationTable, fullfile(outputRoot, 'quality_flag_combination_counts.csv'));

modeSummary = table(modeValues(:), modeFileCounts(:), modeRecordCounts(:), ...
    'VariableNames', {'mode', 'source_files', 'records'});
writetable(modeSummary, fullfile(outputRoot, 'quality_flag_mode_summary.csv'));

productManifest = readtable(fullfile(outputRoot, 'quality_flag_all_species_manifest.csv'));
summary = struct();
summary.algorithm_version = cfg.algorithm_version;
summary.generated_utc = char(datetime('now', TimeZone='UTC', ...
    Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
summary.source_checkpoints = numel(files);
summary.source_checkpoint_load_failures = loadFailures;
summary.raw_files_changed_since_checkpoint = rawChanged;
summary.species = speciesSummary;
summary.mode = modeSummary;
summary.products = height(productManifest);
summary.product_failures = nnz(string(productManifest.status) == "failed");
summary.nv_unmatched = [sum(productManifest.nv_unmatched_H), ...
    sum(productManifest.nv_unmatched_O), sum(productManifest.nv_unmatched_O2)];
summary.spe_unmatched = [sum(productManifest.spe_unmatched_H), ...
    sum(productManifest.spe_unmatched_O), sum(productManifest.spe_unmatched_O2)];
save(fullfile(outputRoot, 'quality_flag_production_validation_summary.mat'), ...
    'summary', 'cfg', 'combinationTable');
fprintf(['Source=%d loadFailures=%d rawChanged=%d products=%d ', ...
    'productFailures=%d NVunmatched=%s SPEunmatched=%s\n'], ...
    summary.source_checkpoints, loadFailures, rawChanged, summary.products, ...
    summary.product_failures, mat2str(summary.nv_unmatched), ...
    mat2str(summary.spe_unmatched));
end
