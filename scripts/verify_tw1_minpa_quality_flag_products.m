function summary = verify_tw1_minpa_quality_flag_products(dataRoot, outputDir)
%VERIFY_TW1_MINPA_QUALITY_FLAG_PRODUCTS Independently reopen every product.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(dataRoot)
    dataRoot = 'D:\Data\TW-1\result\MINPA';
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(repoRoot, 'outputs', ...
        'tw1_minpa_quality_flags_all_species_validation');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

cfg = tw1_minpa_quality_flag_config();
nvDir = fullfile(dataRoot, 'NV_MSO_2');
speDir = fullfile(dataRoot, 'day_spe');
files = dir(fullfile(nvDir, 'NV_*.mat'));
days = strings(0, 1);
for k = 1:numel(files)
    tok = regexp(files(k).name, '^NV_(\d{8})\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        days(end + 1, 1) = string(tok{1}); %#ok<AGROW>
    end
end
days = sort(days);

status = repmat("ok", numel(days), 1);
message = strings(numel(days), 1);
nvRows = zeros(numel(days), 3);
nvAvailable = zeros(numel(days), 3);
speRows = zeros(numel(days), 3);
speAvailable = zeros(numel(days), 3);

for d = 1:numel(days)
    day = char(days(d));
    nvPath = fullfile(nvDir, ['NV_', day, '.mat']);
    spePath = fullfile(speDir, ['Ion_spe_', day, '.mat']);
    try
        [nvRows(d, :), nvAvailable(d, :)] = verify_nv(nvPath, cfg);
        [speRows(d, :), speAvailable(d, :)] = verify_spe(spePath, cfg);
    catch ME
        status(d) = "failed";
        message(d) = string(ME.message);
    end
    if mod(d, 100) == 0
        fprintf('Verified %d/%d product pairs.\n', d, numel(days));
    end
end

verification = table(days, status, message, ...
    nvRows(:, 1), nvRows(:, 2), nvRows(:, 3), ...
    nvAvailable(:, 1), nvAvailable(:, 2), nvAvailable(:, 3), ...
    speRows(:, 1), speRows(:, 2), speRows(:, 3), ...
    speAvailable(:, 1), speAvailable(:, 2), speAvailable(:, 3), ...
    'VariableNames', {'day', 'status', 'message', ...
    'nv_rows_H', 'nv_rows_O', 'nv_rows_O2', ...
    'nv_available_H', 'nv_available_O', 'nv_available_O2', ...
    'spe_rows_H', 'spe_rows_O', 'spe_rows_O2', ...
    'spe_available_H', 'spe_available_O', 'spe_available_O2'});
writetable(verification, fullfile(outputDir, 'all_product_schema_verification.csv'));

summary = struct();
summary.algorithm_version = cfg.algorithm_version;
summary.product_pairs = numel(days);
summary.failures = nnz(status == "failed");
summary.nv_rows = sum(nvRows, 1);
summary.nv_available = sum(nvAvailable, 1);
summary.spe_rows = sum(speRows, 1);
summary.spe_available = sum(speAvailable, 1);
summary.generated_utc = char(datetime('now', TimeZone='UTC', ...
    Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
save(fullfile(outputDir, 'all_product_schema_verification_summary.mat'), ...
    'summary', 'verification', 'cfg');
fprintf(['Product pairs=%d failures=%d NVrows=%s NVavailable=%s ', ...
    'SPErows=%s SPEavailable=%s\n'], summary.product_pairs, summary.failures, ...
    mat2str(summary.nv_rows), mat2str(summary.nv_available), ...
    mat2str(summary.spe_rows), mat2str(summary.spe_available));
end

function [rows, availableCount] = verify_nv(path, cfg)
s = load(path);
assert(isfield(s, 'NH_TW1'), 'Missing NH_TW1 in %s.', path);
n = size(s.NH_TW1, 1);
rows = zeros(1, 3);
availableCount = zeros(1, 3);
for k = 1:numel(cfg.species)
    species = cfg.species{k};
    flag = s.(sprintf('quality_flag_%s_TW1', species));
    bits = s.(sprintf('quality_flag_bit_%s_TW1', species));
    binary = s.(sprintf('quality_flag_binary_%s_TW1', species));
    available = s.(sprintf('quality_flag_available_%s_TW1', species));
    assert(size(flag, 1) == n && size(flag, 2) == 2);
    assert(size(bits, 1) == n && size(bits, 2) == 6);
    assert(numel(binary) == n);
    assert(size(available, 1) == n && size(available, 2) == 2);
    reconstructed = sum(uint32(bits(:, 2:6)) .* uint32(2 .^ (0:4)), 2);
    assert(all(reconstructed == uint32(flag(:, 2))));
    rows(k) = n;
    availableCount(k) = nnz(logical(available(:, 2)));
end
assert(isfield(s, 'MINPA_quality_flag_info'));
assert(strcmp(s.MINPA_quality_flag_info.algorithm_version, cfg.algorithm_version));
end

function [rows, availableCount] = verify_spe(path, cfg)
s = load(path);
topNames = {'H_spe_num', 'O_spe_num', 'O2_spe_num'};
rows = zeros(1, 3);
availableCount = zeros(1, 3);
for k = 1:numel(topNames)
    assert(isfield(s, topNames{k}) && isstruct(s.(topNames{k})));
    segments = fieldnames(s.(topNames{k}));
    for j = 1:numel(segments)
        seg = s.(topNames{k}).(segments{j});
        n = numel(seg.t);
        assert(isfield(seg, 'quality_flag') && numel(seg.quality_flag) == n);
        assert(isfield(seg, 'quality_flag_bit') ...
            && isequal(size(seg.quality_flag_bit), [n, 5]));
        assert(isfield(seg, 'quality_flag_binary') ...
            && numel(seg.quality_flag_binary) == n);
        assert(isfield(seg, 'quality_flag_available') ...
            && numel(seg.quality_flag_available) == n);
        assert(isfield(seg, 'quality_flag_species'));
        reconstructed = sum(uint32(seg.quality_flag_bit) .* uint32(2 .^ (0:4)), 2);
        assert(all(reconstructed == uint32(seg.quality_flag(:))));
        rows(k) = rows(k) + n;
        availableCount(k) = availableCount(k) + nnz(seg.quality_flag_available);
    end
end
assert(isfield(s, 'MINPA_quality_flag_info'));
assert(strcmp(s.MINPA_quality_flag_info.algorithm_version, cfg.algorithm_version));
end
