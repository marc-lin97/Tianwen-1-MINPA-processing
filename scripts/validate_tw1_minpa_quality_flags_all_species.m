function report = validate_tw1_minpa_quality_flags_all_species(outputDir)
%VALIDATE_TW1_MINPA_QUALITY_FLAGS_ALL_SPECIES Regression and product checks.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(repoRoot, 'outputs', ...
        'tw1_minpa_quality_flags_all_species_validation');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

cfg = tw1_minpa_quality_flag_config();
dataRoot = 'D:\Data\TW-1\result\MINPA';
cases = {
    1, '20250916', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD1-DEF_SCI_N_20250916173641_20250917012529_05290_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_7band_bit4_20250916_check', 'tw1_minpa_hplus_quality_flag_audit_records.csv');
    4, '20250916', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD4-MT_SCI_N_20250916005152_20250916012123_05287_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_regression_mode4_12', 'tw1_minpa_hplus_quality_flag_audit_records.csv');
    4, '20250916', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD4-MT_SCI_N_20250916083944_20250916090928_05288_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_regression_mode4_12', 'tw1_minpa_hplus_quality_flag_audit_records.csv');
    4, '20250916', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD4-MT_SCI_N_20250916163808_20250916170751_05289_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_regression_mode4_12', 'tw1_minpa_hplus_quality_flag_audit_records.csv');
    7, '20230406', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD7-FR_SCI_N_20230406141914_20230406150647_02300_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_regression_mode7', 'tw1_minpa_hplus_quality_flag_audit_records.csv');
    7, '20230406', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD7-FR_SCI_N_20230406211510_20230406220642_02301_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_regression_mode7', 'tw1_minpa_hplus_quality_flag_audit_records.csv');
    12, '20250916', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD12-I_SCI_N_20250916045532_20250916051525_05288_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_regression_mode4_12', 'tw1_minpa_hplus_quality_flag_audit_records.csv');
    12, '20250916', ...
    fullfile(dataRoot, 'ori', 'HX1-Or_GRAS_MINPA-MOD12-I_SCI_N_20250916204201_20250916210150_05290_A.mat'), ...
    fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_regression_mode4_12', 'tw1_minpa_hplus_quality_flag_audit_records.csv')};

mode = zeros(size(cases, 1), 1);
records = zeros(size(cases, 1), 1);
hMismatch = zeros(size(cases, 1), 1);
speciesBitFailures = zeros(size(cases, 1), 1);
for c = 1:size(cases, 1)
    mode(c) = cases{c, 1};
    day = cases{c, 2};
    sourcePath = cases{c, 3};
    auditPath = cases{c, 4};
    energy = energy_for_mode(fullfile(dataRoot, 'day_spe', ...
        ['Ion_spe_', day, '.mat']), mode(c));
    out = tw1_minpa_quality_flag_all_species_core(sourcePath, energy, []);
    audit = readtable(auditPath, TextType='string');
    audit = audit(audit.source_file == string(sourcePath), :);
    [matched, location] = ismembertol(out.time_unix_s, ...
        double(audit.time_unix_s), 1e-6, DataScale=1);
    assert(all(matched), 'Audit time mismatch for mode %d.', mode(c));
    records(c) = numel(out.time_unix_s);
    hMismatch(c) = nnz(out.species.H.flag ~= uint32(audit.quality_flag(location)));
    for species = cfg.species
        one = out.species.(species{1});
        reconstructed = sum(uint32(one.bits) .* uint32(2 .^ (0:4)), 2);
        speciesBitFailures(c) = speciesBitFailures(c) ...
            + nnz(reconstructed ~= one.flag) ...
            + nnz(bitget(one.flag, 1) & bitget(one.flag, 2));
    end
end

regression = table(mode, records, hMismatch, speciesBitFailures, ...
    'VariableNames', {'mode', 'records', 'h_flag_mismatch', 'species_bit_failures'});
writetable(regression, fullfile(outputDir, 'mode_regression.csv'));
assert(all(hMismatch == 0), 'H+ regression mismatch detected.');
assert(all(speciesBitFailures == 0), 'Flag bit consistency failure detected.');

productDay = '20250916';
nvPath = fullfile(dataRoot, 'NV_MSO_2', ['NV_', productDay, '.mat']);
spePath = fullfile(dataRoot, 'day_spe', ['Ion_spe_', productDay, '.mat']);
[nvRows, nvUnavailable] = validate_nv_product(nvPath, cfg);
[speRows, speUnavailable] = validate_spe_product(spePath, cfg);

report = struct();
report.algorithm_version = cfg.algorithm_version;
report.generated_utc = char(datetime('now', TimeZone='UTC', ...
    Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
report.regression = regression;
report.regression_records = sum(records);
report.regression_h_mismatches = sum(hMismatch);
report.regression_species_bit_failures = sum(speciesBitFailures);
report.product_day = productDay;
report.nv_rows_per_species = nvRows;
report.nv_unavailable_per_species = nvUnavailable;
report.spe_rows_per_species = speRows;
report.spe_unavailable_per_species = speUnavailable;
save(fullfile(outputDir, 'validation_report.mat'), 'report', 'cfg');
fprintf('Validated %d regression records; H mismatches=%d; bit failures=%d.\n', ...
    report.regression_records, report.regression_h_mismatches, ...
    report.regression_species_bit_failures);
end

function energy = energy_for_mode(spePath, wantedMode)
s = load(spePath, 'H_spe_num');
segments = fieldnames(s.H_spe_num);
energy = [];
for k = 1:numel(segments)
    seg = s.H_spe_num.(segments{k});
    if round(double(seg.mod)) == wantedMode
        energy = double(seg.f(:));
        break
    end
end
assert(~isempty(energy), 'No energy centers for mode %d in %s.', wantedMode, spePath);
end

function [rows, unavailable] = validate_nv_product(path, cfg)
s = load(path);
rows = zeros(1, 3);
unavailable = zeros(1, 3);
for k = 1:numel(cfg.species)
    species = cfg.species{k};
    flag = s.(sprintf('quality_flag_%s_TW1', species));
    bits = s.(sprintf('quality_flag_bit_%s_TW1', species));
    available = s.(sprintf('quality_flag_available_%s_TW1', species));
    binary = s.(sprintf('quality_flag_binary_%s_TW1', species));
    rows(k) = size(flag, 1);
    unavailable(k) = nnz(~logical(available(:, 2)));
    reconstructed = sum(uint32(bits(:, 2:6)) .* uint32(2 .^ (0:4)), 2);
    assert(all(reconstructed == uint32(flag(:, 2))));
    assert(numel(binary) == rows(k));
end
assert(strcmp(s.MINPA_quality_flag_info.algorithm_version, cfg.algorithm_version));
end

function [rows, unavailable] = validate_spe_product(path, cfg)
s = load(path);
topNames = {'H_spe_num', 'O_spe_num', 'O2_spe_num'};
rows = zeros(1, 3);
unavailable = zeros(1, 3);
for k = 1:numel(topNames)
    segments = fieldnames(s.(topNames{k}));
    for j = 1:numel(segments)
        seg = s.(topNames{k}).(segments{j});
        assert(numel(seg.quality_flag) == numel(seg.t));
        assert(size(seg.quality_flag_bit, 2) == 5);
        rows(k) = rows(k) + numel(seg.t);
        unavailable(k) = unavailable(k) + nnz(~seg.quality_flag_available);
    end
end
assert(strcmp(s.MINPA_quality_flag_info.algorithm_version, cfg.algorithm_version));
end
