function summary = audit_tw1_minpa_quality_flags(dateList, maxFiles, stopAfterSamples, makePlots, outputDir, allowedModes)
%AUDIT_TW1_MINPA_QUALITY_FLAGS Build an audit set for MINPA quality flags.
%
% This script reads MINPA ori products read-only, computes H+ per-record
% quality flags from the raw energy-angle-mass data, and writes audit tables
% and example figures for review. It does not modify NV_MSO_2 or spectra files.
%
% The final quality flag is a 5-bit bitmask. Bit order follows the user's
% original flag order 1-5:
%   bit 1, value 1  : valid 2D channel count is 5-10, use with caution
%   bit 2, value 2  : valid 2D channel count is <5, invalid sample
%   bit 3, value 4  : >80% 1D energy channels have DEF >1e5, UV contaminated
%   bit 4, value 8  : 1D peak/valley alternation with a consistent odd/even phase
%   bit 5, value 16 : any 2D DEF channel exceeds 1e10, distorted/caution

if nargin < 1
    dateList = {};
end
if nargin < 2 || isempty(maxFiles)
    maxFiles = inf;
end
if nargin < 3 || isempty(stopAfterSamples)
    stopAfterSamples = true;
end
if nargin < 4 || isempty(makePlots)
    makePlots = true;
end
if nargin < 5
    outputDir = '';
end
if nargin < 6
    allowedModes = [];
end
if ischar(dateList) || isstring(dateList)
    dateList = cellstr(dateList);
end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
paths.oriDir = 'D:\Data\TW-1\result\MINPA\ori';
paths.nvMso2Dir = 'D:\Data\TW-1\result\MINPA\NV_MSO_2';
if isempty(outputDir)
    paths.outputDir = fullfile(repoRoot, 'outputs', 'tw1_minpa_quality_flag_audit');
else
    paths.outputDir = outputDir;
end
paths.figureDir = fullfile(paths.outputDir, 'examples');
if ~exist(paths.outputDir, 'dir')
    mkdir(paths.outputDir);
end
if ~exist(paths.figureDir, 'dir')
    mkdir(paths.figureDir);
end

cfg = quality_flag_config();
files = select_ori_files(paths.oriDir, dateList);
files = filter_ori_files_by_mode(files, allowedModes);
if isfinite(maxFiles)
    files = files(1:min(numel(files), maxFiles));
end

allRows = table();
sampleMap = init_sample_map(cfg.sampleCategories);
fileSummaries = table();

for f = 1:numel(files)
    filePath = fullfile(files(f).folder, files(f).name);
    try
        [rows, sampleMap] = process_ori_file(filePath, cfg, sampleMap);
    catch ME
        warning('Failed %s: %s', filePath, ME.message);
        continue
    end
    if isempty(rows)
        continue
    end
    allRows = [allRows; rows]; %#ok<AGROW>
    fileSummaries = [fileSummaries; summarize_file(rows, files(f).name)]; %#ok<AGROW>
    fprintf('Scanned %s records=%d foundSamples=%d/%d\n', files(f).name, height(rows), sample_count(sampleMap), cfg.maxSamplesTotal);
    if stopAfterSamples && sample_count(sampleMap) >= cfg.maxSamplesTotal && height(allRows) > 20000
        % Keep scanning a little after finding examples so the audit table is
        % not only edge cases, but avoid an unnecessarily large review run.
        break
    end
end

auditCsv = fullfile(paths.outputDir, 'tw1_minpa_hplus_quality_flag_audit_records.csv');
summaryCsv = fullfile(paths.outputDir, 'tw1_minpa_hplus_quality_flag_audit_file_summary.csv');
if ~isempty(allRows)
    writetable(allRows, auditCsv);
end
if ~isempty(fileSummaries)
    writetable(fileSummaries, summaryCsv);
end

samples = sample_map_to_table(sampleMap);
sampleCsv = fullfile(paths.outputDir, 'tw1_minpa_hplus_quality_flag_audit_examples.csv');
if ~isempty(samples)
    writetable(samples, sampleCsv);
    if makePlots
        plot_sample_examples(samples, cfg, paths);
    end
end

summary = struct();
summary.output_dir = paths.outputDir;
summary.audit_csv = auditCsv;
summary.file_summary_csv = summaryCsv;
summary.sample_csv = sampleCsv;
summary.figure_dir = paths.figureDir;
if isempty(allRows)
    summary.files_scanned = 0;
    summary.records_scanned = 0;
else
    summary.files_scanned = numel(unique(allRows.source_file));
    summary.records_scanned = height(allRows);
end
summary.flag_bits = cfg.flagBits;
summary.flag_bit_order = cfg.flagBitOrder;
summary.thresholds = cfg.thresholds;
summary.stop_after_samples = stopAfterSamples;
summary.make_plots = makePlots;
summary.allowed_modes = allowedModes;
summary.samples_found = samples;
summary.notes = [
    "Audit only; raw ori, NV_MSO_2, and spectra products are not modified."
    "This first implementation computes H+ flags. The same mechanics can be extended to O+ and O2+ before product insertion."
    "Odd/even acquisition-error threshold uses current-record spectra only: peaks require DEF >=1e5, valleys must be <50% of both neighbors; >=52.5% finite interior channels are valid peaks/valleys, and >=55% of those extrema fit one odd/even phase pattern."
    "Bit 4 is evaluated independently for all energies, 0-30 eV, 30-300 eV, 300 eV-maximum energy, 0-50 eV, 50-500 eV, and 500-5000 eV. If any one band passes, bit 4 is set; all band decisions are retained in the audit table."
];

summaryMat = fullfile(paths.outputDir, 'tw1_minpa_hplus_quality_flag_audit_summary.mat');
save(summaryMat, 'summary', 'cfg');
summary.summary_mat = summaryMat;

fprintf('Wrote %s\n', auditCsv);
fprintf('Wrote %s\n', sampleCsv);
fprintf('Wrote figures under %s\n', paths.figureDir);
end

function cfg = quality_flag_config()
cfg.thresholds.valid_min = 0;
cfg.thresholds.valid_max_exclusive = 1e8;
cfg.thresholds.sparse_invalid_lt = 5;
cfg.thresholds.sparse_caution_min = 5;
cfg.thresholds.sparse_caution_max = 10;
cfg.thresholds.uv_def_threshold = 1e5;
cfg.thresholds.uv_fraction_threshold = 0.8;
cfg.thresholds.alternating_peak_def_min = 1e5;
cfg.thresholds.alternating_valley_neighbor_fraction_max = 0.5;
cfg.thresholds.alternating_extrema_fraction_threshold = 0.525;
cfg.thresholds.alternating_phase_fraction_threshold = 0.55;
cfg.thresholds.alternating_energy_bands_eV = [0, 30; 30, 300; 300, Inf; 0, 50; 50, 500; 500, 5000];
cfg.thresholds.parity_def_threshold = 1e4; % Auxiliary diagnostic only.
cfg.thresholds.high_channel_def_threshold = 1e10;
cfg.maxSamplesPerCategory = 2;
cfg.sampleCategories = {'normal', 'sparse_invalid', 'sparse_caution', 'uv_contamination', 'evenodd_error', 'high_channel'};
cfg.maxSamplesTotal = cfg.maxSamplesPerCategory * numel(cfg.sampleCategories);
cfg.flagBits = struct( ...
    'sparse_caution', uint32(1), ...
    'sparse_invalid', uint32(2), ...
    'uv_contamination', uint32(4), ...
    'evenodd_error', uint32(8), ...
    'high_channel', uint32(16));
cfg.flagBitOrder = [
    "bit1/value1: valid 2D channel count is 5-10, use with caution"
    "bit2/value2: valid 2D channel count is <5, invalid sample"
    "bit3/value4: >80% 1D energy channels have DEF >1e5, UV contaminated"
    "bit4/value8: odd/even acquisition error in any tested energy band: all energies, 0-30, 30-300, 300-max, 0-50, 50-500, or 500-5000 eV"
    "bit5/value16: any 2D DEF channel exceeds 1e10, distorted/caution"
];
cfg.species = 'H';
end

function files = select_ori_files(oriDir, dateList)
files = dir(fullfile(oriDir, '*.mat'));
if isempty(dateList)
    return
end
wanted = string(dateList);
keep = false(size(files));
for k = 1:numel(files)
    toks = regexp(files(k).name, '_(\d{8})\d{6}_(\d{8})\d{6}_', 'tokens', 'once');
    if isempty(toks)
        continue
    end
    startDay = string(toks{1});
    stopDay = string(toks{2});
    keep(k) = any(wanted == startDay) || any(wanted == stopDay);
end
files = files(keep);
end

function files = filter_ori_files_by_mode(files, allowedModes)
if isempty(allowedModes)
    return
end
allowedModes = double(allowedModes(:));
keep = false(size(files));
for k = 1:numel(files)
    filePath = fullfile(files(k).folder, files(k).name);
    keep(k) = any(parse_mode(filePath) == allowedModes);
end
files = files(keep);
end

function [rows, sampleMap] = process_ori_file(filePath, cfg, sampleMap)
mode = parse_mode(filePath);
if mode == 13
    rows = table();
    return
end
geom = minpa_geometry(mode);
if isempty(geom.energy) || isempty(geom.masses)
    rows = table();
    return
end
groups = species_groups(geom.masses, cfg.species, mode);
if isempty(groups)
    rows = table();
    return
end

s = load(filePath, 'tw1_MINPA_data');
a = s.tw1_MINPA_data;
[times, rawRows] = read_times_and_counts(a, mode);
validTime = isfinite(times);
times = times(validTime);
rawRows = rawRows(validTime, :);
if isempty(times)
    rows = table();
    return
end

n = numel(times);
source = strings(n, 1) + string(filePath);
modeCol = repmat(mode, n, 1);
recordIndex = find(validTime);
qualityFlag = zeros(n, 1, 'uint32');
qualityFlagBinary = strings(n, 1);
flagBit1SparseCaution = false(n, 1);
flagBit2SparseInvalid = false(n, 1);
flagBit3UvContamination = false(n, 1);
flagBit4EvenOddError = false(n, 1);
flagBit5HighChannel = false(n, 1);
valid2dCount = zeros(n, 1);
nonzeroEnergyBins = zeros(n, 1);
nonzeroAngleBins = zeros(n, 1);
uvFraction = nan(n, 1);
oddPositiveFraction = nan(n, 1);
evenPositiveFraction = nan(n, 1);
oddZeroFraction = nan(n, 1);
evenZeroFraction = nan(n, 1);
alternatingExtremaFraction = nan(n, 1);
alternatingExtremaCount = nan(n, 1);
alternatingExtremaTotal = nan(n, 1);
alternatingPhaseFraction = nan(n, 1);
alternatingPhaseCount = nan(n, 1);
alternatingPhaseTotal = nan(n, 1);
flagBit4EvenOddAllEnergy = false(n, 1);
flagBit4EvenOdd0To30eV = false(n, 1);
flagBit4EvenOdd30To300eV = false(n, 1);
flagBit4EvenOdd300ToMaxeV = false(n, 1);
flagBit4EvenOdd0To50eV = false(n, 1);
flagBit4EvenOdd50To500eV = false(n, 1);
flagBit4EvenOdd500To5000eV = false(n, 1);
alternatingExtremaFraction0To30eV = nan(n, 1);
alternatingExtremaCount0To30eV = nan(n, 1);
alternatingExtremaTotal0To30eV = nan(n, 1);
alternatingPhaseFraction0To30eV = nan(n, 1);
alternatingPhaseCount0To30eV = nan(n, 1);
alternatingPhaseTotal0To30eV = nan(n, 1);
alternatingExtremaFraction30To300eV = nan(n, 1);
alternatingExtremaCount30To300eV = nan(n, 1);
alternatingExtremaTotal30To300eV = nan(n, 1);
alternatingPhaseFraction30To300eV = nan(n, 1);
alternatingPhaseCount30To300eV = nan(n, 1);
alternatingPhaseTotal30To300eV = nan(n, 1);
alternatingExtremaFraction300ToMaxeV = nan(n, 1);
alternatingExtremaCount300ToMaxeV = nan(n, 1);
alternatingExtremaTotal300ToMaxeV = nan(n, 1);
alternatingPhaseFraction300ToMaxeV = nan(n, 1);
alternatingPhaseCount300ToMaxeV = nan(n, 1);
alternatingPhaseTotal300ToMaxeV = nan(n, 1);
alternatingExtremaFraction0To50eV = nan(n, 1);
alternatingExtremaCount0To50eV = nan(n, 1);
alternatingExtremaTotal0To50eV = nan(n, 1);
alternatingPhaseFraction0To50eV = nan(n, 1);
alternatingPhaseCount0To50eV = nan(n, 1);
alternatingPhaseTotal0To50eV = nan(n, 1);
alternatingExtremaFraction50To500eV = nan(n, 1);
alternatingExtremaCount50To500eV = nan(n, 1);
alternatingExtremaTotal50To500eV = nan(n, 1);
alternatingPhaseFraction50To500eV = nan(n, 1);
alternatingPhaseCount50To500eV = nan(n, 1);
alternatingPhaseTotal50To500eV = nan(n, 1);
alternatingExtremaFraction500To5000eV = nan(n, 1);
alternatingExtremaCount500To5000eV = nan(n, 1);
alternatingExtremaTotal500To5000eV = nan(n, 1);
alternatingPhaseFraction500To5000eV = nan(n, 1);
alternatingPhaseCount500To5000eV = nan(n, 1);
alternatingPhaseTotal500To5000eV = nan(n, 1);
maxDef2d = nan(n, 1);
maxDefEnergyEV = nan(n, 1);
maxDefAngleIndex = nan(n, 1);

for i = 1:n
    [dpf2d, def2d, def1d] = extract_species_h_cube(rawRows(i, :), geom, groups);
    m = metrics_for_record(dpf2d, def2d, def1d, geom.energy, cfg);
    metricList(i) = m; %#ok<AGROW>
    qualityFlag(i) = m.flag;
    flagBit1SparseCaution(i) = bitand(m.flag, cfg.flagBits.sparse_caution) ~= 0;
    flagBit2SparseInvalid(i) = bitand(m.flag, cfg.flagBits.sparse_invalid) ~= 0;
    flagBit3UvContamination(i) = bitand(m.flag, cfg.flagBits.uv_contamination) ~= 0;
    flagBit4EvenOddError(i) = bitand(m.flag, cfg.flagBits.evenodd_error) ~= 0;
    flagBit5HighChannel(i) = bitand(m.flag, cfg.flagBits.high_channel) ~= 0;
    valid2dCount(i) = m.valid2dCount;
    nonzeroEnergyBins(i) = m.nonzeroEnergyBins;
    nonzeroAngleBins(i) = m.nonzeroAngleBins;
    uvFraction(i) = m.uvFraction;
    oddPositiveFraction(i) = m.oddPositiveFraction;
    evenPositiveFraction(i) = m.evenPositiveFraction;
    oddZeroFraction(i) = m.oddZeroFraction;
    evenZeroFraction(i) = m.evenZeroFraction;
    alternatingExtremaFraction(i) = m.alternatingExtremaFraction;
    alternatingExtremaCount(i) = m.alternatingExtremaCount;
    alternatingExtremaTotal(i) = m.alternatingExtremaTotal;
    alternatingPhaseFraction(i) = m.alternatingPhaseFraction;
    alternatingPhaseCount(i) = m.alternatingPhaseCount;
    alternatingPhaseTotal(i) = m.alternatingPhaseTotal;
    flagBit4EvenOddAllEnergy(i) = m.evenOddAllEnergyPass;
    flagBit4EvenOdd0To30eV(i) = m.evenOdd0To30Pass;
    flagBit4EvenOdd30To300eV(i) = m.evenOdd30To300Pass;
    flagBit4EvenOdd300ToMaxeV(i) = m.evenOdd300ToMaxPass;
    flagBit4EvenOdd0To50eV(i) = m.evenOdd0To50Pass;
    flagBit4EvenOdd50To500eV(i) = m.evenOdd50To500Pass;
    flagBit4EvenOdd500To5000eV(i) = m.evenOdd500To5000Pass;
    alternatingExtremaFraction0To30eV(i) = m.alternatingExtremaFraction0To30;
    alternatingExtremaCount0To30eV(i) = m.alternatingExtremaCount0To30;
    alternatingExtremaTotal0To30eV(i) = m.alternatingExtremaTotal0To30;
    alternatingPhaseFraction0To30eV(i) = m.alternatingPhaseFraction0To30;
    alternatingPhaseCount0To30eV(i) = m.alternatingPhaseCount0To30;
    alternatingPhaseTotal0To30eV(i) = m.alternatingPhaseTotal0To30;
    alternatingExtremaFraction30To300eV(i) = m.alternatingExtremaFraction30To300;
    alternatingExtremaCount30To300eV(i) = m.alternatingExtremaCount30To300;
    alternatingExtremaTotal30To300eV(i) = m.alternatingExtremaTotal30To300;
    alternatingPhaseFraction30To300eV(i) = m.alternatingPhaseFraction30To300;
    alternatingPhaseCount30To300eV(i) = m.alternatingPhaseCount30To300;
    alternatingPhaseTotal30To300eV(i) = m.alternatingPhaseTotal30To300;
    alternatingExtremaFraction300ToMaxeV(i) = m.alternatingExtremaFraction300ToMax;
    alternatingExtremaCount300ToMaxeV(i) = m.alternatingExtremaCount300ToMax;
    alternatingExtremaTotal300ToMaxeV(i) = m.alternatingExtremaTotal300ToMax;
    alternatingPhaseFraction300ToMaxeV(i) = m.alternatingPhaseFraction300ToMax;
    alternatingPhaseCount300ToMaxeV(i) = m.alternatingPhaseCount300ToMax;
    alternatingPhaseTotal300ToMaxeV(i) = m.alternatingPhaseTotal300ToMax;
    alternatingExtremaFraction0To50eV(i) = m.alternatingExtremaFraction0To50;
    alternatingExtremaCount0To50eV(i) = m.alternatingExtremaCount0To50;
    alternatingExtremaTotal0To50eV(i) = m.alternatingExtremaTotal0To50;
    alternatingPhaseFraction0To50eV(i) = m.alternatingPhaseFraction0To50;
    alternatingPhaseCount0To50eV(i) = m.alternatingPhaseCount0To50;
    alternatingPhaseTotal0To50eV(i) = m.alternatingPhaseTotal0To50;
    alternatingExtremaFraction50To500eV(i) = m.alternatingExtremaFraction50To500;
    alternatingExtremaCount50To500eV(i) = m.alternatingExtremaCount50To500;
    alternatingExtremaTotal50To500eV(i) = m.alternatingExtremaTotal50To500;
    alternatingPhaseFraction50To500eV(i) = m.alternatingPhaseFraction50To500;
    alternatingPhaseCount50To500eV(i) = m.alternatingPhaseCount50To500;
    alternatingPhaseTotal50To500eV(i) = m.alternatingPhaseTotal50To500;
    alternatingExtremaFraction500To5000eV(i) = m.alternatingExtremaFraction500To5000;
    alternatingExtremaCount500To5000eV(i) = m.alternatingExtremaCount500To5000;
    alternatingExtremaTotal500To5000eV(i) = m.alternatingExtremaTotal500To5000;
    alternatingPhaseFraction500To5000eV(i) = m.alternatingPhaseFraction500To5000;
    alternatingPhaseCount500To5000eV(i) = m.alternatingPhaseCount500To5000;
    alternatingPhaseTotal500To5000eV(i) = m.alternatingPhaseTotal500To5000;
    maxDef2d(i) = m.maxDef2d;
    maxDefEnergyEV(i) = m.maxDefEnergyEV;
    maxDefAngleIndex(i) = m.maxDefAngleIndex;
end

for i = 1:n
    qualityFlagBinary(i) = flag_binary_string(qualityFlag(i));
    m = metricList(i);
    m.flag = qualityFlag(i);
    sampleMap = maybe_add_samples(sampleMap, m, filePath, mode, recordIndex(i), times(i));
end

timeUtc = strings(n, 1);
for i = 1:n
    timeUtc(i) = string(datetime(times(i), ConvertFrom='posixtime', TimeZone='UTC', Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
end

rows = table(source, recordIndex(:), timeUtc, times(:), modeCol, qualityFlag, ...
    qualityFlagBinary, flagBit1SparseCaution, flagBit2SparseInvalid, flagBit3UvContamination, ...
    flagBit4EvenOddError, flagBit5HighChannel, valid2dCount, nonzeroEnergyBins, nonzeroAngleBins, uvFraction, ...
    oddPositiveFraction, evenPositiveFraction, oddZeroFraction, evenZeroFraction, ...
    alternatingExtremaFraction, alternatingExtremaCount, alternatingExtremaTotal, ...
    alternatingPhaseFraction, alternatingPhaseCount, alternatingPhaseTotal, ...
    flagBit4EvenOddAllEnergy, flagBit4EvenOdd0To30eV, flagBit4EvenOdd30To300eV, flagBit4EvenOdd300ToMaxeV, ...
    flagBit4EvenOdd0To50eV, flagBit4EvenOdd50To500eV, flagBit4EvenOdd500To5000eV, ...
    alternatingExtremaFraction0To30eV, alternatingExtremaCount0To30eV, alternatingExtremaTotal0To30eV, ...
    alternatingPhaseFraction0To30eV, alternatingPhaseCount0To30eV, alternatingPhaseTotal0To30eV, ...
    alternatingExtremaFraction30To300eV, alternatingExtremaCount30To300eV, alternatingExtremaTotal30To300eV, ...
    alternatingPhaseFraction30To300eV, alternatingPhaseCount30To300eV, alternatingPhaseTotal30To300eV, ...
    alternatingExtremaFraction300ToMaxeV, alternatingExtremaCount300ToMaxeV, alternatingExtremaTotal300ToMaxeV, ...
    alternatingPhaseFraction300ToMaxeV, alternatingPhaseCount300ToMaxeV, alternatingPhaseTotal300ToMaxeV, ...
    alternatingExtremaFraction0To50eV, alternatingExtremaCount0To50eV, alternatingExtremaTotal0To50eV, ...
    alternatingPhaseFraction0To50eV, alternatingPhaseCount0To50eV, alternatingPhaseTotal0To50eV, ...
    alternatingExtremaFraction50To500eV, alternatingExtremaCount50To500eV, alternatingExtremaTotal50To500eV, ...
    alternatingPhaseFraction50To500eV, alternatingPhaseCount50To500eV, alternatingPhaseTotal50To500eV, ...
    alternatingExtremaFraction500To5000eV, alternatingExtremaCount500To5000eV, alternatingExtremaTotal500To5000eV, ...
    alternatingPhaseFraction500To5000eV, alternatingPhaseCount500To5000eV, alternatingPhaseTotal500To5000eV, ...
    maxDef2d, maxDefEnergyEV, maxDefAngleIndex, ...
    'VariableNames', {'source_file', 'record_index', 'time_utc', 'time_unix_s', 'mode', 'quality_flag', ...
    'quality_flag_binary', 'flag_bit1_sparse_caution', 'flag_bit2_sparse_invalid', 'flag_bit3_uv_contamination', ...
    'flag_bit4_evenodd_error', 'flag_bit5_high_channel', 'valid_2d_channel_count', ...
    'nonzero_energy_bin_count', 'nonzero_angle_bin_count', 'uv_fraction', ...
    'odd_positive_fraction', 'even_positive_fraction', 'odd_zero_fraction', 'even_zero_fraction', ...
    'alternating_extrema_fraction', 'alternating_extrema_count', 'alternating_extrema_total', ...
    'alternating_phase_fraction', 'alternating_phase_count', 'alternating_phase_total', ...
    'flag_bit4_evenodd_all_energy', 'flag_bit4_evenodd_0_30eV', 'flag_bit4_evenodd_30_300eV', 'flag_bit4_evenodd_300_maxeV', ...
    'flag_bit4_evenodd_0_50eV', 'flag_bit4_evenodd_50_500eV', 'flag_bit4_evenodd_500_5000eV', ...
    'alternating_extrema_fraction_0_30eV', 'alternating_extrema_count_0_30eV', 'alternating_extrema_total_0_30eV', ...
    'alternating_phase_fraction_0_30eV', 'alternating_phase_count_0_30eV', 'alternating_phase_total_0_30eV', ...
    'alternating_extrema_fraction_30_300eV', 'alternating_extrema_count_30_300eV', 'alternating_extrema_total_30_300eV', ...
    'alternating_phase_fraction_30_300eV', 'alternating_phase_count_30_300eV', 'alternating_phase_total_30_300eV', ...
    'alternating_extrema_fraction_300_maxeV', 'alternating_extrema_count_300_maxeV', 'alternating_extrema_total_300_maxeV', ...
    'alternating_phase_fraction_300_maxeV', 'alternating_phase_count_300_maxeV', 'alternating_phase_total_300_maxeV', ...
    'alternating_extrema_fraction_0_50eV', 'alternating_extrema_count_0_50eV', 'alternating_extrema_total_0_50eV', ...
    'alternating_phase_fraction_0_50eV', 'alternating_phase_count_0_50eV', 'alternating_phase_total_0_50eV', ...
    'alternating_extrema_fraction_50_500eV', 'alternating_extrema_count_50_500eV', 'alternating_extrema_total_50_500eV', ...
    'alternating_phase_fraction_50_500eV', 'alternating_phase_count_50_500eV', 'alternating_phase_total_50_500eV', ...
    'alternating_extrema_fraction_500_5000eV', 'alternating_extrema_count_500_5000eV', 'alternating_extrema_total_500_5000eV', ...
    'alternating_phase_fraction_500_5000eV', 'alternating_phase_count_500_5000eV', 'alternating_phase_total_500_5000eV', ...
    'max_def_2d', 'max_def_energy_eV', 'max_def_angle_index'});
end

function mode = parse_mode(filePath)
[~, name, ext] = fileparts(filePath);
tok = regexp([name ext], 'MINPA-MOD(\d+)-', 'tokens', 'once');
if isempty(tok)
    mode = NaN;
else
    mode = str2double(tok{1});
end
end

function [times, rawRows] = read_times_and_counts(a, mode)
utc = squeeze(struct2cell(a(1).value));
times0 = nan(numel(utc), 1);
for i = 1:numel(utc)
    txt = string(utc{i});
    if strlength(txt) > 20
        try
            times0(i) = posixtime(datetime(txt, InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''', TimeZone='UTC'));
        catch
            times0(i) = posixtime(datetime(txt, InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', TimeZone='UTC'));
        end
    end
end

dataValue = a(42).value;
nBase = min(numel(times0), numel(dataValue));
times0 = times0(1:nBase);
dataValue = dataValue(1:nBase);
if nBase == 0
    times = zeros(0, 1);
    rawRows = zeros(0, 0);
    return
end

if mode == 12
    tempValue = dataValue(1).t;
    rawRows = nan(numel(times0) * 2, numel(tempValue) / 2);
    times = sort([times0 + 4.1 / 4; times0 + 4.1 * 3 / 4]);
    for i = 1:numel(times0)
        row = double(dataValue(i).t);
        if numel(row) == size(rawRows, 2) * 2
            rawRows(2 * i - 1, :) = row(1:end/2);
            rawRows(2 * i, :) = row(end/2+1:end);
        end
    end
else
    tempValue = dataValue(1).t;
    rawRows = nan(numel(times0), numel(tempValue));
    times = times0;
    for i = 1:numel(times0)
        row = double(dataValue(i).t);
        if numel(row) == size(rawRows, 2)
            rawRows(i, :) = row;
        end
    end
end
end

function [dpf2d, def2d, def1d] = extract_species_h_cube(raw, geom, groups)
nEnergy = numel(geom.energy);
nAngle = numel(geom.omega);
dpf2d = zeros(nEnergy, nAngle);
for g = groups(:)'
    oneMass = raw(g:geom.massLen:end);
    if numel(oneMass) ~= nEnergy * nAngle
        dpf2d(:) = NaN;
        break
    end
    dpf2d = dpf2d + reshape(oneMass, [nAngle, nEnergy])';
end
def2d = dpf2d .* geom.energy(:);
omegaSum = sum(geom.omega, 'omitnan');
if isfinite(omegaSum) && omegaSum > 0
    def1d = sum(def2d .* geom.omega(:)', 2, 'omitnan') ./ omegaSum;
else
    def1d = nan(nEnergy, 1);
end
end

function m = metrics_for_record(dpf2d, def2d, def1d, energy, cfg)
valid2d = isfinite(dpf2d) & dpf2d > cfg.thresholds.valid_min & dpf2d < cfg.thresholds.valid_max_exclusive;
high1d = isfinite(def1d) & def1d > cfg.thresholds.parity_def_threshold;
low1d = isfinite(def1d) & def1d < cfg.thresholds.parity_def_threshold;
odd = false(size(def1d));
odd(1:2:end) = true;
even = ~odd;

m.valid2dCount = nnz(valid2d);
m.nonzeroEnergyBins = nnz(any(dpf2d > 0, 2));
m.nonzeroAngleBins = nnz(any(dpf2d > 0, 1));
m.uvFraction = mean(isfinite(def1d) & def1d > cfg.thresholds.uv_def_threshold);
m.oddPositiveFraction = mean(high1d(odd));
m.evenPositiveFraction = mean(high1d(even));
m.oddZeroFraction = mean(low1d(odd));
m.evenZeroFraction = mean(low1d(even));
[m.alternatingExtremaFraction, m.alternatingExtremaCount, m.alternatingExtremaTotal, ...
    m.alternatingPhaseFraction, m.alternatingPhaseCount, m.alternatingPhaseTotal, ...
    m.evenOddAllEnergyPass] = alternating_metrics_for_mask(def1d, energy, true(size(energy)), cfg);
[m.alternatingExtremaFraction0To30, m.alternatingExtremaCount0To30, m.alternatingExtremaTotal0To30, ...
    m.alternatingPhaseFraction0To30, m.alternatingPhaseCount0To30, m.alternatingPhaseTotal0To30, ...
    m.evenOdd0To30Pass] = alternating_metrics_for_mask(def1d, energy, energy >= 0 & energy < 30, cfg);
[m.alternatingExtremaFraction30To300, m.alternatingExtremaCount30To300, m.alternatingExtremaTotal30To300, ...
    m.alternatingPhaseFraction30To300, m.alternatingPhaseCount30To300, m.alternatingPhaseTotal30To300, ...
    m.evenOdd30To300Pass] = alternating_metrics_for_mask(def1d, energy, energy >= 30 & energy < 300, cfg);
[m.alternatingExtremaFraction300ToMax, m.alternatingExtremaCount300ToMax, m.alternatingExtremaTotal300ToMax, ...
    m.alternatingPhaseFraction300ToMax, m.alternatingPhaseCount300ToMax, m.alternatingPhaseTotal300ToMax, ...
    m.evenOdd300ToMaxPass] = alternating_metrics_for_mask(def1d, energy, energy >= 300, cfg);
[m.alternatingExtremaFraction0To50, m.alternatingExtremaCount0To50, m.alternatingExtremaTotal0To50, ...
    m.alternatingPhaseFraction0To50, m.alternatingPhaseCount0To50, m.alternatingPhaseTotal0To50, ...
    m.evenOdd0To50Pass] = alternating_metrics_for_mask(def1d, energy, energy >= 0 & energy < 50, cfg);
[m.alternatingExtremaFraction50To500, m.alternatingExtremaCount50To500, m.alternatingExtremaTotal50To500, ...
    m.alternatingPhaseFraction50To500, m.alternatingPhaseCount50To500, m.alternatingPhaseTotal50To500, ...
    m.evenOdd50To500Pass] = alternating_metrics_for_mask(def1d, energy, energy >= 50 & energy < 500, cfg);
[m.alternatingExtremaFraction500To5000, m.alternatingExtremaCount500To5000, m.alternatingExtremaTotal500To5000, ...
    m.alternatingPhaseFraction500To5000, m.alternatingPhaseCount500To5000, m.alternatingPhaseTotal500To5000, ...
    m.evenOdd500To5000Pass] = alternating_metrics_for_mask(def1d, energy, energy >= 500 & energy < 5000, cfg);
[m.maxDef2d, imax] = max(def2d(:), [], 'omitnan');
if isempty(imax) || ~isfinite(m.maxDef2d)
    m.maxDefEnergyEV = NaN;
    m.maxDefAngleIndex = NaN;
else
    [ie, ia] = ind2sub(size(def2d), imax);
    m.maxDefEnergyEV = energy(ie);
    m.maxDefAngleIndex = ia;
end

flag = uint32(0);
if m.valid2dCount < cfg.thresholds.sparse_invalid_lt
    flag = bitor(flag, cfg.flagBits.sparse_invalid);
elseif m.valid2dCount >= cfg.thresholds.sparse_caution_min && m.valid2dCount <= cfg.thresholds.sparse_caution_max
    flag = bitor(flag, cfg.flagBits.sparse_caution);
end
if m.uvFraction > cfg.thresholds.uv_fraction_threshold
    flag = bitor(flag, cfg.flagBits.uv_contamination);
end
if m.evenOddAllEnergyPass || m.evenOdd0To30Pass || m.evenOdd30To300Pass || m.evenOdd300ToMaxPass ...
        || m.evenOdd0To50Pass || m.evenOdd50To500Pass || m.evenOdd500To5000Pass
    flag = bitor(flag, cfg.flagBits.evenodd_error);
end
if isfinite(m.maxDef2d) && m.maxDef2d > cfg.thresholds.high_channel_def_threshold
    flag = bitor(flag, cfg.flagBits.high_channel);
end
m.flag = flag;
end

function [extFrac, extCount, extTotal, phaseFrac, phaseCount, phaseTotal, pass] = alternating_metrics_for_mask(def1d, energy, mask, cfg)
mask = logical(mask(:));
energy = double(energy(:));
y = double(def1d(:));
mask = mask & isfinite(energy);
if nnz(mask) < 3
    extFrac = NaN;
    extCount = 0;
    extTotal = 0;
    phaseFrac = NaN;
    phaseCount = 0;
    phaseTotal = 0;
    pass = false;
    return
end
y = y(mask);
[extFrac, extCount, extTotal] = alternating_extrema_fraction(y, ...
    cfg.thresholds.alternating_peak_def_min, ...
    cfg.thresholds.alternating_valley_neighbor_fraction_max);
[phaseFrac, phaseCount, phaseTotal] = alternating_phase_fraction(y, ...
    cfg.thresholds.alternating_peak_def_min, ...
    cfg.thresholds.alternating_valley_neighbor_fraction_max);
pass = isfinite(extFrac) ...
    && extFrac >= cfg.thresholds.alternating_extrema_fraction_threshold ...
    && isfinite(phaseFrac) ...
    && phaseFrac >= cfg.thresholds.alternating_phase_fraction_threshold;
end

function [fraction, count, total] = alternating_extrema_fraction(def1d, peakDefMin, valleyNeighborFractionMax)
y = double(def1d(:));
if numel(y) < 3
    fraction = NaN;
    count = 0;
    total = 0;
    return
end
left = y(1:end-2);
mid = y(2:end-1);
right = y(3:end);
usable = isfinite(left) & isfinite(mid) & isfinite(right);
isPeak = mid > left & mid > right & mid >= peakDefMin;
isValley = mid < valleyNeighborFractionMax .* left & mid < valleyNeighborFractionMax .* right;
count = nnz(usable & (isPeak | isValley));
total = nnz(usable);
if total > 0
    fraction = count / total;
else
    fraction = NaN;
end
end

function [fraction, count, total] = alternating_phase_fraction(def1d, peakDefMin, valleyNeighborFractionMax)
y = double(def1d(:));
if numel(y) < 3
    fraction = NaN;
    count = 0;
    total = 0;
    return
end
left = y(1:end-2);
mid = y(2:end-1);
right = y(3:end);
usable = isfinite(left) & isfinite(mid) & isfinite(right);
isPeak = mid > left & mid > right & mid >= peakDefMin;
isValley = mid < valleyNeighborFractionMax .* left & mid < valleyNeighborFractionMax .* right;
interiorIndex = (2:numel(y)-1)';
oddInterior = mod(interiorIndex, 2) == 1;

patternOddPeak = (isPeak & oddInterior) | (isValley & ~oddInterior);
patternEvenPeak = (isPeak & ~oddInterior) | (isValley & oddInterior);
countOddPeak = nnz(usable & patternOddPeak);
countEvenPeak = nnz(usable & patternEvenPeak);
count = max(countOddPeak, countEvenPeak);
total = nnz(usable & (isPeak | isValley));
if total > 0
    fraction = count / total;
else
    fraction = NaN;
end
end

function sampleMap = init_sample_map(categories)
sampleMap = struct();
for k = 1:numel(categories)
    sampleMap.(categories{k}) = table();
end
end

function sampleMap = maybe_add_samples(sampleMap, m, filePath, mode, recordIndex, timeUnix)
cats = categories_for_metric(m);
for c = 1:numel(cats)
    cat = cats{c};
    tbl = sampleMap.(cat);
    if height(tbl) >= 2
        continue
    end
    flagBit1SparseCaution = bitand(m.flag, uint32(1)) ~= 0;
    flagBit2SparseInvalid = bitand(m.flag, uint32(2)) ~= 0;
    flagBit3UvContamination = bitand(m.flag, uint32(4)) ~= 0;
    flagBit4EvenOddError = bitand(m.flag, uint32(8)) ~= 0;
    flagBit5HighChannel = bitand(m.flag, uint32(16)) ~= 0;
    sampleMap.(cat) = [tbl; table(string(cat), string(filePath), recordIndex, timeUnix, mode, m.flag, ...
        flag_binary_string(m.flag), flagBit1SparseCaution, flagBit2SparseInvalid, flagBit3UvContamination, ...
        flagBit4EvenOddError, flagBit5HighChannel, m.valid2dCount, m.nonzeroEnergyBins, m.nonzeroAngleBins, m.uvFraction, ...
        m.oddPositiveFraction, m.evenPositiveFraction, m.oddZeroFraction, m.evenZeroFraction, ...
        m.alternatingExtremaFraction, m.alternatingExtremaCount, m.alternatingExtremaTotal, ...
        m.alternatingPhaseFraction, m.alternatingPhaseCount, m.alternatingPhaseTotal, ...
        m.evenOddAllEnergyPass, m.evenOdd0To30Pass, m.evenOdd30To300Pass, m.evenOdd300ToMaxPass, ...
        m.evenOdd0To50Pass, m.evenOdd50To500Pass, m.evenOdd500To5000Pass, ...
        m.alternatingExtremaFraction0To30, m.alternatingExtremaCount0To30, m.alternatingExtremaTotal0To30, ...
        m.alternatingPhaseFraction0To30, m.alternatingPhaseCount0To30, m.alternatingPhaseTotal0To30, ...
        m.alternatingExtremaFraction30To300, m.alternatingExtremaCount30To300, m.alternatingExtremaTotal30To300, ...
        m.alternatingPhaseFraction30To300, m.alternatingPhaseCount30To300, m.alternatingPhaseTotal30To300, ...
        m.alternatingExtremaFraction300ToMax, m.alternatingExtremaCount300ToMax, m.alternatingExtremaTotal300ToMax, ...
        m.alternatingPhaseFraction300ToMax, m.alternatingPhaseCount300ToMax, m.alternatingPhaseTotal300ToMax, ...
        m.alternatingExtremaFraction0To50, m.alternatingExtremaCount0To50, m.alternatingExtremaTotal0To50, ...
        m.alternatingPhaseFraction0To50, m.alternatingPhaseCount0To50, m.alternatingPhaseTotal0To50, ...
        m.alternatingExtremaFraction50To500, m.alternatingExtremaCount50To500, m.alternatingExtremaTotal50To500, ...
        m.alternatingPhaseFraction50To500, m.alternatingPhaseCount50To500, m.alternatingPhaseTotal50To500, ...
        m.alternatingExtremaFraction500To5000, m.alternatingExtremaCount500To5000, m.alternatingExtremaTotal500To5000, ...
        m.alternatingPhaseFraction500To5000, m.alternatingPhaseCount500To5000, m.alternatingPhaseTotal500To5000, ...
        m.maxDef2d, m.maxDefEnergyEV, m.maxDefAngleIndex, ...
        'VariableNames', {'category', 'source_file', 'record_index', 'time_unix_s', 'mode', 'quality_flag', ...
        'quality_flag_binary', 'flag_bit1_sparse_caution', 'flag_bit2_sparse_invalid', 'flag_bit3_uv_contamination', ...
        'flag_bit4_evenodd_error', 'flag_bit5_high_channel', 'valid_2d_channel_count', ...
        'nonzero_energy_bin_count', 'nonzero_angle_bin_count', 'uv_fraction', ...
        'odd_positive_fraction', 'even_positive_fraction', 'odd_zero_fraction', 'even_zero_fraction', ...
        'alternating_extrema_fraction', 'alternating_extrema_count', 'alternating_extrema_total', ...
        'alternating_phase_fraction', 'alternating_phase_count', 'alternating_phase_total', ...
        'flag_bit4_evenodd_all_energy', 'flag_bit4_evenodd_0_30eV', 'flag_bit4_evenodd_30_300eV', 'flag_bit4_evenodd_300_maxeV', ...
        'flag_bit4_evenodd_0_50eV', 'flag_bit4_evenodd_50_500eV', 'flag_bit4_evenodd_500_5000eV', ...
        'alternating_extrema_fraction_0_30eV', 'alternating_extrema_count_0_30eV', 'alternating_extrema_total_0_30eV', ...
        'alternating_phase_fraction_0_30eV', 'alternating_phase_count_0_30eV', 'alternating_phase_total_0_30eV', ...
        'alternating_extrema_fraction_30_300eV', 'alternating_extrema_count_30_300eV', 'alternating_extrema_total_30_300eV', ...
        'alternating_phase_fraction_30_300eV', 'alternating_phase_count_30_300eV', 'alternating_phase_total_30_300eV', ...
        'alternating_extrema_fraction_300_maxeV', 'alternating_extrema_count_300_maxeV', 'alternating_extrema_total_300_maxeV', ...
        'alternating_phase_fraction_300_maxeV', 'alternating_phase_count_300_maxeV', 'alternating_phase_total_300_maxeV', ...
        'alternating_extrema_fraction_0_50eV', 'alternating_extrema_count_0_50eV', 'alternating_extrema_total_0_50eV', ...
        'alternating_phase_fraction_0_50eV', 'alternating_phase_count_0_50eV', 'alternating_phase_total_0_50eV', ...
        'alternating_extrema_fraction_50_500eV', 'alternating_extrema_count_50_500eV', 'alternating_extrema_total_50_500eV', ...
        'alternating_phase_fraction_50_500eV', 'alternating_phase_count_50_500eV', 'alternating_phase_total_50_500eV', ...
        'alternating_extrema_fraction_500_5000eV', 'alternating_extrema_count_500_5000eV', 'alternating_extrema_total_500_5000eV', ...
        'alternating_phase_fraction_500_5000eV', 'alternating_phase_count_500_5000eV', 'alternating_phase_total_500_5000eV', ...
        'max_def_2d', 'max_def_energy_eV', 'max_def_angle_index'})]; %#ok<AGROW>
end
end

function s = flag_binary_string(flag)
bits = bitget(uint32(flag), 1:5);
s = string(sprintf('%d%d%d%d%d', bits));
end

function cats = categories_for_metric(m)
cats = {};
if m.flag == 0
    cats{end+1} = 'normal'; %#ok<AGROW>
end
if bitand(m.flag, uint32(2)) ~= 0
    cats{end+1} = 'sparse_invalid'; %#ok<AGROW>
end
if bitand(m.flag, uint32(1)) ~= 0
    cats{end+1} = 'sparse_caution'; %#ok<AGROW>
end
if bitand(m.flag, uint32(4)) ~= 0
    cats{end+1} = 'uv_contamination'; %#ok<AGROW>
end
if bitand(m.flag, uint32(8)) ~= 0
    cats{end+1} = 'evenodd_error'; %#ok<AGROW>
end
if bitand(m.flag, uint32(16)) ~= 0
    cats{end+1} = 'high_channel'; %#ok<AGROW>
end
end

function n = sample_count(sampleMap)
fn = fieldnames(sampleMap);
n = 0;
for k = 1:numel(fn)
    n = n + height(sampleMap.(fn{k}));
end
end

function samples = sample_map_to_table(sampleMap)
fn = fieldnames(sampleMap);
samples = table();
for k = 1:numel(fn)
    samples = [samples; sampleMap.(fn{k})]; %#ok<AGROW>
end
if ~isempty(samples)
    timeUtc = strings(height(samples), 1);
    for i = 1:height(samples)
        timeUtc(i) = string(datetime(samples.time_unix_s(i), ConvertFrom='posixtime', TimeZone='UTC', Format='yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
    end
    samples.time_utc = timeUtc;
    samples = movevars(samples, 'time_utc', 'After', 'time_unix_s');
end
end

function out = summarize_file(rows, name)
out = table(string(name), height(rows), nnz(rows.quality_flag == 0), ...
    nnz(bitand(rows.quality_flag, uint32(1)) ~= 0), ...
    nnz(bitand(rows.quality_flag, uint32(2)) ~= 0), ...
    nnz(bitand(rows.quality_flag, uint32(4)) ~= 0), ...
    nnz(bitand(rows.quality_flag, uint32(8)) ~= 0), ...
    nnz(bitand(rows.quality_flag, uint32(16)) ~= 0), ...
    'VariableNames', {'source_file', 'records', 'normal_count', 'sparse_caution_count', ...
    'sparse_invalid_count', 'uv_contamination_count', 'evenodd_error_count', 'high_channel_count'});
end

function plot_sample_examples(samples, cfg, paths)
for i = 1:height(samples)
    try
        sample = samples(i, :);
        plot_one_sample(sample, cfg, paths);
    catch ME
        warning('Could not plot sample %d: %s', i, ME.message);
    end
end
end

function plot_one_sample(sample, cfg, paths)
filePath = char(sample.source_file);
mode = sample.mode;
geom = minpa_geometry(mode);
groups = species_groups(geom.masses, cfg.species, mode);
s = load(filePath, 'tw1_MINPA_data');
[times, rawRows] = read_times_and_counts(s.tw1_MINPA_data, mode);
[~, idx] = min(abs(times - sample.time_unix_s));
[dpf2d, def2d, def1d] = extract_species_h_cube(rawRows(idx, :), geom, groups);

[density, velocity, speed] = lookup_nv_mso2_h(paths.nvMso2Dir, sample.time_unix_s);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1300, 760]);
layout = tiledlayout(fig, 2, 2, TileSpacing='compact', Padding='compact');

ax1 = nexttile(layout, 1);
imagesc(ax1, 1:size(def2d, 2), log10(geom.energy), log10(def2d + 1));
colorbar(ax1);
xlabel(ax1, 'Angle bin');
ylabel(ax1, 'log_{10}(Energy/eV)');
title(ax1, 'H^+ 2D DEF log_{10}(DEF+1)');

ax2 = nexttile(layout, 2);
semilogy(ax2, geom.energy, max(def1d, 1), '-o', MarkerSize=3, LineWidth=0.9);
grid(ax2, 'on');
xlabel(ax2, 'Energy (eV)');
ylabel(ax2, '1D DEF');
title(ax2, 'Solid-angle weighted H^+ spectrum');

ax3 = nexttile(layout, 3);
bar(ax3, [sample.valid_2d_channel_count, sample.nonzero_energy_bin_count, sample.nonzero_angle_bin_count]);
ax3.XTickLabel = {'valid 2D', 'nonzero E', 'nonzero angle'};
ylabel(ax3, 'Count');
title(ax3, 'Effective channel metrics');
grid(ax3, 'on');

ax4 = nexttile(layout, 4);
axis(ax4, 'off');
info = {
    sprintf('Category: %s', sample.category)
    sprintf('UTC: %s', sample.time_utc)
    sprintf('Mode: %g, record index: %d', sample.mode, sample.record_index)
    sprintf('Flag bitmask: %d', sample.quality_flag)
    sprintf('Valid 2D channels: %d', sample.valid_2d_channel_count)
    sprintf('UV fraction: %.3f', sample.uv_fraction)
    sprintf('Alternating extrema: %.3f (%g/%g)', sample.alternating_extrema_fraction, sample.alternating_extrema_count, sample.alternating_extrema_total)
    sprintf('Alternating phase: %.3f (%g/%g)', sample.alternating_phase_fraction, sample.alternating_phase_count, sample.alternating_phase_total)
    sprintf('Odd pos / even pos: %.3f / %.3f', sample.odd_positive_fraction, sample.even_positive_fraction)
    sprintf('Max 2D DEF: %.3g at %.3g eV, angle %g', sample.max_def_2d, sample.max_def_energy_eV, sample.max_def_angle_index)
    sprintf('NV_MSO_2 H density: %.4g cm^{-3}', density)
    sprintf('NV_MSO_2 H velocity: [%.3g, %.3g, %.3g] km/s', velocity(1), velocity(2), velocity(3))
    sprintf('NV_MSO_2 H speed: %.3g km/s', speed)
    };
text(ax4, 0.02, 0.98, info, VerticalAlignment='top', FontName='Consolas', FontSize=10);

title(layout, sprintf('TW-1 MINPA H^+ Quality Flag Example: %s', sample.category), FontWeight='bold');
safeCat = regexprep(char(sample.category), '[^\w]+', '_');
timeTag = datestr(datetime(sample.time_unix_s, ConvertFrom='posixtime', TimeZone='UTC'), 'yyyymmdd_HHMMSSFFF');
png = fullfile(paths.figureDir, sprintf('%s_%s_flag%d.png', safeCat, timeTag, sample.quality_flag));
exportgraphics(fig, png, Resolution=180);
close(fig);
end

function [density, velocity, speed] = lookup_nv_mso2_h(nvDir, timeUnix)
day = datestr(datetime(timeUnix, ConvertFrom='posixtime', TimeZone='UTC'), 'yyyymmdd');
path = fullfile(nvDir, ['NV_', day, '.mat']);
density = NaN;
velocity = [NaN, NaN, NaN];
speed = NaN;
if ~isfile(path)
    return
end
s = load(path, 'NH_TW1', 'VH_TW1_MSO');
[dt, idx] = min(abs(s.NH_TW1(:, 1) - timeUnix));
if isempty(idx) || dt > 1e-3
    return
end
density = s.NH_TW1(idx, 2);
velocity = s.VH_TW1_MSO(idx, 2:4);
speed = sqrt(sum(velocity.^2));
end

function geom = minpa_geometry(mode)
energy = minpa_energy(mode);
masses = minpa_masses(mode);
pEdges = pitch_edges(mode);
if isempty(pEdges)
    geom = struct('energy', [], 'masses', [], 'massLen', 0, 'omega', []);
    return
end
if mode == 12
    nAzimuth = 1;
else
    nAzimuth = 16;
end
azEdges = linspace(0, 360, nAzimuth + 1);
dphi = deg2rad(360 / nAzimuth);
omegaPitch = (cosd(pEdges(1:end-1) + 90) - cosd(pEdges(2:end) + 90)) * dphi;
geom.energy = energy(:);
geom.masses = masses(:);
geom.massLen = numel(masses);
geom.omega = repelem(omegaPitch(:), nAzimuth);
geom.azimuth = repmat((azEdges(1:end-1) + azEdges(2:end))' / 2, numel(omegaPitch), 1);
end

function groups = species_groups(masses, species, mode)
switch species
    case 'H'
        target = 1;
        fallback = [1, 2];
    case 'O'
        target = 16;
        fallback = [19, 20];
    case 'O2'
        target = 32;
        fallback = [27, 28];
end
groups = find(abs(masses - target) < 1e-6);
if isempty(groups) && mode == 12
    groups = fallback;
end
end

function e = minpa_energy(mode)
switch mode
    case {1, 2}
        e = [2.81,3.548928,4.482167,5.660813,7.149401,9.029433,11.40385,14.40264,18.19001,22.97332,29.01447,36.64422,46.28032,58.45036,73.82067,93.23282,117.7497,148.7135,187.8198,237.2095,299.587,378.3675,477.8644,603.5253,762.2305,962.6694,1215.816,1535.532,1939.321,2449.292,3093.366,3906.809,4934.157,6231.661,7870.361,9939.98,12553.83,15855.03,20024.33,25290];
    case {3, 4, 5, 7, 8}
        e = [2.81,3.246924,3.751784,4.335145,5.009211,5.788088,6.688071,7.727991,8.929607,10.31806,11.9224,13.77621,15.91825,18.39336,21.25332,24.55798,28.37647,32.7887,37.88697,43.77797,50.58496,58.45036,67.53873,78.04025,90.17464,104.1958,120.3971,139.1175,160.7487,185.7433,214.6243,247.996,286.5566,331.113,382.5974,442.087,510.8266,590.2544,682.0324,788.0808,910.6185,1052.21,1215.816,1404.862,1623.303,1875.708,2167.36,2504.36,2893.76,3343.707,3863.617,4464.366,5158.525,5960.618,6887.428,7958.346,9195.78,10625.62,12277.79,14186.84,16392.74,18941.63,21886.84,25290];
    case 6
        e = [2.81,3.410666,4.139729,5.024638,6.098705,7.402364,8.984693,10.90526,13.23637,16.06578,19.5,23.66831,28.72765,34.86848,42.32196,51.3687,62.34928,75.67706,91.85379,111.4885,135.3202,164.2463,199.3556,241.9698,293.6933,356.4731,432.6728,525.161,637.4194,773.6741,939.0547,1139.787,1383.428,1679.149,2038.084,2473.745,3002.533,3644.355,4423.372,5368.912,6516.57,7909.553,9600.298,11652.46,14143.29,17166.56,20836.08,25290];
    case {9, 10, 11}
        e = [44.96,49.67818,54.89149,60.65189,67.0168,74.04965,81.82054,90.40692,99.89437,110.3775,121.9606,134.7594,148.9013,164.5272,181.793,200.8706,221.9503,245.2421,270.9783,299.4152,330.8363,365.5548,403.9167,446.3044,493.1403,544.8912,602.073,665.2555,735.0686,812.2079,897.4423,991.6213,1095.684,1210.667,1337.716,1478.098,1633.212,1804.604,1993.982,2203.234,2434.445,2689.919,2972.204,3284.112,3628.752,4009.559,4430.329,4895.255,5408.971,5976.597,6603.791,7296.804,8062.543,8908.639,9843.526,10876.52,12017.92,13279.1,14672.63,16212.4,17913.76,19793.66,21870.84,24166];
    case 12
        e = [2.81,3.267539,3.799578,4.418245,5.137647,5.974187,6.946936,8.078073,9.393388,10.92287,12.70139,14.7695,17.17435,19.97077,23.22251,27.00373,31.40062,36.51344,42.45875,49.37211,57.41115,66.75914,77.62922,90.26922,104.9673,122.0587,141.9329,165.0432,191.9164,223.1653,259.5023,301.7558,350.8893,408.023,474.4595,551.7135,641.5465,746.0065,867.4753,1008.722,1172.968,1363.957,1586.043,1844.292,2144.589,2493.783,2899.834,3372];
    otherwise
        e = [];
end
end

function m = minpa_masses(mode)
if mode >= 1 && mode <= 6
    m = [1, 2, 4, 16, 38, 32, 44, 65];
elseif mode == 7 || mode == 8
    m = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 22, 28, 32, 44, 69];
elseif mode >= 9 && mode <= 11
    m = 1;
elseif mode == 12
    m = [0.76,1.25,1.75,2.25,2.75,3.25,3.75,4.26,5.66,6.34,7.65,8.35,9.62,10.38,11.61,12.39,13.55,14.47,15.56,16.45,17.45,18.58,20.98,23.04,26.97,28.92,30.94,33.17,40.07,48.12,64.68,68.95];
else
    m = [];
end
end

function p = pitch_edges(mode)
if mode >= 1 && mode <= 6
    p = linspace(0, 90, 5);
elseif mode >= 7 && mode <= 11
    p = linspace(0, 90, 17);
elseif mode == 12
    p = [67.4, 90];
else
    p = [];
end
end
