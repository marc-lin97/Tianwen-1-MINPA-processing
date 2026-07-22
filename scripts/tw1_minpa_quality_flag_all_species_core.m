function out = tw1_minpa_quality_flag_all_species_core(filePath, energyEV, timeRangeUnix)
%TW1_MINPA_QUALITY_FLAG_ALL_SPECIES_CORE Compute three species flags in one read.
%
% filePath is read-only. energyEV is the mode-specific center-energy vector
% from the matching day_spe product. timeRangeUnix is [start, stop), or [].

if nargin < 3
    timeRangeUnix = [];
end

cfg = tw1_minpa_quality_flag_config();
mode = parse_mode(filePath);
[masses, pitchEdges, azimuthCount] = mode_geometry(mode);
energyEV = double(energyEV(:));
if isempty(masses) || isempty(pitchEdges) || isempty(energyEV)
    error('Unsupported MINPA mode %g in %s.', mode, filePath);
end

s = load(filePath, 'tw1_MINPA_data');
if ~isfield(s, 'tw1_MINPA_data')
    error('Missing tw1_MINPA_data in %s.', filePath);
end
[times, rawRows, recordIndex] = read_times_and_counts(s.tw1_MINPA_data, mode);

keep = isfinite(times);
if ~isempty(timeRangeUnix)
    keep = keep & times >= timeRangeUnix(1) & times < timeRangeUnix(2);
end
times = times(keep);
rawRows = rawRows(keep, :);
recordIndex = recordIndex(keep);

dphi = deg2rad(360 / azimuthCount);
omegaPitch = (cosd(pitchEdges(1:end-1) + 90) - ...
    cosd(pitchEdges(2:end) + 90)) * dphi;
omega = repelem(omegaPitch(:), azimuthCount);
nAngle = numel(omega);
nEnergy = numel(energyEV);
expectedWidth = numel(masses) * nAngle * nEnergy;
if ~isempty(rawRows) && size(rawRows, 2) ~= expectedWidth
    error(['Raw row width mismatch in %s: got %d, expected %d ', ...
        '(mode=%d, mass=%d, energy=%d, angle=%d).'], ...
        filePath, size(rawRows, 2), expectedWidth, mode, numel(masses), nEnergy, nAngle);
end

out = struct();
out.source_file = filePath;
out.mode = mode;
out.time_unix_s = times(:);
out.record_index = recordIndex(:);
out.energy_eV = energyEV;
out.species = struct();

for k = 1:numel(cfg.species)
    species = cfg.species{k};
    groups = species_groups(masses, species, mode);
    if isempty(groups)
        result = empty_species_result(numel(times), false);
    else
        result = compute_species_flags(rawRows, energyEV, omega, ...
            numel(masses), groups, cfg);
    end
    out.species.(species) = result;
end
end

function result = compute_species_flags(rawRows, energy, omega, massLen, groups, cfg)
nRecord = size(rawRows, 1);
nEnergy = numel(energy);
nAngle = numel(omega);
nChannel = nEnergy * nAngle;

dpfFlat = zeros(nRecord, nChannel);
for g = groups(:)'
    oneMass = rawRows(:, g:massLen:end);
    if size(oneMass, 2) ~= nChannel
        error('Species extraction produced %d channels; expected %d.', ...
            size(oneMass, 2), nChannel);
    end
    dpfFlat = dpfFlat + oneMass;
end

defFlat = dpfFlat .* repelem(energy(:)', nAngle);
weightMatrix = zeros(nChannel, nEnergy);
omegaSum = sum(omega, 'omitnan');
for e = 1:nEnergy
    idx = (e - 1) * nAngle + (1:nAngle);
    weightMatrix(idx, e) = omega ./ omegaSum;
end
defForSum = defFlat;
defForSum(isnan(defForSum)) = 0;
def1d = defForSum * weightMatrix;

valid2d = isfinite(dpfFlat) ...
    & dpfFlat > cfg.thresholds.valid_raw_channel_min ...
    & dpfFlat < cfg.thresholds.valid_raw_channel_max_exclusive;
validCount = sum(valid2d, 2);
uvFraction = mean(isfinite(def1d) ...
    & def1d > cfg.thresholds.uv_def_threshold, 2);
maxDef2d = max(defFlat, [], 2, 'omitnan');

bands = cfg.thresholds.alternating_energy_bands_eV;
bandPass = false(nRecord, size(bands, 1));
for b = 1:size(bands, 1)
    if b == 1
        mask = true(size(energy));
    else
        mask = energy >= bands(b, 1) & energy < bands(b, 2);
    end
    bandPass(:, b) = alternating_band_pass(def1d(:, mask), cfg);
end
evenOddPass = any(bandPass, 2);

flag = zeros(nRecord, 1, 'uint32');
sparseInvalid = validCount < cfg.thresholds.sparse_invalid_lt;
sparseCaution = validCount >= cfg.thresholds.sparse_caution_min ...
    & validCount <= cfg.thresholds.sparse_caution_max;
flag(sparseInvalid) = bitor(flag(sparseInvalid), cfg.flag_bits.sparse_invalid);
flag(sparseCaution) = bitor(flag(sparseCaution), cfg.flag_bits.sparse_caution);
uv = uvFraction > cfg.thresholds.uv_fraction_threshold;
flag(uv) = bitor(flag(uv), cfg.flag_bits.uv_contamination);
flag(evenOddPass) = bitor(flag(evenOddPass), cfg.flag_bits.evenodd_error);
high = isfinite(maxDef2d) & maxDef2d > cfg.thresholds.high_channel_def_threshold;
flag(high) = bitor(flag(high), cfg.flag_bits.high_channel);

result = struct();
result.available = true(nRecord, 1);
result.flag = flag;
result.bits = false(nRecord, 5);
for bitIndex = 1:5
    result.bits(:, bitIndex) = logical(bitget(flag, bitIndex));
end
result.binary = flag_binary_strings(flag, cfg.missing_binary, result.available);
result.valid_2d_channel_count = uint32(validCount);
result.uv_fraction = uvFraction;
result.evenodd_band_pass = bandPass;
result.max_def_2d = maxDef2d;
end

function pass = alternating_band_pass(y, cfg)
nRecord = size(y, 1);
if size(y, 2) < 3
    pass = false(nRecord, 1);
    return
end

left = y(:, 1:end-2);
mid = y(:, 2:end-1);
right = y(:, 3:end);
usable = isfinite(left) & isfinite(mid) & isfinite(right);
isPeak = mid > left & mid > right ...
    & mid >= cfg.thresholds.alternating_peak_def_min;
isValley = mid < cfg.thresholds.alternating_valley_neighbor_fraction_max .* left ...
    & mid < cfg.thresholds.alternating_valley_neighbor_fraction_max .* right;
isExtrema = usable & (isPeak | isValley);

extremaCount = sum(isExtrema, 2);
extremaTotal = sum(usable, 2);
extremaFraction = extremaCount ./ extremaTotal;
extremaFraction(extremaTotal == 0) = NaN;

interiorIndex = 2:(size(y, 2) - 1);
oddInterior = mod(interiorIndex, 2) == 1;
patternOddPeak = (isPeak & oddInterior) | (isValley & ~oddInterior);
patternEvenPeak = (isPeak & ~oddInterior) | (isValley & oddInterior);
countOddPeak = sum(usable & patternOddPeak, 2);
countEvenPeak = sum(usable & patternEvenPeak, 2);
phaseCount = max(countOddPeak, countEvenPeak);
phaseFraction = phaseCount ./ extremaCount;
phaseFraction(extremaCount == 0) = NaN;

pass = isfinite(extremaFraction) ...
    & extremaFraction >= cfg.thresholds.alternating_extrema_fraction_threshold ...
    & isfinite(phaseFraction) ...
    & phaseFraction >= cfg.thresholds.alternating_phase_fraction_threshold;
end

function result = empty_species_result(n, available)
cfg = tw1_minpa_quality_flag_config();
result = struct();
result.available = repmat(logical(available), n, 1);
result.flag = zeros(n, 1, 'uint32');
result.bits = false(n, 5);
result.binary = repmat(string(cfg.missing_binary), n, 1);
result.valid_2d_channel_count = zeros(n, 1, 'uint32');
result.uv_fraction = nan(n, 1);
result.evenodd_band_pass = false(n, 7);
result.max_def_2d = nan(n, 1);
end

function binary = flag_binary_strings(flag, missingBinary, available)
lookup = strings(32, 1);
for value = 0:31
    bits = bitget(uint32(value), 1:5);
    lookup(value + 1) = string(sprintf('%d%d%d%d%d', bits));
end
binary = lookup(double(flag) + 1);
binary(~available) = string(missingBinary);
end

function mode = parse_mode(filePath)
[~, name, ext] = fileparts(filePath);
tok = regexp([name, ext], 'MINPA-MOD(\d+)-', 'tokens', 'once');
if isempty(tok)
    mode = NaN;
else
    mode = str2double(tok{1});
end
end

function [times, rawRows, recordIndex] = read_times_and_counts(a, mode)
utc = squeeze(struct2cell(a(1).value));
times0 = nan(numel(utc), 1);
for i = 1:numel(utc)
    txt = string(utc{i});
    if strlength(txt) <= 20
        continue
    end
    try
        times0(i) = posixtime(datetime(txt, ...
            InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''', TimeZone='UTC'));
    catch
        times0(i) = posixtime(datetime(txt, ...
            InputFormat='yyyy-MM-dd''T''HH:mm:ss.SSS''Z''', TimeZone='UTC'));
    end
end

dataValue = a(42).value;
nBase = min(numel(times0), numel(dataValue));
times0 = times0(1:nBase);
dataValue = dataValue(1:nBase);
if nBase == 0
    times = zeros(0, 1);
    rawRows = zeros(0, 0);
    recordIndex = zeros(0, 1);
    return
end

if mode == 12
    tempValue = dataValue(1).t;
    rawRows = nan(numel(times0) * 2, numel(tempValue) / 2);
    times = sort([times0 + 4.1 / 4; times0 + 4.1 * 3 / 4]);
    recordIndex = repelem((1:numel(times0))', 2);
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
    recordIndex = (1:numel(times0))';
    for i = 1:numel(times0)
        row = double(dataValue(i).t);
        if numel(row) == size(rawRows, 2)
            rawRows(i, :) = row;
        end
    end
end
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
    otherwise
        error('Unknown species %s.', species);
end
groups = find(abs(masses - target) < 1e-6);
if isempty(groups) && mode == 12
    groups = fallback;
end
end

function [masses, pitchEdges, azimuthCount] = mode_geometry(mode)
if mode >= 1 && mode <= 6
    masses = [1, 2, 4, 16, 38, 32, 44, 65];
    pitchEdges = linspace(0, 90, 5);
    azimuthCount = 16;
elseif mode == 7 || mode == 8
    masses = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 22, 28, 32, 44, 69];
    pitchEdges = linspace(0, 90, 17);
    azimuthCount = 16;
elseif mode >= 9 && mode <= 11
    masses = 1;
    pitchEdges = linspace(0, 90, 17);
    azimuthCount = 16;
elseif mode == 12
    masses = [0.76,1.25,1.75,2.25,2.75,3.25,3.75,4.26,5.66,6.34, ...
        7.65,8.35,9.62,10.38,11.61,12.39,13.55,14.47,15.56,16.45, ...
        17.45,18.58,20.98,23.04,26.97,28.92,30.94,33.17,40.07,48.12,64.68,68.95];
    pitchEdges = [67.4, 90];
    azimuthCount = 1;
else
    masses = [];
    pitchEdges = [];
    azimuthCount = 0;
end
end
