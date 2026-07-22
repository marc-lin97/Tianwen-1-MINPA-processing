function summary = verify_tw1_minpa_velocity_mso_rotation(dateToken, outputDir)
%VERIFY_TW1_MINPA_VELOCITY_MSO_ROTATION Validate MINPA Probe-to-MSO velocity rotation.
%
% This function only reads the legacy Tianwen-1 MINPA NV/NV_MSO products and
% MOMAG attitude. Derived JSON and CSV files are written below outputDir,
% which defaults to this repository's outputs directory.

if nargin < 1 || strlength(string(dateToken)) == 0
    dateToken = '20211203';
end
if nargin < 2 || strlength(string(outputDir)) == 0
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(repoRoot, 'outputs', 'tw1_minpa_velocity_mso_rotation_validation');
end

dateToken = char(dateToken);
outputDir = char(outputDir);

nvPath = fullfile('D:\Data\TW-1\result\MINPA\NV', ['NV_', dateToken, '.mat']);
nvMsoPath = fullfile('D:\Data\TW-1\result\MINPA\NV_MSO', ['NV_', dateToken, '.mat']);
momagPath = fullfile('D:\Data\TW-1\result\MOMAG\C\01Hz_all', ['Bss', dateToken, '.mat']);
assert(isfile(nvPath), 'Input file does not exist: %s', nvPath);
assert(isfile(nvMsoPath), 'Input file does not exist: %s', nvMsoPath);
assert(isfile(momagPath), 'Input file does not exist: %s', momagPath);

sourcesBefore = [dir(nvPath), dir(nvMsoPath), dir(momagPath)];

nv = load(nvPath, 'NH_TW1', 'NO_TW1', 'NO2_TW1', 'VH_TW1', 'VO_TW1', 'VO2_TW1');
nvMso = load(nvMsoPath, 'VH_TW1_MSO', 'VO_TW1_MSO', 'VO2_TW1_MSO');
att = load(momagPath, 'Pitch_TW1', 'Roll_TW1', 'Yaw_TW1');

tMomag = att.Pitch_TW1(:, 1);
[tMomag, order] = sort(tMomag(:));
pitchDeg = att.Pitch_TW1(order, 2);
rollDeg = att.Roll_TW1(order, 2);
yawDeg = att.Yaw_TW1(order, 2);

species = {
    'Hplus',  'NH_TW1',  'VH_TW1',  'VH_TW1_MSO';
    'Oplus',  'NO_TW1',  'VO_TW1',  'VO_TW1_MSO';
    'O2plus', 'NO2_TW1', 'VO2_TW1', 'VO2_TW1_MSO'
};

if ~isfolder(outputDir)
    mkdir(outputDir);
end

summary = struct;
summary.date = dateToken;
summary.source_files = struct( ...
    'nv_probe_like', nvPath, ...
    'nv_mso_stored', nvMsoPath, ...
    'momag_attitude', momagPath);
summary.rotation_convention = ...
    'C_MSO_from_Probe = Rz(-yaw) * Ry(-pitch) * Rx(-roll)';
summary.minpa_pre_rotation_mapping = ...
    'The script reports both the TW1_MINPA_redeal_3.m mapping [t,-V3,-V1,V2] and the best signed permutation from local NV columns to the stored NV_MSO product.';
summary.attitude_match_tolerance_s = 5.0;
summary.species = struct;

sampleRows = {};
sampleLimitPerSpecies = 12;

for s = 1:size(species, 1)
    speciesName = species{s, 1};
    densityName = species{s, 2};
    probeName = species{s, 3};
    msoName = species{s, 4};

    tMinpa = nv.(densityName)(:, 1);
    vLegacy = nv.(probeName);
    vRaw = vLegacy(:, 2:4);
    vStored = nvMso.(msoName);

    assert(size(vRaw, 1) == size(vStored, 1), ...
        'Probe and MSO velocity row counts differ for %s.', speciesName);

    nearest = interp1(tMomag, (1:numel(tMomag)).', tMinpa, 'nearest', NaN);
    nearest = round(nearest);
    hasNearest = isfinite(nearest) & nearest >= 1 & nearest <= numel(tMomag);
    dt = NaN(size(tMinpa));
    dt(hasNearest) = abs(tMinpa(hasNearest) - tMomag(nearest(hasNearest)));

    valid = hasNearest & dt < 5.0 & ...
        all(isfinite(vRaw), 2) & all(isfinite(vStored(:, 2:4)), 2) & ...
        all(abs([vRaw, vStored(:, 2:4)]) < 1.0e29, 2);

    [legacyProbe, legacyMappingText] = applySourceMapping(vRaw, [3, 1, 2], [-1, -1, 1]);
    legacyCalculated = calculateMso(legacyProbe, valid, nearest, rollDeg, pitchDeg, yawDeg);
    legacyStats = comparisonStats(legacyCalculated, vStored(:, 2:4), valid, dt);

    perms3 = perms(1:3);
    signs = signedTriplets();
    bestRmse = Inf;
    bestProbe = legacyProbe;
    bestCalculated = legacyCalculated;
    bestMappingText = legacyMappingText;
    bestStats = legacyStats;
    for p = 1:size(perms3, 1)
        for q = 1:size(signs, 1)
            [candidateProbe, candidateText] = applySourceMapping(vRaw, perms3(p, :), signs(q, :));
            candidateCalculated = calculateMso(candidateProbe, valid, nearest, rollDeg, pitchDeg, yawDeg);
            candidateStats = comparisonStats(candidateCalculated, vStored(:, 2:4), valid, dt);
            if candidateStats.vector_rmse_km_s < bestRmse
                bestRmse = candidateStats.vector_rmse_km_s;
                bestProbe = candidateProbe;
                bestCalculated = candidateCalculated;
                bestMappingText = candidateText;
                bestStats = candidateStats;
            end
        end
    end

    stats = struct;
    stats.record_count = size(vRaw, 1);
    stats.attitude_matched_count = nnz(hasNearest & dt < 5.0);
    stats.compared_count = bestStats.compared_count;
    stats.legacy_redeal_3_source_to_probe_mapping = legacyMappingText;
    stats.legacy_redeal_3 = legacyStats;
    stats.best_source_to_probe_mapping = bestMappingText;
    stats.best_match = bestStats;
    summary.species.(speciesName) = stats;

    validIndex = find(valid);
    if ~isempty(validIndex)
        take = validIndex(round(linspace(1, numel(validIndex), min(sampleLimitPerSpecies, numel(validIndex)))));
        for n = 1:numel(take)
            i = take(n);
            j = nearest(i);
            row = {
                speciesName, i, tMinpa(i), dt(i), ...
                rollDeg(j), pitchDeg(j), yawDeg(j), ...
                bestMappingText, ...
                bestProbe(i, 1), bestProbe(i, 2), bestProbe(i, 3), ...
                bestCalculated(i, 1), bestCalculated(i, 2), bestCalculated(i, 3), ...
                vStored(i, 2), vStored(i, 3), vStored(i, 4), ...
                norm(bestCalculated(i, :) - vStored(i, 2:4))
            };
            sampleRows(end + 1, :) = row; %#ok<AGROW>
        end
    end
end

summaryPath = fullfile(outputDir, ['tw1_minpa_velocity_mso_rotation_', dateToken, '_summary.json']);
fid = fopen(summaryPath, 'w');
assert(fid >= 0, 'Could not create summary file: %s', summaryPath);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(summary, PrettyPrint=true));
clear cleanup;

sampleTable = cell2table(sampleRows, 'VariableNames', { ...
    'species', 'record_index', 'epoch_unix_s', 'attitude_match_dt_s', ...
    'roll_deg', 'pitch_deg', 'yaw_deg', ...
    'source_to_probe_mapping', ...
    'vx_probe_km_s', 'vy_probe_km_s', 'vz_probe_km_s', ...
    'vx_mso_calculated_km_s', 'vy_mso_calculated_km_s', 'vz_mso_calculated_km_s', ...
    'vx_mso_stored_km_s', 'vy_mso_stored_km_s', 'vz_mso_stored_km_s', ...
    'vector_error_km_s'});
writetable(sampleTable, fullfile(outputDir, ['tw1_minpa_velocity_mso_rotation_', dateToken, '_samples.csv']));

sourcesAfter = [dir(nvPath), dir(nvMsoPath), dir(momagPath)];
for k = 1:numel(sourcesBefore)
    assert(sourcesBefore(k).bytes == sourcesAfter(k).bytes && ...
        sourcesBefore(k).datenum == sourcesAfter(k).datenum, ...
        'Source file metadata changed during validation: %s', sourcesBefore(k).name);
end

fprintf('Validated MINPA velocity MSO rotation for %s. Summary: %s\n', dateToken, summaryPath);
end


function c = tw1ProbeToMsoMatrix(rollDeg, pitchDeg, yawDeg)
roll = deg2rad(rollDeg);
pitch = deg2rad(pitchDeg);
yaw = deg2rad(yawDeg);
cr = cos(roll); sr = sin(roll);
cp = cos(pitch); sp = sin(pitch);
cy = cos(yaw); sy = sin(yaw);

c = zeros(3, 3);
c(1, 1) = cy * cp;
c(1, 2) = cy * sp * sr + sy * cr;
c(1, 3) = sy * sr - cy * sp * cr;
c(2, 1) = -sy * cp;
c(2, 2) = cy * cr - sy * sp * sr;
c(2, 3) = sy * sp * cr + cy * sr;
c(3, 1) = sp;
c(3, 2) = -cp * sr;
c(3, 3) = cp * cr;
end


function signs = signedTriplets()
signs = [
    -1, -1, -1;
    -1, -1,  1;
    -1,  1, -1;
    -1,  1,  1;
     1, -1, -1;
     1, -1,  1;
     1,  1, -1;
     1,  1,  1
];
end


function [mapped, text] = applySourceMapping(raw, permutation, signs)
mapped = raw(:, permutation) .* signs;
parts = strings(1, 3);
for k = 1:3
    if signs(k) > 0
        prefix = '+';
    else
        prefix = '-';
    end
    parts(k) = prefix + "V" + string(permutation(k));
end
text = char("[" + strjoin(parts, ",") + "]");
end


function calculated = calculateMso(probeVelocity, valid, nearest, rollDeg, pitchDeg, yawDeg)
calculated = NaN(size(probeVelocity));
validIndex = find(valid);
for k = 1:numel(validIndex)
    i = validIndex(k);
    j = nearest(i);
    c = tw1ProbeToMsoMatrix(rollDeg(j), pitchDeg(j), yawDeg(j));
    calculated(i, :) = (c * probeVelocity(i, :).').';
end
end


function stats = comparisonStats(calculated, stored, valid, dt)
delta = calculated(valid, :) - stored(valid, :);
vectorError = vecnorm(delta, 2, 2);
calcNorm = vecnorm(calculated(valid, :), 2, 2);
storedNorm = vecnorm(stored(valid, :), 2, 2);

stats = struct;
stats.compared_count = nnz(valid);
stats.component_rmse_km_s = sqrt(mean(delta.^2, 1));
stats.vector_rmse_km_s = sqrt(mean(vectorError.^2));
stats.vector_error_median_km_s = median(vectorError);
stats.vector_error_p99_km_s = samplePercentile(vectorError, 99.0);
stats.vector_error_max_km_s = max(vectorError);
stats.max_abs_component_error_km_s = max(abs(delta), [], 'all');
stats.calculated_to_stored_speed_rmse_km_s = sqrt(mean((calcNorm - storedNorm).^2));
stats.max_attitude_match_dt_s = max(dt(valid));
end


function value = samplePercentile(values, percentile)
if isempty(values)
    value = NaN;
    return
end
sorted = sort(values(:));
position = 1.0 + (numel(sorted) - 1.0) * percentile / 100.0;
lower = floor(position);
upper = ceil(position);
if lower == upper
    value = sorted(lower);
else
    fraction = position - lower;
    value = sorted(lower) * (1.0 - fraction) + sorted(upper) * fraction;
end
end
