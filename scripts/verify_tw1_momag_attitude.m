function summary = verify_tw1_momag_attitude(dataPath, outputDir)
%VERIFY_TW1_MOMAG_ATTITUDE Validate the orbiter-body to MSO RPY transform.
%
% This function only reads dataPath. Derived JSON and CSV files are written
% below outputDir, which defaults to this repository's outputs directory.

if nargin < 1 || strlength(string(dataPath)) == 0
    error('A Tianwen-1 MOMAG MAT file path is required.');
end
if nargin < 2 || strlength(string(outputDir)) == 0
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(repoRoot, 'outputs', 'tw1_momag_attitude_validation');
end

dataPath = char(dataPath);
outputDir = char(outputDir);
assert(isfile(dataPath), 'Input file does not exist: %s', dataPath);
sourceBefore = dir(dataPath);

required = {
    'Probe_Attitude_Roll_MSO', 'Probe_Attitude_Pitch_MSO', ...
    'Probe_Attitude_Yaw_MSO', 'X_Orbiter', 'Y_Orbiter', 'Z_Orbiter', ...
    'X_MSO', 'Y_MSO', 'Z_MSO', 'S'
};
d = load(dataPath, required{:});

rollDeg = d.Probe_Attitude_Roll_MSO(:);
pitchDeg = d.Probe_Attitude_Pitch_MSO(:);
yawDeg = d.Probe_Attitude_Yaw_MSO(:);
bOrbiter = [d.X_Orbiter(:), d.Y_Orbiter(:), d.Z_Orbiter(:)];
bMsoStored = [d.X_MSO(:), d.Y_MSO(:), d.Z_MSO(:)];
nRecords = size(bOrbiter, 1);
assert(numel(rollDeg) == nRecords && numel(pitchDeg) == nRecords && ...
    numel(yawDeg) == nRecords && size(bMsoStored, 1) == nRecords, ...
    'Attitude and magnetic-field arrays must have the same record count.');

% Product convention for right-handed active rotations on column vectors:
% C_MSO<-Orbiter = Rz(-yaw) * Ry(-pitch) * Rx(-roll).
c = tw1OrbiterToMsoMatrix(rollDeg, pitchDeg, yawDeg);
bMsoCalculated = squeeze(pagemtimes(c, reshape(bOrbiter.', 3, 1, nRecords))).';

valid = all(isfinite([rollDeg, pitchDeg, yawDeg, bOrbiter, bMsoStored]), 2) & ...
    all(abs([rollDeg, pitchDeg, yawDeg, bOrbiter, bMsoStored]) < 1.0e29, 2);
assert(any(valid), 'No finite records are available for validation.');

delta = bMsoCalculated(valid, :) - bMsoStored(valid, :);
vectorError = vecnorm(delta, 2, 2);
calculatedNorm = vecnorm(bMsoCalculated(valid, :), 2, 2);
orbiterNorm = vecnorm(bOrbiter(valid, :), 2, 2);
storedNorm = vecnorm(bMsoStored(valid, :), 2, 2);

checkIndex = unique(round(linspace(1, nRecords, min(nRecords, 2000))));
orthogonalityError = zeros(numel(checkIndex), 1);
determinantError = zeros(numel(checkIndex), 1);
for k = 1:numel(checkIndex)
    q = c(:, :, checkIndex(k));
    orthogonalityError(k) = norm(q * q.' - eye(3), 'fro');
    determinantError(k) = abs(det(q) - 1.0);
end

summary = struct;
summary.source_file = dataPath;
summary.product = 'Tianwen-1 MOMAG DB 32 Hz SCI P';
summary.source_time_start_utc = char(d.S.Observation_Area.Time_Coordinates.start_date_time);
summary.source_time_stop_utc = char(d.S.Observation_Area.Time_Coordinates.stop_date_time);
summary.record_count = nRecords;
summary.valid_record_count = nnz(valid);
summary.attitude_angle_unit = 'deg';
summary.magnetic_field_unit_in_label = 'none';
summary.rotation_convention = ...
    'C_MSO_from_Orbiter = Rz(-yaw) * Ry(-pitch) * Rx(-roll)';
summary.roll_range_deg = [min(rollDeg(valid)), max(rollDeg(valid))];
summary.pitch_range_deg = [min(pitchDeg(valid)), max(pitchDeg(valid))];
summary.yaw_range_deg = [min(yawDeg(valid)), max(yawDeg(valid))];
summary.component_rmse_data_unit = sqrt(mean(delta.^2, 1));
summary.vector_rmse_data_unit = sqrt(mean(vectorError.^2));
summary.vector_error_median_data_unit = median(vectorError);
summary.vector_error_p99_data_unit = samplePercentile(vectorError, 99.0);
summary.vector_error_max_data_unit = max(vectorError);
summary.orbiter_to_calculated_norm_rmse_data_unit = ...
    sqrt(mean((calculatedNorm - orbiterNorm).^2));
summary.calculated_to_stored_norm_rmse_data_unit = ...
    sqrt(mean((calculatedNorm - storedNorm).^2));
summary.max_orthogonality_error_fro = max(orthogonalityError);
summary.max_abs_determinant_minus_one = max(determinantError);

if ~isfolder(outputDir)
    mkdir(outputDir);
end
summaryPath = fullfile(outputDir, 'tw1_momag_attitude_validation_summary.json');
fid = fopen(summaryPath, 'w');
assert(fid >= 0, 'Could not create summary file: %s', summaryPath);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(summary, PrettyPrint=true));
clear cleanup;

validIndex = find(valid);
sampleIndex = validIndex(round(linspace(1, numel(validIndex), min(12, numel(validIndex)))));
sampleTable = table( ...
    sampleIndex, rollDeg(sampleIndex), pitchDeg(sampleIndex), yawDeg(sampleIndex), ...
    bOrbiter(sampleIndex, 1), bOrbiter(sampleIndex, 2), bOrbiter(sampleIndex, 3), ...
    bMsoCalculated(sampleIndex, 1), bMsoCalculated(sampleIndex, 2), ...
    bMsoCalculated(sampleIndex, 3), bMsoStored(sampleIndex, 1), ...
    bMsoStored(sampleIndex, 2), bMsoStored(sampleIndex, 3), ...
    vecnorm(bMsoCalculated(sampleIndex, :) - bMsoStored(sampleIndex, :), 2, 2), ...
    'VariableNames', {'record_index', 'roll_deg', 'pitch_deg', 'yaw_deg', ...
    'bx_orbiter', 'by_orbiter', 'bz_orbiter', 'bx_mso_calculated', ...
    'by_mso_calculated', 'bz_mso_calculated', 'bx_mso_stored', ...
    'by_mso_stored', 'bz_mso_stored', 'vector_error'});
writetable(sampleTable, fullfile(outputDir, 'tw1_momag_attitude_validation_samples.csv'));

sourceAfter = dir(dataPath);
assert(sourceBefore.bytes == sourceAfter.bytes && sourceBefore.datenum == sourceAfter.datenum, ...
    'The source file metadata changed during validation.');

fprintf('Validated %d/%d records. Vector RMSE = %.12g data units.\n', ...
    summary.valid_record_count, summary.record_count, summary.vector_rmse_data_unit);
fprintf('Summary: %s\n', summaryPath);
end


function c = tw1OrbiterToMsoMatrix(rollDeg, pitchDeg, yawDeg)
roll = deg2rad(rollDeg);
pitch = deg2rad(pitchDeg);
yaw = deg2rad(yawDeg);
cr = cos(roll); sr = sin(roll);
cp = cos(pitch); sp = sin(pitch);
cy = cos(yaw); sy = sin(yaw);

n = numel(roll);
c = zeros(3, 3, n);
c(1, 1, :) = cy .* cp;
c(1, 2, :) = cy .* sp .* sr + sy .* cr;
c(1, 3, :) = sy .* sr - cy .* sp .* cr;
c(2, 1, :) = -sy .* cp;
c(2, 2, :) = cy .* cr - sy .* sp .* sr;
c(2, 3, :) = sy .* sp .* cr + cy .* sr;
c(3, 1, :) = sp;
c(3, 2, :) = -cp .* sr;
c(3, 3, :) = cp .* cr;
end


function value = samplePercentile(values, percentile)
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
