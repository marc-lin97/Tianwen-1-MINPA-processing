% Standalone, restartable production entry point.
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
run_tw1_minpa_quality_flags_all_species({}, false, inf, false);
