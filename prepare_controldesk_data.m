%% FILE 1: CONTROLDESK DATA PREPARATION
csvFile = '';
periods = [];
Ts = 50e-6;
smoothCurrents = true;
sgOrder = 3;
sgFrame = 7;
polePairs = 5;

%% Read the ControlDesk export
if strlength(csvFile) == 0
    [name, folder] = uigetfile('*.csv', 'Select the ControlDesk CSV');
    assert(~isequal(name, 0), 'No CSV selected.');
    csvFile = fullfile(folder, name);
end
assert(isfile(csvFile), 'Missing ControlDesk CSV: %s', csvFile);
lines = readlines(csvFile);
nameLine = find(startsWith(lines, "trace_names,"), 1);
pathLine = find(startsWith(lines, "path,"), 1);
valueLine = find(startsWith(lines, "trace_values,"), 1);
assert(~isempty(nameLine) && ~isempty(pathLine) && ~isempty(valueLine), ...
    'Expected ControlDesk trace_names, path, and trace_values records.');

names = strip(split(lines(nameLine), ","));
names = names(2:end);
names(1) = "Time_s";
paths = strip(split(lines(pathLine), ","));
paths = paths(2:end);
assert(numel(paths) == numel(names), 'Channel paths and names have different lengths.');

dataLines = lines(valueLine:end);
dataLines = dataLines(strlength(strip(dataLines)) > 0);
fields = split(dataLines, ",", 2);
A = str2double(fields(:,2:end));
assert(size(A,2) == numel(names) && all(isfinite(A), 'all'), ...
    'Unexpected channel count or nonfinite samples.');

tSource = A(:,1);
dt = diff(tSource);
assert(numel(dt) >= 14 && all(dt > 0), 'Expected at least 15 samples with increasing timestamps.');
assert(max(dt) < 1.5*Ts && abs(median(dt)-Ts) < 0.01*Ts, ...
    'Expected approximately %.3g ms between samples without gaps.', 1000*Ts);

dqCols = channel_columns(names, ["idm_pm", "iqm_pm", "Vd", "Vq", "we"]);
iabcCols = channel_columns(names, ["Ia_PM", "Ib_PM", "Ic_PM"]);
angleCol = find(names == "Out1" & contains(paths, "/Rem"));
assert(isscalar(angleCol), 'Expected one Out1 channel from Model Root/Rem.');
thetaSource = unwrap(A(:,angleCol)) - pi/2;

%% Resample and filter
tFull = (tSource(1):Ts:tSource(end)).';
rawFull = interp1(tSource, A(:,dqCols), tFull, 'linear');
abcFull = interp1(tSource, A(:,iabcCols), tFull, 'linear');
thetaFull = interp1(tSource, thetaSource, tFull, 'linear');
assert(all(isfinite([rawFull, abcFull, thetaFull]), 'all'), 'Resampling produced nonfinite values.');

filteredFull = rawFull;
validFull = true(size(tFull));
if smoothCurrents
    assert(exist('sgolayfilt', 'file') == 2, 'Current smoothing requires Signal Processing Toolbox.');
    assert(mod(sgFrame,2) == 1 && sgFrame > sgOrder && sgFrame <= numel(tFull), ...
        'Choose an odd SG frame larger than the polynomial order and no longer than the record.');
    filteredFull(:,1:2) = sgolayfilt(rawFull(:,1:2), sgOrder, sgFrame);
    edge = (sgFrame-1)/2;
    validFull([1:edge, end-edge+1:end]) = false;
end

%% Select samples and save
sourcePeriods = periods;
if isempty(sourcePeriods)
    sourcePeriods = [tFull(1), tFull(end)];
end
assert(size(sourcePeriods,2) == 2 && all(isfinite(sourcePeriods), 'all') && ...
    all(sourcePeriods(:,2) > sourcePeriods(:,1)) && ...
    all(sourcePeriods(2:end,1) > sourcePeriods(1:end-1,2)), ...
    'Enter ordered, separated [start end] windows in seconds.');
tol = 1e-9;
assert(sourcePeriods(1,1) >= tFull(1)-tol && sourcePeriods(end,2) <= tFull(end)+tol, ...
    'A window is outside the resampled record.');

rowsByPeriod = cell(size(sourcePeriods,1),1);
for k = 1:numel(rowsByPeriod)
    rows = find(tFull >= sourcePeriods(k,1)-tol & tFull <= sourcePeriods(k,2)+tol);
    assert(numel(rows) >= 15, 'Period %d has fewer than 15 samples.', k);
    rowsByPeriod{k} = rows;
end
selected = vertcat(rowsByPeriod{:});
counts = cellfun(@numel, rowsByPeriod);
lastRows = cumsum(counts);
firstRows = [1; lastRows(1:end-1)+1];

[~, stem] = fileparts(csvFile);
resultFolder = fullfile(pwd, 'results', stem);
if ~isfolder(resultFolder)
    [ok, message] = mkdir(resultFolder);
    assert(ok, 'Cannot create output folder: %s', message);
end

D = struct;
D.datasetId = stem;
D.sourceFile = csvFile;
D.Ts = Ts;
D.baseTs = median(dt);
D.downsampleFactor = 1;
D.sourcePeriods = sourcePeriods;
D.columns = {'id', 'iq', 'vd', 'vq', 'we'};
D.signalNames = D.columns;
D.tOriginal = tFull(selected);
D.tFile = D.tOriginal;
D.t = (0:numel(selected)-1).'*Ts;
D.tCombined = D.t;
D.segment = repelem((1:numel(counts)).', counts);
D.tLocal = D.t - D.t(firstRows(D.segment));
D.periods = [D.t(firstRows), D.t(lastRows)];
D.joinRows = firstRows(2:end);
D.raw = rawFull(selected,:);
D.filtered = filteredFull(selected,:);
D.validForIdentification = validFull(selected);
D.thetaElectrical = thetaFull(selected);
D.iabcRaw = abcFull(selected,:);
D.iabcFiltered = D.iabcRaw;
D.phaseCurrentName = 'Ia_PM, Ib_PM, Ic_PM';
D.angleName = 'Out1 from Model Root/Rem, shifted by -pi/2';
D.angleKind = 'electrical_rad';
D.angleOffset = 0;
D.speedKind = 'electrical_rad_s';
D.polePairs = polePairs;
D.filterBeforeSelection = true;
D.smoothCurrents = smoothCurrents;
D.sgOrder = sgOrder;
D.sgFrame = sgFrame;
D.units = 'time s, current A, voltage in logged scale, electrical speed rad/s';
dataFile = fullfile(resultFolder, 'combined_data.mat');
save(dataFile, 'D', '-v7.3');

C = table(D.segment, D.tOriginal, D.tLocal, D.tCombined, D.validForIdentification, ...
    'VariableNames', {'Period', 'OriginalTime', 'LocalTime', 'CombinedDisplayTime', 'ValidForIdentification'});
for k = 1:numel(D.columns)
    C.([D.columns{k} '_raw']) = D.raw(:,k);
    C.([D.columns{k} '_sg']) = D.filtered(:,k);
end
C.theta_e = D.thetaElectrical;
C.ia = D.iabcRaw(:,1);
C.ib = D.iabcRaw(:,2);
C.ic = D.iabcRaw(:,3);
writetable(C, fullfile(resultFolder, 'combined_data.csv'));

%% Plot the signals
figure('Name', 'ControlDesk signals for SINDy');
labels = {'id (A)', 'iq (A)', 'vd (logged scale)', 'vq (logged scale)', 'we (rad/s)'};
for col = 1:numel(labels)
    subplot(5,1,col);
    hold on;
    plot(D.t, D.raw(:,col), 'Color', [0.65 0.65 0.65]);
    if smoothCurrents && col <= 2
        plot(D.t, D.filtered(:,col), 'b', 'LineWidth', 1.2);
    end
    grid on;
    ylabel(labels{col});
end
xlabel('Combined time (s)');
fprintf('Saved %d samples at %.3f ms to %s\n', numel(D.t), 1000*D.Ts, dataFile);

function idx = channel_columns(names, wanted)
idx = zeros(1, numel(wanted));
for j = 1:numel(wanted)
    col = find(names == wanted(j));
    assert(isscalar(col), 'Expected exactly one channel named %s.', wanted(j));
    idx(j) = col;
end
end
