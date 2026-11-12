function [SindyModel, SindyReplay] = identify_sindy(dataFile, opts)
%IDENTIFY_SINDY File 2. Fit and replay a SINDy model for d-q currents.

arguments
    dataFile (1,1) string = "combined_data.mat"
    opts.OutputFolder (1,1) string = "results"
    opts.IncludeId (1,1) logical = true
    opts.LibraryType (1,1) string {mustBeMember(opts.LibraryType, ["physics", "broad"])} = "broad"
    opts.NormalizeColumns (1,1) logical = true
    opts.Lambda (1,1) double {mustBeFinite, mustBeNonnegative} = 0.2
    opts.LambdaRel (1,1) double {mustBeFinite, mustBeNonnegative} = 0.2
    opts.Ridge (1,1) double {mustBeFinite, mustBeNonnegative} = 1e-6
    opts.TrainingPeriods double = []
    opts.FitTimeWindows double = []
    opts.FitTimeBasis (1,1) string {mustBeMember(opts.FitTimeBasis, ["combined", "file"])} = "combined"
    opts.ReplayEnabled (1,1) logical = true
    opts.ReplayWindowSec (1,1) double {mustBeFinite, mustBePositive} = 0.02
    opts.ReplaySubsteps (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.MaxCurrent (1,1) double {mustBeFinite, mustBePositive} = 1e3
    opts.SaveTrainingCopy (1,1) logical = false
    opts.WriteReplayCSV (1,1) logical = false
    opts.Plot (1,1) logical = true
    opts.PlotStep (1,1) double {mustBeInteger, mustBePositive} = 1
end

%% Load and validate the dataset
assert(isfile(dataFile), 'Missing dataset: %s', dataFile);
S = load(dataFile, 'D');
assert(isfield(S, 'D') && isstruct(S.D) && isscalar(S.D), ...
    'The MAT file must contain a scalar struct D.');
D = S.D;
assert(all(isfield(D, {'filtered', 't', 'segment', 'Ts'})), ...
    'D must contain filtered, t, segment, and Ts.');
validateattributes(D.filtered, {'numeric'}, {'2d', 'real', 'nonempty'});
assert(size(D.filtered,2) >= 5, 'Expected D.filtered columns [id iq Vd Vq we].');
X = double(D.filtered(:,1:5));
assert(all(isfinite(X), 'all'), 'Nonfinite values in the filtered input columns.');
N = size(X,1);
validateattributes(D.t, {'numeric'}, {'vector', 'real', 'finite', 'numel', N});
validateattributes(D.segment, {'numeric'}, {'vector', 'real', 'finite', 'numel', N});
validateattributes(D.Ts, {'numeric'}, {'scalar', 'real', 'positive', 'finite'});
t = D.t(:);
segment = D.segment(:);
Ts = double(D.Ts);
segments = unique(segment, 'stable');
nPeriods = numel(segments);

trainingPeriods = opts.TrainingPeriods;
if isempty(trainingPeriods)
    trainingPeriods = segments.';
end
assert(all(ismember(trainingPeriods(:), segments)), 'Invalid training segment selected.');
fprintf('\nDataset: %s | samples %d | Ts %.6g s | segments %d\n', dataFile, N, Ts, nPeriods);

%% Build the candidate library
[Theta, terms, nonlinearTerms, cfg] = sindy_library(X);
nTerms = numel(terms);
fprintf('\nCandidate library (%d terms):\n', nTerms);
disp(terms(:));

%% Differentiate each contiguous segment and select training rows
replayRows = true(N,1);
filterEdge = 0;
if isfield(D, 'validForIdentification')
    validateattributes(D.validForIdentification, {'numeric', 'logical'}, ...
        {'vector', 'real', 'finite', 'numel', N});
    replayRows = logical(D.validForIdentification(:));
elseif isfield(D, 'sgFrame') && ~(isfield(D, 'filterBeforeSelection') && D.filterBeforeSelection)
    validateattributes(D.sgFrame, {'numeric'}, {'scalar', 'nonnegative', 'integer'});
    filterEdge = ceil((double(D.sgFrame)-1)/2);
end
fitEdge = filterEdge + 2;
fitRows = false(N,1);
dX = nan(N,2);
hasRaw = opts.Plot && isfield(D, 'raw') && ~isempty(D.raw);
dXraw = nan(N,2*hasRaw);
if hasRaw
    validateattributes(D.raw, {'numeric'}, {'2d', 'real', 'nrows', N});
    assert(size(D.raw,2) >= 2, 'Expected D.raw columns id and iq.');
end

newSegment = [true; segment(2:end) ~= segment(1:end-1)];
if isfield(D, 'joinRows') && ~isempty(D.joinRows)
    validateattributes(D.joinRows, {'numeric'}, {'vector', 'integer', '>=', 2, '<=', N});
    newSegment(D.joinRows(:)) = true;
end
segmentStarts = [find(newSegment); N+1];
segmentRun = cumsum(newSegment);
for b = 1:numel(segmentStarts)-1
    ii = (segmentStarts(b):segmentStarts(b+1)-1).';
    if numel(ii) >= 2
        for k = 1:2
            dX(ii,k) = gradient(X(ii,k), Ts);
            if hasRaw
                dXraw(ii,k) = gradient(double(D.raw(ii,k)), Ts);
            end
        end
    end
    position = (1:numel(ii)).';
    fitRows(ii) = ismember(segment(ii(1)), trainingPeriods) & ...
        position > fitEdge & position <= numel(ii)-fitEdge;
    if filterEdge > 0
        replayRows(ii) = position > filterEdge & position <= numel(ii)-filterEdge;
    end
end
fitRows = fitRows & replayRows;

if ~isempty(opts.FitTimeWindows)
    validateattributes(opts.FitTimeWindows, {'double'}, {'2d', 'ncols', 2, 'real', 'finite'});
    assert(all(opts.FitTimeWindows(:,1) <= opts.FitTimeWindows(:,2)), ...
        'FitTimeWindows must have [start end] in each row.');
    fitClock = t;
    if opts.FitTimeBasis == "file"
        assert(isfield(D, 'tFile'), 'File-relative training windows require D.tFile.');
        validateattributes(D.tFile, {'numeric'}, {'vector', 'real', 'finite', 'numel', N});
        fitClock = D.tFile(:);
    end
    inWindow = false(N,1);
    for w = 1:size(opts.FitTimeWindows,1)
        inWindow = inWindow | (fitClock >= opts.FitTimeWindows(w,1) & ...
            fitClock <= opts.FitTimeWindows(w,2));
    end
    fitRows = fitRows & inWindow;
end
assert(nnz(fitRows) >= max(20, 2*nTerms), 'Too few training samples for the candidate library.');
fprintf('Training samples: %d\n', nnz(fitRows));

%% Identify the d-q equations
states = 2;
stateTag = 'q';
if opts.IncludeId
    states = [1 2];
    stateTag = 'dq';
end
stateNames = {'id', 'iq'};
physicsTerms = {{'Vd', 'id', 'iq*we'}, {'Vq', 'iq', 'we', 'id*we'}};
Xi = zeros(nTerms,2);
fitInfo = cell(1,2);
for k = states
    selected = 1:nTerms;
    if opts.LibraryType == "physics"
        selected = find(ismember(terms, [physicsTerms{k}, nonlinearTerms]));
    end
    [Xi(selected,k), info] = sparsify_dynamics(Theta(fitRows,selected), dX(fitRows,k), opts);
    info.selectedTerms = terms(selected);
    fitInfo{k} = info;
    if ~info.identifiable
        warning('Equation %s: the candidate matrix is rank deficient or has zero columns.', stateNames{k});
    end
    if info.condition > 1e9
        warning('Equation %s: poorly conditioned library (condition %.3g).', stateNames{k}, info.condition);
    end
    fprintf('\nd%s/dt = ', stateNames{k});
    active = find(Xi(:,k) ~= 0);
    if isempty(active)
        fprintf('0');
    end
    for j = active.'
        fprintf('%+.8g*(%s) ', Xi(j,k), terms{j});
    end
    fprintf('\n');
end
coefficientTable = table(terms(:), Xi(:,1), Xi(:,2), ...
    'VariableNames', {'Term', 'did_dt', 'diq_dt'});
disp(coefficientTable);

%% Save the identified model
[~, dataStem] = fileparts(dataFile);
modelStem = sprintf('SINDy_%s_%s_%s', dataStem, opts.LibraryType, stateTag);
resultFolder = opts.OutputFolder;
if ~isfolder(resultFolder)
    [ok, message] = mkdir(resultFolder);
    assert(ok, 'Cannot create output folder: %s', message);
end

SindyModel = struct;
SindyModel.libraryVersion = 'dq_state_nonlinear_v3';
SindyModel.terms = terms;
SindyModel.Xi = Xi;
SindyModel.includeId = opts.IncludeId;
SindyModel.libraryType = char(opts.LibraryType);
SindyModel.includeIdIq = cfg.includeIdIq;
SindyModel.includeId2 = cfg.includeId2;
SindyModel.includeIq2 = cfg.includeIq2;
SindyModel.includeIq3 = cfg.includeIq3;
SindyModel.includeWeIq2 = cfg.includeWeIq2;
SindyModel.lambda = opts.Lambda;
SindyModel.lambdaRel = opts.LambdaRel;
SindyModel.ridge = opts.Ridge;
SindyModel.normalizedForSelection = opts.NormalizeColumns;
SindyModel.Ts = Ts;
SindyModel.trainingPeriods = trainingPeriods;
SindyModel.fitTimeWindows = opts.FitTimeWindows;
SindyModel.fitTimeBasis = char(opts.FitTimeBasis);
SindyModel.fitRows = find(fitRows);
SindyModel.fitInfo = fitInfo;
SindyModel.dataFile = char(dataFile);
SindyModel.inputColumns = {'id', 'iq', 'vd', 'vq', 'we'};
SindyModel.coefficients = coefficientTable;
SindyModel.cfg = cfg;
modelPath = fullfile(resultFolder, [modelStem '_model.mat']);
save(modelPath, 'SindyModel', '-v7.3');
writetable(coefficientTable, fullfile(resultFolder, [modelStem '_coefficients.csv']));
if opts.SaveTrainingCopy
    save(fullfile(resultFolder, [modelStem '_training_data.mat']), 'D', '-v7.3');
end

%% Replay and score the currents
currents = nan(N,2);
if ~opts.IncludeId
    currents(:,1) = X(:,1);
end
failed = zeros(nPeriods,1);
nWindows = zeros(nPeriods,1);
if opts.ReplayEnabled
    [currents, failed, nWindows] = replay_currents(X, segment, segmentRun, segments, Ts, ...
        replayRows, currents, Xi, states, opts);
end
completed = nWindows > 0 & failed == 0;
replayMSE = nan(nPeriods,2);
relRMSE = nan(nPeriods,2);
for j = 1:nPeriods
    ii = segment == segments(j) & replayRows;
    for k = states
        measured = X(ii,k);
        err = currents(ii,k) - measured;
        finiteRows = isfinite(err);
        if any(finiteRows)
            replayMSE(j,k) = mean(err(finiteRows).^2);
            relRMSE(j,k) = sqrt(replayMSE(j,k)) / max(mean(abs(measured(finiteRows))), eps);
        end
    end
end
MSEtable = table(segments, ismember(segments, trainingPeriods), completed, ...
    nWindows, failed, replayMSE(:,1), replayMSE(:,2), relRMSE(:,1), relRMSE(:,2), ...
    'VariableNames', {'Segment', 'Training', 'AllWindowsOK', 'Windows', 'FailedWindows', ...
    'id_MSE_A2', 'iq_MSE_A2', 'id_relRMSE', 'iq_relRMSE'});
if opts.ReplayEnabled
    fprintf('\nWindowed replay\n');
    disp(MSEtable);
end

SindyReplay = struct;
SindyReplay.t = t;
SindyReplay.segment = segment;
SindyReplay.filteredMeasured = X(:,1:2);
SindyReplay.modelCurrents = currents;
SindyReplay.completed = completed;
SindyReplay.MSE = MSEtable;
SindyReplay.fitRows = fitRows;
SindyReplay.idIdentified = opts.IncludeId;
SindyReplay.replayWindowSec = opts.ReplayWindowSec;
SindyReplay.cfg = cfg;
save(fullfile(resultFolder, [modelStem '_replay.mat']), 'SindyReplay', '-v7.3');
writetable(MSEtable, fullfile(resultFolder, [modelStem '_replay_MSE.csv']));
if opts.WriteReplayCSV
    rows = table(segment, t, X(:,1), X(:,2), currents(:,1), currents(:,2), ...
        'VariableNames', {'Segment', 'Time', 'id_measured', 'iq_measured', 'id_model', 'iq_model'});
    writetable(rows, fullfile(resultFolder, [modelStem '_replay.csv']));
end

if opts.Plot
    plot_results(t, segmentRun, X, D, dXraw, dX, Theta*Xi, ...
        currents, fitRows, states, stateNames, opts, fullfile(resultFolder, [modelStem '_replay.png']));
end
fprintf('\nModel saved: %s\n', modelPath);
end

function [coeff, info] = sparsify_dynamics(A, dx, opts)
present = any(A ~= 0,1);
coeff = zeros(size(A,2),1);
info = struct('omittedColumns', find(~present), 'identifiable', false, ...
    'condition', Inf, 'threshold', NaN);
if ~any(present)
    return
end
A = A(:,present);
scale = ones(1,size(A,2));
threshold = opts.Lambda;
ridge = 0;
if opts.NormalizeColumns
    scale = sqrt(mean(A.^2,1));
    scale(scale < eps) = 1;
    threshold = opts.LambdaRel * max(sqrt(mean(dx.^2)), eps);
    ridge = opts.Ridge;
end
A = A ./ scale;
info.threshold = threshold;
info.identifiable = all(present) && rank(A) == size(A,2);
info.condition = cond(A);
if ridge > 0
    solve = @(M,y) (M'*M + ridge*size(M,1)*eye(size(M,2))) \ (M'*y);
else
    solve = @(M,y) lsqminnorm(M, y);
end
c = solve(A, dx);
active = true(size(c));
for iteration = 1:10
    keep = active & abs(c) >= threshold;
    if isequal(keep, active)
        break
    end
    active = keep;
    c(~active) = 0;
    if ~any(active)
        break
    end
    c(active) = solve(A(:,active), dx);
end
coeff(present) = c(:) ./ scale(:);
end

function [currents, failed, nWindows] = replay_currents(X, segment, segmentRun, segments, Ts, ...
        valid, currents, Xi, states, opts)
failed = zeros(numel(segments),1);
nWindows = zeros(numel(segments),1);
for j = 1:numel(segments)
    ii = find(segment == segments(j) & valid);
    if isempty(ii)
        continue
    end
    bounds = [0; find(diff(ii) > 1 | diff(segmentRun(ii)) > 0); numel(ii)];
    for r = 1:numel(bounds)-1
        run = ii(bounds(r)+1:bounds(r+1));
        n = numel(run);
        if n < 2
            continue
        end
        windowLength = min(n, max(2, round(opts.ReplayWindowSec / Ts)));
        nRegular = floor(n / windowLength);
        lastRegular = nRegular * windowLength;
        regularWindows = reshape(run(1:lastRegular), windowLength, nRegular);
        if lastRegular < n
            windows = {regularWindows, run(lastRegular+1:end)};
        else
            windows = {regularWindows};
        end
        for g = 1:numel(windows)
            idx = windows{g};
            [out, alive] = integrate_windows(X, idx, Ts, Xi, states, opts);
            currents(idx(:),states) = reshape(out, [], numel(states));
            nWindows(j) = nWindows(j) + numel(alive);
            failed(j) = failed(j) + nnz(~alive);
        end
    end
end
end

function [out, alive] = integrate_windows(X, idx, Ts, Xi, states, opts)
% Advance all windows of the same length together with fixed-step RK4.
[nSamples, nWindows] = size(idx);
nStates = numel(states);
x = X(idx(1,:),states);
alive = all(abs(x) <= opts.MaxCurrent,2);
x(~alive,:) = NaN;
out = nan(nSamples,nWindows,nStates);
out(1,:,:) = reshape(x, 1, nWindows, nStates);
h = Ts / opts.ReplaySubsteps;
rhs = @(x,u) current_derivative(x, u, Xi, states);

for m = 1:nSamples-1
    if ~any(alive)
        break
    end
    u0 = X(idx(m,:),:);
    delta = X(idx(m+1,:),:) - u0;
    for q = 1:opts.ReplaySubsteps
        a = (q-1) / opts.ReplaySubsteps;
        b = q / opts.ReplaySubsteps;
        ua = u0 + delta*a;
        um = u0 + delta*((a+b)/2);
        ub = u0 + delta*b;
        k1 = rhs(x, ua);
        k2 = rhs(x + h/2*k1, um);
        k3 = rhs(x + h/2*k2, um);
        k4 = rhs(x + h*k3, ub);
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
        alive = alive & all(isfinite(x) & abs(x) <= opts.MaxCurrent,2);
        x(~alive,:) = NaN;
    end
    out(m+1,:,:) = reshape(x, 1, nWindows, nStates);
end
end

function dx = current_derivative(x, u, Xi, states)
% Replace the measured currents only for states identified by the model.
u(:,states) = x;
dx = sindy_library(u) * Xi(:,states);
end

function plot_results(t, segmentRun, X, D, dXraw, dX, dXmodel, currents, ...
        fitRows, states, names, opts, replayPath)
pp = 1:opts.PlotStep:numel(t);
figure('Name', 'SINDy derivative identification');
for k = 1:2
    subplot(2,2,k);
    hold on;
    if ~isempty(dXraw)
        plot_segments(t(pp), D.raw(pp,k), segmentRun(pp), 'Color', [0.7 0.7 0.7], 'DisplayName', 'Raw');
    end
    plot_segments(t(pp), X(pp,k), segmentRun(pp), 'b', 'LineWidth', 1.2, 'DisplayName', 'Filtered');
    plot(t(fitRows), X(fitRows,k), '.', 'Color', [0.9 0.45 0], 'DisplayName', 'Training samples');
    grid on;
    xlabel('Combined time (s)');
    ylabel([names{k} ' (A)']);
    legend('show');

    subplot(2,2,k+2);
    hold on;
    if ~isempty(dXraw)
        plot_segments(t(pp), dXraw(pp,k), segmentRun(pp), 'Color', [0.7 0.7 0.7], 'DisplayName', 'Raw');
    end
    plot_segments(t(pp), dX(pp,k), segmentRun(pp), 'b', 'DisplayName', 'Filtered');
    if ismember(k, states)
        plot_segments(t(pp), dXmodel(pp,k), segmentRun(pp), '--r', 'LineWidth', 1.2, 'DisplayName', 'SINDy');
    end
    grid on;
    xlabel('Combined time (s)');
    ylabel(['d' names{k} '/dt (A/s)']);
    legend('show');
end

if ~opts.ReplayEnabled
    return
end
figReplay = figure('Name', 'SINDy replay');
for k = 1:2
    subplot(2,1,k);
    hold on;
    plot_segments(t(pp), X(pp,k), segmentRun(pp), 'b', 'DisplayName', 'Measured');
    if ismember(k, states)
        plot_segments(t(pp), currents(pp,k), segmentRun(pp), '--r', 'DisplayName', 'SINDy');
    end
    grid on;
    xlabel('Combined time (s)');
    ylabel([names{k} ' (A)']);
    legend('show');
end
try
    exportgraphics(figReplay, replayPath, 'Resolution', 160);
catch
    saveas(figReplay, replayPath);
end
end

function plot_segments(t, y, segmentRun, varargin)
t = t(:);
y = y(:);
segmentRun = segmentRun(:);
gaps = [false; segmentRun(2:end) ~= segmentRun(1:end-1)];
rows = (1:numel(t)).' + cumsum(gaps);
tx = nan(numel(t)+nnz(gaps),1);
yx = tx;
tx(rows) = t;
yx(rows) = y;
plot(tx, yx, varargin{:});
end
