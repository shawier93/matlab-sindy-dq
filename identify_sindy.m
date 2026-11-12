%% FILE 2: SINDy IDENTIFICATION - STATE NONLINEARITIES (ROBUST REPLAY)
% Run after prepare_controldesk_data.m.

%% =========================================================
% 1. SETTINGS
% ==========================================================
includeId   = true;      % true = identify id and iq; false = identify iq (measured id)
libraryType = 'broad';  % 'physics' or 'broad'

%% SPARSE REGRESSION
normalizeColumns = true;  % scale columns before selection (recommended: columns are collinear)
lambda    = 0.2;         % threshold on RAW coefficients      (used if normalizeColumns = false)
lambdaRel = 2;         % threshold relative to rms(dx/dt)   (used if normalizeColumns = true)
ridge     = 1e-6;         % small ridge for collinear columns (only with normalizeColumns = true)

%% DATA SELECTION
trainingPeriods = [];     % [] = all saved segments
fitTimeWindows  = [];     % [] = all valid times, or Nx2 [t0 t1; ...]
fitTimeBasis    = 'combined';   % 'combined' (= D.t) or 'file' (time inside each file)

%% REPLAY
replayEnabled   = true;
replayWindowSec = 0.02;        % window length for 'windowed' (about one electrical period)
replaySubsteps  = 1;           % RK4 substeps per data sample (raise if model is stiff)
maxCurrent      = 1e3;         % |current| above this marks the window as failed (A)

%% OUTPUT
saveTrainingCopy = false;      % copy of D next to the model (large)
writeReplayCSV   = false;      % one row per sample (large)
plotStep         = 1;          % plot every Nth sample (raise for huge datasets)

dataFileName = 'combined_data.mat';
assert(ismember(libraryType, {'physics','broad'}), 'Choose physics or broad.');
stateTag = 'q';
if includeId, stateTag = 'dq'; end
modelStem = ['SINDy_combined_data_' libraryType '_' stateTag];
modelFileName = [modelStem '.mat'];

%% =========================================================
% 2. CHECK WORKSPACE DATA
% ==========================================================
assert(exist('D','var') == 1 && isstruct(D) && isscalar(D), ...
    'Workspace variable D is required. Run prepare_controldesk_data.m first.');
if ~exist('resultFolder','var') || isempty(resultFolder)
    resultFolder = fullfile(pwd,'results');
end
if ~isfolder(resultFolder)
    [ok,message] = mkdir(resultFolder);
    assert(ok,'Cannot create output folder: %s',message);
end
if ~exist('dataFile','var') || isempty(dataFile)
    dataFile = fullfile(resultFolder,dataFileName);
end

assert(isfield(D,'filtered') && ~isempty(D.filtered),'D.filtered is empty. Check File 1.');
assert(size(D.filtered,2)>=5,'Expected [id iq Vd Vq we].');
assert(all(isfinite(D.filtered(:,1:5)),'all'),'Nonfinite values in filtered data.');
N = size(D.filtered,1);
assert(numel(D.t)==N && numel(D.segment)==N,'Data lengths do not match.');
D.t = D.t(:);
D.segment = D.segment(:);
validateattributes(D.Ts,{'numeric'},{'scalar','positive','finite'});

segments = unique(D.segment(:),'stable')';
nPeriods = numel(segments);
if isempty(trainingPeriods), trainingPeriods = segments; end
assert(all(ismember(trainingPeriods,segments)),'Invalid training period selected.');

fprintf('\n======================================\n');
fprintf('SINDy STATE NONLINEAR IDENTIFICATION\n');
fprintf('======================================\n');
fprintf('Dataset: %s | samples %d | Ts %.6g s | segments %d\n',dataFileName,N,D.Ts,nPeriods);
if isfield(D,'segInfo')
    disp(table((1:nPeriods)',D.segInfo.freqHz(:),D.segInfo.torquePct(:),D.segInfo.joinFromPrevious(:), ...
        'VariableNames',{'Segment','f_elec_Hz','Torque_pct','JoinFromPrevious'}));
end

%% =========================================================
% 3. LIBRARY
% ==========================================================
X = D.filtered(:,1:5);
[Theta,terms,nonlinearNames,cfg] = sindy_library(X);
nTerms = numel(terms);
fprintf('\nCandidate library (%d terms):\n',nTerms);
for j = 1:nTerms, fprintf('%2d : %s\n',j,terms{j}); end

%% =========================================================
% 4. DIFFERENTIATE EACH SEGMENT SEPARATELY
% ==========================================================
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
hasRaw = isfield(D, 'raw') && ~isempty(D.raw);
dXraw = nan(N,2*hasRaw);
if hasRaw
    validateattributes(D.raw, {'numeric'}, {'2d', 'real', 'nrows', N});
    assert(size(D.raw,2) >= 2, 'Expected D.raw columns id and iq.');
end

newSegment = [true; D.segment(2:end) ~= D.segment(1:end-1)];
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
            dX(ii,k) = gradient(X(ii,k), D.Ts);
            if hasRaw
                dXraw(ii,k) = gradient(double(D.raw(ii,k)), D.Ts);
            end
        end
    end
    position = (1:numel(ii)).';
    fitRows(ii) = ismember(D.segment(ii(1)), trainingPeriods) & ...
        position > fitEdge & position <= numel(ii)-fitEdge;
    if filterEdge > 0
        replayRows(ii) = position > filterEdge & position <= numel(ii)-filterEdge;
    end
end
fitRows = fitRows & replayRows;

%% =========================================================
% 5. SELECT TRAINING SAMPLES
% ==========================================================
switch fitTimeBasis
    case 'combined', fitClock = D.t;
    case 'file',     fitClock = D.tFile(:);
    otherwise, error('fitTimeBasis must be combined or file.');
end
if ~isempty(fitTimeWindows)
    assert(size(fitTimeWindows,2)==2,'fitTimeWindows must be Nx2.');
    tm = false(N,1);
    for w = 1:size(fitTimeWindows,1)
        tm = tm | (fitClock>=fitTimeWindows(w,1) & fitClock<=fitTimeWindows(w,2));
    end
    fitRows = fitRows & tm;
end
assert(nnz(fitRows)>=max(20,2*nTerms),'Too few training samples.');
fprintf('\nTraining samples: %d\n',nnz(fitRows));

%% =========================================================
% 6. SPARSE IDENTIFICATION
% ==========================================================
Xi = zeros(nTerms,2);
fitInfo = cell(1,2);
if includeId, states = [1 2]; else, states = 2; end

for k = states
    switch libraryType
        case 'broad'
            selected = 1:nTerms;
        case 'physics'
            if k==1, candidateNames = {'Vd','id','iq*we'};
            else,    candidateNames = {'Vq','iq','we','id*we'}; end
            candidateNames = [candidateNames,nonlinearNames];
            selected = find(ismember(terms,candidateNames));
        otherwise
            error('Unknown libraryType.');
    end
    fprintf('\nIdentifying equation %d using:\n',k); disp(terms(selected)');

    [coeff,info] = sparsifyDynamics(Theta(fitRows,selected),dX(fitRows,k), ...
        lambda,lambdaRel,normalizeColumns,ridge);
    Xi(selected,k) = coeff;
    info.selectedTerms = terms(selected);
    fitInfo{k} = info;

    if ~info.identifiable
        warning('Equation %d: library is rank deficient (columns are collinear).',k);
    end
    if info.condition>1e9
        warning('Equation %d: poorly conditioned library (condition %.3g).',k,info.condition);
    end
end

%% =========================================================
% 7. DISPLAY EQUATIONS
% ==========================================================
stateNames = {'id','iq'};
fprintf('\n======================================\nIDENTIFIED SINDy EQUATIONS\n======================================\n');
for k = states
    fprintf('\nd%s/dt = ',stateNames{k});
    active = find(Xi(:,k)~=0);
    if isempty(active), fprintf('0');
    else, for j = active', fprintf('%+.8g*(%s) ',Xi(j,k),terms{j}); end, end
    fprintf('\n');
end
coefficientTable = table(terms(:),Xi(:,1),Xi(:,2),'VariableNames',{'Term','did_dt','diq_dt'});
disp(coefficientTable);

%% =========================================================
% 8. SAVE MODEL
% ==========================================================
SindyModel = struct;
SindyModel.libraryVersion = 'dq_state_nonlinear_v3';
SindyModel.terms = terms;  SindyModel.Xi = Xi;
SindyModel.includeId = includeId;  SindyModel.libraryType = libraryType;
SindyModel.includeIdIq = cfg.includeIdIq;  SindyModel.includeId2 = cfg.includeId2;
SindyModel.includeIq2 = cfg.includeIq2;    SindyModel.includeIq3 = cfg.includeIq3;
SindyModel.includeWeIq2 = cfg.includeWeIq2;
SindyModel.lambda = lambda;  SindyModel.lambdaRel = lambdaRel;  SindyModel.ridge = ridge;
SindyModel.normalizedForSelection = normalizeColumns;
SindyModel.Ts = D.Ts;  SindyModel.trainingPeriods = trainingPeriods;
SindyModel.fitTimeWindows = fitTimeWindows;  SindyModel.fitTimeBasis = fitTimeBasis;
SindyModel.fitRows = find(fitRows);  SindyModel.fitInfo = fitInfo;
SindyModel.dataFile = dataFile;
SindyModel.inputColumns = {'id','iq','vd','vq','we'};
SindyModel.coefficients = coefficientTable;  SindyModel.cfg = cfg;
modelPath = fullfile(resultFolder,modelFileName);
save(modelPath,'SindyModel','-v7.3');
writetable(coefficientTable,fullfile(resultFolder,[modelStem '_coefficients.csv']));
if saveTrainingCopy
    save(fullfile(resultFolder,[modelStem '_training_data.mat']),'D','-v7.3');
end

%% =========================================================
% 9. DERIVATIVE PREDICTION
% ==========================================================
dXmodel = Theta*Xi;
pp = 1:plotStep:N;
figure('Name','SINDy derivative identification');
for k = 1:2
    subplot(2,2,k);
    rawPlot = D.filtered(pp,k);
    if hasRaw, rawPlot = D.raw(pp,k); end
    [tx,z] = break_lines(D.t(pp),[rawPlot,D.filtered(pp,k)],segmentRun(pp));
    plot(tx,z(:,1),'Color',[0.7 0.7 0.7]); hold on;
    plot(tx,z(:,2),'b','LineWidth',1.2);
    plot(D.t(fitRows),D.filtered(fitRows,k),'.','Color',[0.9 0.45 0]);
    grid on; xlabel('Combined time (s)'); ylabel([stateNames{k} ' (A)']);
    legend('Raw','Filtered','Training samples');

    subplot(2,2,k+2);
    rawDerivative = dX(pp,k);
    if hasRaw, rawDerivative = dXraw(pp,k); end
    [tx,z] = break_lines(D.t(pp),[rawDerivative,dX(pp,k),dXmodel(pp,k)],segmentRun(pp));
    plot(tx,z(:,1),'Color',[0.8 0.8 0.8]); hold on; plot(tx,z(:,2),'b');
    if k==2 || includeId
        plot(tx,z(:,3),'--r','LineWidth',1.2); legend('Raw','Filtered','SINDy');
    else
        legend('Raw','Filtered');
    end
    grid on; xlabel('Combined time (s)'); ylabel(['d' stateNames{k} '/dt (A/s)']);
end

%% =========================================================
% 10. REPLAY (ROBUST)
% ==========================================================
SindyCurrents = nan(N,2);  if ~includeId, SindyCurrents(:,1) = D.filtered(:,1); end
replayCompleted = false(nPeriods,1);  failedWindows = zeros(nPeriods,1);  nWindows = zeros(nPeriods,1);
if replayEnabled
    [SindyCurrents,failedWindows,nWindows] = replay_currents( ...
        X,D.segment,segmentRun,segments,D.Ts,replayRows,SindyCurrents, ...
        Xi,states,replayWindowSec,replaySubsteps,maxCurrent);
    replayCompleted = nWindows > 0 & failedWindows == 0;
end

%% =========================================================
% 11. MSE
% ==========================================================
replayMSE = nan(nPeriods,2);  relRMSE = nan(nPeriods,2);
for j = 1:nPeriods
    ii = find(D.segment==segments(j));
    ii = ii(replayRows(ii));
    for k = 1:2
        if k==1 && ~includeId, continue; end
        e = SindyCurrents(ii,k) - D.filtered(ii,k);
        ok = isfinite(e);
        if any(ok)
            replayMSE(j,k) = mean(e(ok).^2);
            relRMSE(j,k)   = sqrt(replayMSE(j,k))/max(mean(abs(D.filtered(ii(ok),k))),eps);
        end
    end
end
MSEtable = table(segments(:),ismember(segments(:),trainingPeriods),replayCompleted, ...
    nWindows,failedWindows,replayMSE(:,1),replayMSE(:,2),relRMSE(:,1),relRMSE(:,2), ...
    'VariableNames',{'Segment','Training','AllWindowsOK','Windows','FailedWindows', ...
    'id_MSE_A2','iq_MSE_A2','id_relRMSE','iq_relRMSE'});
fprintf('\nWINDOWED REPLAY MSE\n'); disp(MSEtable);

%% =========================================================
% 12. PLOT REPLAY
% ==========================================================
figReplay = figure('Name','SINDy replay');
for k = 1:2
    subplot(2,1,k);
    [tx,z] = break_lines(D.t(pp),[D.filtered(pp,k),SindyCurrents(pp,k)],segmentRun(pp));
    plot(tx,z(:,1),'b','LineWidth',1.2); hold on;
    plot(tx,z(:,2),'--r','LineWidth',1.2); grid on;
    xlabel('Time (s)'); ylabel([stateNames{k} ' (A)']);
    if k==1 && ~includeId, legend('Measured id','Measured id (used as input)');
    else, legend('Actual','SINDy'); end
end

%% =========================================================
% 13. SAVE REPLAY
% ==========================================================
SindyReplay = struct;
SindyReplay.t = D.t;  SindyReplay.segment = D.segment;
SindyReplay.filteredMeasured = D.filtered(:,1:2);
SindyReplay.modelCurrents = SindyCurrents;
SindyReplay.completed = replayCompleted;  SindyReplay.MSE = MSEtable;
SindyReplay.fitRows = fitRows;  SindyReplay.idIdentified = includeId;
SindyReplay.replayWindowSec = replayWindowSec;
SindyReplay.cfg = cfg;
save(fullfile(resultFolder,[modelStem '_replay.mat']),'SindyReplay','-v7.3');
writetable(MSEtable,fullfile(resultFolder,[modelStem '_replay_MSE.csv']));
if writeReplayCSV
    RT = table(D.segment,D.t,D.filtered(:,1),D.filtered(:,2),SindyCurrents(:,1),SindyCurrents(:,2), ...
        'VariableNames',{'Segment','Time','id_measured','iq_measured','id_model','iq_model'});
    writetable(RT,fullfile(resultFolder,[modelStem '_replay.csv']));
end
try
    exportgraphics(figReplay,fullfile(resultFolder,[modelStem '_replay.png']),'Resolution',160);
catch
    saveas(figReplay,fullfile(resultFolder,[modelStem '_replay.png']));
end
fprintf('\nFILE 2 COMPLETED\nModel saved: %s\n',modelPath);

%% =========================================================
% LOCAL FUNCTIONS
% ==========================================================


function [coeff,info] = sparsifyDynamics(A,dx,lambda,lambdaRel,normalizeColumns,ridge)
present = any(A~=0,1);
coeff = zeros(size(A,2),1);
info.omittedColumns = find(~present);
info.identifiable = false;  info.condition = Inf;  info.threshold = NaN;
if ~any(present), return; end
A = A(:,present);
scale = ones(1,size(A,2));
if normalizeColumns
    scale = sqrt(mean(A.^2,1));  scale(scale<eps) = 1;
end
As = A./scale;
if normalizeColumns
    thr = lambdaRel*max(sqrt(mean(dx.^2)),eps);   % threshold in A/s of contribution
    rid = ridge;
else
    thr = lambda;  rid = 0;
end
info.threshold = thr;
info.identifiable = all(present) && rank(As)==size(As,2);
info.condition = cond(As);

solve = @(M,y) (M'*M + rid*size(M,1)*eye(size(M,2)))\(M'*y);
c = solve(As,dx);
for it = 1:10
    small = abs(c)<thr;
    c(small) = 0;
    big = ~small;
    if ~any(big), break; end
    c(big) = solve(As(:,big),dx);
end
coeff(present) = c(:)./scale(:);     % back to physical units
end

function [currents, failed, nWindows] = replay_currents(X, segment, segmentRun, segments, Ts, ...
        valid, currents, Xi, states, replayWindowSec, replaySubsteps, maxCurrent)
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
        windowLength = min(n, max(2, round(replayWindowSec / Ts)));
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
            [out, alive] = integrate_windows(X, idx, Ts, Xi, states, replaySubsteps, maxCurrent);
            currents(idx(:),states) = reshape(out, [], numel(states));
            nWindows(j) = nWindows(j) + numel(alive);
            failed(j) = failed(j) + nnz(~alive);
        end
    end
end
end

function [out, alive] = integrate_windows(X, idx, Ts, Xi, states, replaySubsteps, maxCurrent)
% Advance all windows of the same length together with fixed-step RK4.
[nSamples, nWindows] = size(idx);
nStates = numel(states);
x = X(idx(1,:),states);
alive = all(abs(x) <= maxCurrent,2);
x(~alive,:) = NaN;
out = nan(nSamples,nWindows,nStates);
out(1,:,:) = reshape(x, 1, nWindows, nStates);
h = Ts / replaySubsteps;
rhs = @(x,u) current_derivative(x, u, Xi, states);

for m = 1:nSamples-1
    if ~any(alive)
        break
    end
    u0 = X(idx(m,:),:);
    delta = X(idx(m+1,:),:) - u0;
    for q = 1:replaySubsteps
        a = (q-1) / replaySubsteps;
        b = q / replaySubsteps;
        ua = u0 + delta*a;
        um = u0 + delta*((a+b)/2);
        ub = u0 + delta*b;
        k1 = rhs(x, ua);
        k2 = rhs(x + h/2*k1, um);
        k3 = rhs(x + h/2*k2, um);
        k4 = rhs(x + h*k3, ub);
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
        alive = alive & all(isfinite(x) & abs(x) <= maxCurrent,2);
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

function [tx,z] = break_lines(t,y,segment)
tx = []; z = [];
for s = unique(segment(:))'
    ii = segment==s;
    tx = [tx;t(ii);NaN];  z = [z;y(ii,:);nan(1,size(y,2))]; %#ok<AGROW>
end
end
