function [Theta, terms, nonlinearTerms, cfg] = sindy_library(Y)
%SINDY_LIBRARY Evaluate the candidate features for rows of [id iq Vd Vq we].

arguments
    Y (:,5) double
end

id = Y(:,1);
iq = Y(:,2);
Vd = Y(:,3);
Vq = Y(:,4);
we = Y(:,5);
iq2 = iq.^2;

baseTerms = {'1', 'Vd', 'Vq', 'id', 'iq', 'we', 'id*we', 'iq*we'};
nonlinearTerms = {'id*iq', 'iq^2', 'iq^3', 'we*iq^2'};
Theta = [ones(size(id)), Vd, Vq, id, iq, we, id.*we, iq.*we, ...
    id.*iq, iq2, iq2.*iq, we.*iq2];
terms = [baseTerms, nonlinearTerms];

if nargout >= 4
    % Legacy MAT-file metadata.
    cfg = struct('includeIdIq', true, 'includeId2', false, 'includeIq2', true, ...
        'includeIq3', true, 'includeWeIq2', true, 'includeId3', false, ...
        'includeId2Iq', false, 'includeIdIq2', false, ...
        'includeVoltageLinear', false, 'includeVoltageQuadratic', false);
end
end
