function [x0, targets,dx,dx_targets] = createMiniBatch(numTimesteps,numTimesPerObs,miniBatchSize,X,dX)
%  take parts of every trajectory
nn = length(X);
x0 = [];  % need to be dlarray for MATLAB to track
dx = zeros([size(X{1},1),nn*miniBatchSize]);  % just array 
targets = zeros([size(X{1},1) nn*miniBatchSize numTimesPerObs]);
% the derivative we want to fit
% the first dimension depends on the dimension of original dde
% we consider a scalar case so far
dx_targets = zeros([1 nn*miniBatchSize]);
for k = 1:nn
    s = randperm(numTimesteps - numTimesPerObs, miniBatchSize);
    x0 = [x0 dlarray(X{k}(:, s))]; % initial states in the minibatch
    dx(:,(k-1)*miniBatchSize+1:k*miniBatchSize) = dX{k}(:, s);
    for i = 1:miniBatchSize
        targets(:, (k-1)*miniBatchSize+i, 1:numTimesPerObs) = X{k}(:, s(i) + 1:(s(i) + numTimesPerObs));
        dx_targets((k-1)*miniBatchSize+i) = dX{k}(1, s(i));
    end
end
end