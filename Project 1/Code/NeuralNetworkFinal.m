clear;
clc;
close all;

load('data.mat');

rng(0);

[ntrain, dim] = size(Xtrain);
[nvalid, ~] = size(Xvalid);
[ntest, ~] = size(Xtest);

% pre-process X
[Xtrain, mu, sigma] = Standardize(Xtrain);
Xtrain = [ones(ntrain, 1), Xtrain];
Xvalid = Standardize(Xvalid, mu, sigma);
Xvalid = [ones(nvalid, 1), Xvalid];
Xtest = Standardize(Xtest, mu, sigma);
Xtest = [ones(ntest, 1), Xtest];
dim = dim + 1;

% pre-process y
nLabel = max(ytrain);
yExpanded = -zeros(ntrain, nLabel);
for indexTrain = 1:ntrain
    yExpanded(indexTrain, ytrain(indexTrain)) = 1;
end

nHidden = 5 * 5 + 1;
weightLinear = InitializeWeightsConv(nHidden, nLabel);

iterMax = 100000;
iterRecord = 20;
iterStep = floor(iterMax / iterRecord);

alpha = 1e-3;
lambda = 0.05;
error = zeros(iterRecord, 1);
NNClassificationLoss = ...
    @(weightLinear, index, nHidden)...
    ClassificationLossFinal(weightLinear, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel);

for iter = 1:iterMax
    if mod(iter - 1, iterStep) == 0
        indexRecord = (iter - 1) / iterStep + 1;
        yPred = ClassificationPredictFinal...
            (weightLinear, Xvalid, nHidden, nLabel);
        error(indexRecord) = sum(yPred ~= yvalid) / nvalid;
        fprintf('Training iteration = %d\tValidation error = %f\n', ...
            iter - 1, error(indexRecord));
    end
    indexTrain = ceil(rand * ntrain);
    [~, gradLinear] = NNClassificationLoss...
        (weightLinear, indexTrain, nHidden);
    weightLinear = weightLinear - ...
        alpha * (gradLinear + lambda * weightLinear);
end

figure;
plot((0:iterRecord - 1) * iterStep, error, '-+');
xlabel('Iteration');
ylabel('Error rate');
title('Validation Set Error Rate');

yPred = ClassificationPredictFinal(weightLinear, Xtest, nHidden, nLabel);
fprintf('Test error with final model = %f\n', ...
    sum(yPred ~= ytest) / ntest);