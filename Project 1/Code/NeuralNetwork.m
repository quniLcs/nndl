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
yExpanded = -ones(ntrain, nLabel);
for index = 1:ntrain
    yExpanded(index, ytrain(index)) = 1;
end

nHidden = 10;
weights = InitializeWeights(dim, nHidden, nLabel);

iterMax = 100000;
iterRecord = 20;
iterStep = floor(iterMax / iterRecord);

step = 1e-3;
error = zeros(iterRecord, 1);
NNClassificationLoss = @(weights, index) ClassificationLoss(weights, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel);

for iter = 1:iterMax
    if mod(iter - 1, iterStep) == 0
        index = (iter - 1) / iterStep + 1;
        yPred = ClassificationPredict(weights, Xvalid, nHidden, nLabel);
        error(index) = sum(yPred ~= yvalid) / nvalid;
        fprintf('Training iteration = %d\tValidation error = %f\n', ...
            iter - 1, error(index));
    end
    index = ceil(rand * ntrain);
    [~, grad] = NNClassificationLoss(weights, index);
    weights = weights - step * grad;
end

figure;
plot((0:iterRecord - 1) * iterStep, error, '-+');
xlabel('Iteration')
ylabel('Error rate')
title('Validation Set Error Rate')

yPred = ClassificationPredict(weights, Xtest, nHidden, nLabel);
fprintf('Test error with final model = %f\n', sum(yPred ~= ytest) / ntest);