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

iterMax = 100000;
iterRecord = 20;
iterStep = floor(iterMax / iterRecord);

nHidden = {10, 50, 100, 200, 300, [10, 10], [10, 10, 10]};
nTrials = length(nHidden);
error = zeros(iterRecord, nTrials);

% change the network sturcture
for indexTrial = 1:nTrials
    weights = InitializeWeights(dim, nHidden{indexTrial}, nLabel);

    step = 1e-3;
    NNClassificationLoss = @(weights, index) ...
        ClassificationLoss(weights, Xtrain(index,:), ...
        yExpanded(index,:), nHidden{indexTrial}, nLabel);

    for iter = 1:iterMax
        if mod(iter - 1, iterStep) == 0
            indexRecord = (iter - 1) / iterStep + 1;
            yPred = ClassificationPredict(weights, Xvalid, nHidden{indexTrial}, nLabel);
            error(indexRecord,indexTrial) = sum(yPred ~= yvalid) / nvalid;
            fprintf('Training iteration = %d\tValidation error = %f\n', ...
                iter - 1, error(indexRecord,indexTrial));
        end
        indexTrain = ceil(rand * ntrain);
        [~, grad] = NNClassificationLoss(weights, indexTrain);
        weights = weights - step * grad;
    end    

    yPred = ClassificationPredict(weights, Xtest, nHidden{indexTrial}, nLabel);
    fprintf('Test error with final model = %f\n', sum(yPred ~= ytest) / ntest);
end

figure;
plot((0: iterRecord - 1) * iterStep, error, '-+');
legend('10', '50', '100', '200', '300', '[10, 10]', '[10, 10, 10]')
xlabel('Iteration')
ylabel('Error rate')
title('Validation Set Error Rate with Different Network Sturcture')