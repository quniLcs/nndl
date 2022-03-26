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
for indexTrain = 1:ntrain
    yExpanded(indexTrain, ytrain(indexTrain)) = 1;
end

nHidden = 10;

iterMax = 100000;
iterRecord = 20;
iterStep = floor(iterMax / iterRecord);

alpha = 1e-3;

prob = [1, 0.5];
nTrial = length(prob);
error = zeros(iterRecord, nTrial);
NNClassificationLoss = ...
    @(weightLinear, index, prob)...
    ClassificationLossDropout(weightLinear, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel, prob);

for indexTrial = 1:nTrial
    weightLinear = InitializeWeightsBasic(dim, nHidden, nLabel);
    for iter = 1:iterMax
        if mod(iter - 1, iterStep) == 0
            indexRecord = (iter - 1) / iterStep + 1;
            yPred = ClassificationPredictBasic...
                (weightLinear, Xvalid, nHidden, nLabel);
            error(indexRecord, indexTrial) = sum(yPred ~= yvalid) / nvalid;
            fprintf('Training iteration = %d\tValidation error = %f\n', ...
                iter - 1, error(indexRecord, indexTrial));
        end
        indexTrain = ceil(rand * ntrain);
        [~, gradLinear] = NNClassificationLoss...
            (weightLinear, indexTrain, prob(indexTrial));
        weightLinear = weightLinear - alpha * gradLinear;
    end

    yPred = ClassificationPredictBasic...
        (weightLinear, Xtest, nHidden, nLabel);
    fprintf('Test error with final model = %f\n', ...
        sum(yPred ~= ytest) / ntest);
end

figure;
plot((0:iterRecord - 1) * iterStep, error, '-+');
legend('p = 1', 'p = 0.5')
xlabel('Iteration');
ylabel('Error rate');
title('Validation Set Error Rate with Dropout');