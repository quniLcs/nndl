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

iterMax = 100000;
iterRecord = 20;
iterStep = floor(iterMax / iterRecord);

alpha = 1e-3;

error = zeros(iterRecord, 2);
nHidden = [10; 5 * 5 + 1];
NNInitializeWeights = ...
    {@(nHidden, nLabel) InitializeWeightsBasic(dim, nHidden, nLabel), ...
    @(nHidden, nLabel) InitializeWeightsConv(nHidden, nLabel)};
NNClassificationLoss = ...
    {@(weightLinear, index, nHidden)...
    ClassificationLossBasic(weightLinear, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel),...
    @(weightLinear, index, nHidden)...
    ClassificationLossConv(weightLinear, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel)};
NNClassificationPredict = ...
    {@ClassificationPredictBasic, ...
    @ClassificationPredictConv};

for indexTrial = 1:2
    weightLinear = NNInitializeWeights{indexTrial}...
        (nHidden(indexTrial, :), nLabel);
    for iter = 1:iterMax
        if mod(iter - 1, iterStep) == 0
            indexRecord = (iter - 1) / iterStep + 1;
            yPred = NNClassificationPredict{indexTrial}...
                (weightLinear, Xvalid, nHidden(indexTrial, :), nLabel);
            error(indexRecord, indexTrial) = sum(yPred ~= yvalid) / nvalid;
            fprintf('Training iteration = %d\tValidation error = %f\n', ...
                iter - 1, error(indexRecord, indexTrial));
        end
        indexTrain = ceil(rand * ntrain);
        [~, gradLinear] = NNClassificationLoss{indexTrial}...
            (weightLinear, indexTrain, nHidden(indexTrial, :));
        weightLinear = weightLinear - alpha * gradLinear;
    end

    yPred = NNClassificationPredict{indexTrial}...
        (weightLinear, Xtest, nHidden(indexTrial, :), nLabel);
    fprintf('Test error with final model = %f\n', ...
        sum(yPred ~= ytest) / ntest);
end

figure;
plot((0:iterRecord - 1) * iterStep, error, '-+');
legend('without convolution', 'with convolution');
xlabel('Iteration');
ylabel('Error rate');
title('Validation Set Error Rate with or without Convolution');