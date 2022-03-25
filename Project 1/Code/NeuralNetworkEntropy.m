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

nLabel = max(ytrain);
nHidden = 10;

iterMax = 100000;
iterRecord = 20;
iterStep = floor(iterMax / iterRecord);

alpha = 1e-3;
error = zeros(iterRecord, 2);
expand = {@ones, @zeros};
NNClassificationLoss = ...
    {@(weightLinear, yExpanded, index)...
    ClassificationLossBasic(weightLinear, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel), ...
    @(weightLinear, yExpanded, index)...
    ClassificationLossEntropy(weightLinear, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel)};

for indexTrial = 1:2
    % pre-process y
    yExpanded = -expand{indexTrial}(ntrain, nLabel);
    for index = 1:ntrain
        yExpanded(index, ytrain(index)) = 1;
    end
    
    weightLinear = InitializeWeightsBasic(dim, nHidden, nLabel);
    
    for iter = 1:iterMax
        if mod(iter - 1, iterStep) == 0
            index = (iter - 1) / iterStep + 1;
            yPred = ClassificationPredictBasic...
                (weightLinear, Xvalid, nHidden, nLabel);
            error(index, indexTrial) = sum(yPred ~= yvalid) / nvalid;
            fprintf('Training iteration = %d\tValidation error = %f\n', ...
                iter - 1, error(index, indexTrial));
        end
        index = ceil(rand * ntrain);
        [~, gradLinear] = NNClassificationLoss{indexTrial}...
            (weightLinear, yExpanded, index);
        weightLinear = weightLinear - alpha * gradLinear;
    end

    yPred = ...
        ClassificationPredictBasic(weightLinear, Xtest, nHidden, nLabel);
    fprintf('Test error with final model = %f\n', ...
        sum(yPred ~= ytest) / ntest);
end
    
figure;
plot((0:iterRecord - 1) * iterStep, error, '-+');
legend('Squared Error', 'Cross Entropy Loss')
xlabel('Iteration');
ylabel('Error rate');
title('Validation Set Error Rate with Different Loss Function');