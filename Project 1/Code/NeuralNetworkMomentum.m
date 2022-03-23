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

NNClassificationLoss = @(weights, index) ClassificationLoss(weights, ...
    Xtrain(index,:), yExpanded(index,:), nHidden, nLabel);

alpha = [repmat([1e-3, 1e-2, 1e-4, 1e-3], iterMax, 1), ...
    1e-2 ./ exp(1e-5 * (1:iterMax))', ...
    1e-2 * cos((1:iterMax) / iterMax * pi / 2)'];
rho = [0, 0, 0, 0.9, 0, 0];
nTrials = length(rho);
error = zeros(iterRecord, nTrials);

for indexTrial = 1:nTrials
    velocity = 0;
    weights = InitializeWeights(dim, nHidden, nLabel);
    
    for iter = 1:iterMax
        if mod(iter - 1, iterStep) == 0
            indexRecord = (iter - 1) / iterStep + 1;
            yPred = ClassificationPredict(weights, ...
                Xvalid, nHidden, nLabel);
            error(indexRecord, indexTrial) = sum(yPred ~= yvalid) / nvalid;
            fprintf('Training iteration = %d\tValidation error = %f\n', ...
                iter - 1, error(indexRecord, indexTrial));
        end
        indexTrain = ceil(rand * ntrain);
        [~, grad] = NNClassificationLoss(weights, indexTrain);
        velocity = rho(indexTrial) * velocity + grad;
        weights = weights - alpha(iter, indexTrial) * velocity;
    end

    yPred = ClassificationPredict(weights, Xtest, nHidden, nLabel);
    fprintf('Test error with final model = %f\n', ...
        sum(yPred ~= ytest) / ntest);
end

figure;
plot((0:iterRecord - 1) * iterStep, error, '-+');
legend('learning rate 1e-3', 'learning rate 1e-2', ...
    'learning rate 1e-4', 'learning rate 1e-3 with momentum', ...
    'learning rate exponential decay', 'learning rate cosine decay');
xlabel('Iteration');
ylabel('Error rate');
title('Validation Set Error Rate with Different Optimization Method');