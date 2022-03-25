function [feature, yPred] = ...
    ClassificationPredictTuning(weightLinear, X, nHidden, nLabel)

    [n, dim] = size(X);
    [weightInput, weightHidden, weightOutput] = ...
        FormWeightsBasic(weightLinear, dim, nHidden, nLabel);

    % compute output
    NetActivation = cell(length(nHidden), 1);
    Activation = cell(length(nHidden), 1);
    feature = zeros(n, nHidden(end));
    yPred = zeros(n, nLabel);
    for indexInput = 1:n
        NetActivation{1} = X(indexInput, :) * weightInput;
        Activation{1} = tanh(NetActivation{1});
        for indexHidden = 2: length(nHidden)
            NetActivation{indexHidden} = Activation{indexHidden - 1} * ...
                weightHidden{indexHidden - 1};
            Activation{indexHidden} = tanh(NetActivation{indexHidden});
        end
        feature(indexInput,:) = Activation{end};
        yPred(indexInput,:) = Activation{end} * weightOutput;
    end
    [~, yPred] = max(yPred, [], 2);
end
