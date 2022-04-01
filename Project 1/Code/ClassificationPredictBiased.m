function yPred = ClassificationPredictBiased...
    (weightLinear, X, nHidden, nLabel)

    [n, dim] = size(X);
    [weightInput, weightHidden, weightOutput] = ...
        FormWeightsBiased(weightLinear, dim, nHidden, nLabel);

    % compute output
    netActivation = cell(length(nHidden), 1);
    Activation = cell(length(nHidden), 1);
    yPred = zeros(n, nLabel);
    for indexInput = 1:n
        netActivation{1} = X(indexInput, :) * weightInput;
        Activation{1} = [tanh(netActivation{1}), 1];
        for indexHidden = 2: length(nHidden)
            netActivation{indexHidden} = Activation{indexHidden - 1} * ...
                weightHidden{indexHidden - 1};
            Activation{indexHidden} = ...
                [tanh(netActivation{indexHidden}), 1];
        end
        yPred(indexInput,:) = Activation{end} * weightOutput;
    end
    [~, yPred] = max(yPred, [], 2);
end
