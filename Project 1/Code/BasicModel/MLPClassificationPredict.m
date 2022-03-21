function yPred = MLPClassificationPredict(weights, X, nHidden, nLabel)
    [n, dim] = size(X);
    [weightsInput, weightsHidden, weightsOutput] = ...
        FormWeights(weights, dim, nHidden, nLabel);

    % compute output
    NetActivation = cell(length(nHidden));
    Activation = cell(length(nHidden));
    yPred = zeros(n, nLabel);
    for indexInput = 1:n
        NetActivation{1} = X(indexInput, :) * weightsInput;
        Activation{1} = tanh(NetActivation{1});
        for indexHidden = 2: length(nHidden)
            NetActivation{indexHidden} = Activation{indexHidden - 1} * ...
                weightsHidden{indexHidden - 1};
            Activation{indexHidden} = tanh(NetActivation{indexHidden});
        end
        yPred(indexInput,:) = Activation{end} * weightsOutput;
    end
    [~, yPred] = max(yPred, [], 2);
end
