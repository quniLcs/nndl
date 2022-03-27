function yPred = ClassificationPredictFinal...
    (weightLinear, X, nHidden, nLabel)

    [n, ~] = size(X);
    [weightInput, biasInput, weightHidden, weightOutput] = ...
        FormWeightsConv(weightLinear, nHidden, nLabel);

    % compute output
    NetActivation = cell(length(nHidden), 1);
    Activation = cell(length(nHidden), 1);
    yPred = zeros(n, nLabel);
    for indexInput = 1:n
        img = reshape(X(indexInput, 2:end), 16, 16);
        NetActivation{1} = conv2(img, weightInput, 'same') + biasInput;
        NetActivation{1} = NetActivation{1}(:)';
        Activation{1} = tanh(NetActivation{1});
        for indexHidden = 2: length(nHidden)
            NetActivation{indexHidden} = Activation{indexHidden - 1} * ...
                weightHidden{indexHidden - 1};
            Activation{indexHidden} = tanh(NetActivation{indexHidden});
        end
        yPred(indexInput,:) = Activation{end} * weightOutput;
    end
    [~, yPred] = max(yPred, [], 2);
end
