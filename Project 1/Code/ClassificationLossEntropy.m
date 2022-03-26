function [loss, gradLinear] = ClassificationLossEntropy...
    (weightLinear, X, y, nHidden, nLabel)

    [n,dim] = size(X);
    [weightInput, weightHidden, weightOutput] = ...
        FormWeightsBasic(weightLinear, dim, nHidden, nLabel);

    loss = 0;
    if nargout > 1
        [gradInput, gradHidden, gradOutput] = InitializeGradBasic...
            (weightInput, weightHidden, weightOutput, nHidden);
    end

    % compute output
    netActivation = cell(length(nHidden), 1);
    Activation = cell(length(nHidden), 1);
    for indexInput = 1:n
        netActivation{1} = X(indexInput, :) * weightInput;
        Activation{1} = tanh(netActivation{1});
        for indexHidden = 2:length(nHidden)
            netActivation{indexHidden} = Activation{indexHidden - 1} * ...
                weightHidden{indexHidden - 1};
            Activation{indexHidden} = tanh(netActivation{indexHidden});
        end
        yPred = Activation{end} * weightOutput;
        
        softmax = exp(yPred) / sum(exp(yPred));
        loss = loss - y(indexInput, :) * log(softmax)';
        
        if nargout > 1
            % output layer
            error = softmax - y(indexInput, :);
            gradOutput = gradOutput + Activation{end}' * error;
            error = sech(netActivation{end}) .^ 2 .* ...
                (error * weightOutput');
            % hidden layers
            for indexHidden = length(nHidden) - 1: -1: 1
                gradHidden{indexHidden} = gradHidden{indexHidden} + ...
                    Activation{indexHidden}' * error;
                error = (error * weightHidden{indexHidden}') .*  ...
                    sech(netActivation{indexHidden}) .^ 2;
            end
            % input layer
            gradInput = gradInput + X(indexInput,:)' * error;
        end
    end

    % put gradient into vector
    if nargout > 1
        gradLinear = FormGradBasic(gradInput, gradHidden, gradOutput, ...
            weightLinear, dim, nHidden, nLabel);
    end
end