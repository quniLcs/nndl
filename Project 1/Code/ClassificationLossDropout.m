function [loss, gradLinear] = ClassificationLossDropout...
    (weightLinear, X, y, nHidden, nLabel, prob)

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
        Activation{1} = tanh(netActivation{1}) .* ...
            (rand(1, nHidden(1)) <= prob) / prob;
        for indexHidden = 2:length(nHidden)
            netActivation{indexHidden} = ...
                Activation{indexHidden - 1} * ...
                weightHidden{indexHidden - 1};
            Activation{indexHidden} = ...
                tanh(netActivation{indexHidden}) .* ...
                (rand(1, nHidden(indexHidden)) <= prob) / prob;
        end
        yPred = Activation{end} * weightOutput;
        
        error = yPred - y(indexInput, :);
        loss = loss + sum(error .^ 2);
        
        if nargout > 1
            % output layer
            error = 2 * error;
            gradOutput = gradOutput + Activation{end}' * error;
            error = sech(netActivation{end}) / prob .^ 2 .* ...
                (error * weightOutput');
            % hidden layers
            for indexHidden = length(nHidden) - 1: -1: 1
                gradHidden{indexHidden} = gradHidden{indexHidden} + ...
                    Activation{indexHidden}' * error;
                error = sech(netActivation{indexHidden}) / prob .^ 2 .* ... 
                    (error * weightHidden{indexHidden}');
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