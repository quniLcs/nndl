function [loss, grad] = ...
    ClassificationLossRegularization(weights, X, y, nHidden, nLabel)

    [n,dim] = size(X);
    [weightsInput, weightsHidden, weightsOutput] = ...
        FormWeights(weights, dim, nHidden, nLabel);

    loss = 0;
    if nargout > 1
        [gradInput, gradHidden, gradOutput] = ...
            InitializeGrad(weightsInput, weightsHidden, weightsOutput, ...
            nHidden);
    end

    % compute output
    netActivation = cell(length(nHidden), 1);
    Activation = cell(length(nHidden), 1);
    for indexInput = 1:n
        netActivation{1} = X(indexInput, :) * weightsInput;
        Activation{1} = tanh(netActivation{1});
        for indexHidden = 2:length(nHidden)
            netActivation{indexHidden} = Activation{indexHidden - 1} * ...
                weightsHidden{indexHidden - 1};
            Activation{indexHidden} = tanh(netActivation{indexHidden});
        end
        yPred = Activation{end} * weightsOutput;
        
        error = yPred - y(indexInput, :);
        loss = loss + sum(error .^ 2);
        
        if nargout > 1
            % output layer
            error = 2 * error;
            gradOutput = gradOutput + Activation{end}' * error;
            error = sech(netActivation{end}) .^ 2 .* ...
                (error * weightsOutput');
            % hidden layers
            for indexHidden = length(nHidden) - 1: -1: 1
                gradHidden{indexHidden} = gradHidden{indexHidden} + ...
                    Activation{indexHidden}' * error;
                error = (error * weightsHidden{indexHidden}') .*  ...
                    sech(netActivation{indexHidden}) .^ 2;
            end
            % input layer
            gradInput = gradInput + X(indexInput,:)' * error;
        end
    end

    % put gradient into vector
    if nargout > 1
        grad = FormGrad(gradInput, gradHidden, gradOutput, ...
            weights, dim, nHidden, nLabel);
    end
end