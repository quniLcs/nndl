function [loss, gradLinear] = ClassificationLossFinal...
    (weightLinear, X, y, nHidden, nLabel)

    [n, ~] = size(X);
    [weightInput, biasInput, weightHidden, weightOutput] = ...
        FormWeightsConv(weightLinear, nHidden, nLabel);

    loss = 0;
    if nargout > 1
        [gradInput, gradHidden, gradOutput] = InitializeGradBasic...
            ([weightInput(:); biasInput], ...
            weightHidden, weightOutput, nHidden);
    end

    % compute output
    netActivation = cell(length(nHidden), 1);
    Activation = cell(length(nHidden), 1);
    for indexInput = 1:n
        img = reshape(X(indexInput, 2:end), 16, 16);
        netActivation{1} = conv2(img, weightInput, 'same') + biasInput;
        netActivation{1} = netActivation{1}(:)';
        Activation{1} = tanh(netActivation{1});
        for indexHidden = 2:length(nHidden)
            netActivation{indexHidden} = ...
                Activation{indexHidden - 1} * ...
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
                error = sech(netActivation{indexHidden}) .^ 2 .* ... 
                    (error * weightHidden{indexHidden}');
            end
            % input layer
            kernal = reshape(error, 16, 16);
            img = reshape(X(indexInput, 2:end), 16, 16);
            img = imrotate(img, 180);
            conv = conv2(img, kernal, 'full');
            conv = conv(14:18, 14:18);
            gradInput(1: 5 * 5) = gradInput(1: 5 * 5) + conv(:);
            gradInput(end) = gradInput(end) + sum(error);
        end
    end

    % put gradient into vector
    if nargout > 1
        gradLinear = FormGradConv(gradInput, gradHidden, gradOutput, ...
            weightLinear, nHidden, nLabel);
    end
end