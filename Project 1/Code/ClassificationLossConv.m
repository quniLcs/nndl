function [loss, gradLinear] = ClassificationLossConv...
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
        netActivation{1} = zeros(16, 16, 5);
        for indexConv = 1:5
            netActivation{1}(:, :, indexConv) = ...
                conv2(img, weightInput(:, :, indexConv), 'same') + ...
                biasInput(indexConv);
        end
        netActivation{1} = netActivation{1}(:)';
        Activation{1} = tanh(netActivation{1});
        for indexHidden = 2:length(nHidden)
            netActivation{indexHidden} = ...
                Activation{indexHidden - 1} * ...
                weightHidden{indexHidden - 1};
            Activation{indexHidden} = tanh(netActivation{indexHidden});
        end
        yPred = Activation{end} * weightOutput;
        
        error = yPred - y(indexInput, :);
        loss = loss + sum(error .^ 2);
        
        if nargout > 1
            % output layer
            error = 2 * error;
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
            for indexConv = 1:5
                for indexHori = 1:5
                    for indexVerti = 1:5
                        img = reshape(X(indexInput, 2:end), 16, 16);
                        kernal = zeros(5, 5);
                        kernal(indexVerti, indexHori) = 1;
                        img = conv2(img, kernal, 'same');
                        gradInput(5 * 5 * (indexConv - 1) + ...
                            5 * (indexHori - 1) + indexVerti) = ...
                            gradInput(5 * 5 * (indexConv - 1) + ...
                            5 * (indexHori - 1) + indexVerti) + ...
                            error(256 * (indexConv - 1) + 1: ...
                            256 * indexConv) * img(:);
                    end
                end
                gradInput(5 * 5 * 5 + indexConv) = ...
                    gradInput(5 * 5 * 5 + indexConv) + ...
                    sum(error(256 * (indexConv - 1) + 1: 256 * indexConv));
            end
        end
    end

    % put gradient into vector
    if nargout > 1
        gradLinear = FormGradConv(gradInput, gradHidden, gradOutput, ...
            weightLinear, nHidden, nLabel);
    end
end