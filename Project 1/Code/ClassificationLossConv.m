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
                % for indexHori = 1:5
                %     for indexVerti = 1:5
                %         img = reshape(X(indexInput, 2:end), 16, 16);
                %         kernal = zeros(5, 5);
                %         kernal(indexVerti, indexHori) = 1;
                %         img = conv2(img, kernal, 'same');
                %         gradInput(5 * 5 * (indexConv - 1) + ...
                %             5 * (indexHori - 1) + indexVerti) = ...
                %             gradInput(5 * 5 * (indexConv - 1) + ...
                %             5 * (indexHori - 1) + indexVerti) + ...
                %             error(256 * (indexConv - 1) + 1: ...
                %             256 * indexConv) * img(:);
                %     end
                % end
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