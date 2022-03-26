function [loss, gradLinear] = ClassificationLossAugmentation...
    (weightLinear, X, y, nHidden, nLabel, prob)

    [n, dim] = size(X);
    [weightInput, weightHidden, weightOutput] = ...
        FormWeightsBasic(weightLinear, dim, nHidden, nLabel);

    loss = 0;
    if nargout > 1
        [gradInput, gradHidden, gradOutput] = InitializeGradBasic...
            (weightInput, weightHidden, weightOutput, nHidden);
    end
    
    netActivation = cell(length(nHidden), 1);
    Activation = cell(length(nHidden), 1);
    for indexInput = 1:n
        % data augmentation
        img = reshape(X(indexInput, 2:end), 16, 16);
        % image translate
        if rand < prob(1)
            img = imtranslate(img, randn(1, 2), 'FillValues', rand);
        end
        % image rotation
        if rand < prob(2)
            img = imrotate(img, 5 * randn, 'crop');
        end
        % image resizing
        if rand < prob(3)
            resize = rand / 10 + 1;
            img = imresize(img, resize);
            [len, ~] = size(img);
            horizon = randi([1, len - 16 + 1]);
            vertical = randi([1, len - 16 + 1]);
            img = imcrop(img, [horizon, vertical, 16 - 1, 16 - 1]);
        end
        X(indexInput, :) = [1, img(:)'];
        
        % compute output
        netActivation{1} = X(indexInput, :) * weightInput;
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
            gradInput = gradInput + X(indexInput,:)' * error;
        end
    end

    % put gradient into vector
    if nargout > 1
        gradLinear = FormGradBasic(gradInput, gradHidden, gradOutput, ...
            weightLinear, dim, nHidden, nLabel);
    end
end