function [gradInput, gradHidden, gradOutput] = ...
    InitializeGrad(weightsInput, weightsHidden, weightsOutput, nHidden)

    gradInput = zeros(size(weightsInput));
    gradHidden = cell(length(nHidden) - 1);
    for indexHidden = 2: length(nHidden)
        gradHidden{indexHidden - 1} = ...
            zeros(size(weightsHidden{indexHidden - 1})); 
    end
    gradOutput = zeros(size(weightsOutput));
end