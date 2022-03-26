function [gradInput, gradHidden, gradOutput] = InitializeGradBasic...
    (weightInput, weightHidden, weightOutput, nHidden)

    gradInput = zeros(size(weightInput));
    gradHidden = cell(length(nHidden) - 1);
    for indexHidden = 2: length(nHidden)
        gradHidden{indexHidden - 1} = ...
            zeros(size(weightHidden{indexHidden - 1})); 
    end
    gradOutput = zeros(size(weightOutput));
end