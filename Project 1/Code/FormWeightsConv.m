function [weightInput, biasInput, weightHidden, weightOutput] = ...
    FormWeightsConv(weightLinear, nHidden, nLabel)

    weightInput = reshape(weightLinear(1: 5 * 5 * 5), 5, 5, 5);
    biasInput = weightLinear(5 * 5 * 5 + 1: nHidden(1));
    offset = nHidden(1);
    nHidden(1) = 16 * 16 * 5;
    weightHidden = cell(length(nHidden) - 1);
    for indexHidden = 2: length(nHidden)
        weightHidden{indexHidden - 1} = ...
            reshape(weightLinear(offset + 1: offset + ...
            nHidden(indexHidden - 1) * nHidden(indexHidden)), ...
            nHidden(indexHidden - 1), nHidden(indexHidden));
        offset = offset + nHidden(indexHidden - 1) * nHidden(indexHidden);
    end
    weightOutput = reshape(weightLinear(offset + 1: offset + ...
        nHidden(end) * nLabel), nHidden(end), nLabel);
end