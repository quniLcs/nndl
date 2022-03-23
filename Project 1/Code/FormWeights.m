function [weightsInput, weightsHidden, weightsOutput] = ...
    FormWeights(weights, dim, nHidden, nLabel)

    weightsInput = reshape(weights(1: dim * nHidden(1)), dim, nHidden(1));
    offset = dim * nHidden(1);
    weightsHidden = cell(length(nHidden) - 1);
    for indexHidden = 2: length(nHidden)
        weightsHidden{indexHidden - 1} = reshape(weights(offset + 1: ...
            offset + nHidden(indexHidden - 1) * nHidden(indexHidden)), ...
            nHidden(indexHidden - 1), nHidden(indexHidden));
        offset = offset + nHidden(indexHidden - 1) * nHidden(indexHidden);
    end
    weightsOutput = reshape(weights(offset + 1: ...
        offset + nHidden(end) * nLabel), nHidden(end), nLabel);
end