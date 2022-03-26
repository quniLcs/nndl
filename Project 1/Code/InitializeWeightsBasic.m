function weightLinear = InitializeWeightsBasic(dim, nHidden, nLabel)
    nParams = dim * nHidden(1);
    for indexHidden = 2: length(nHidden)
        nParams = nParams + ...
            nHidden(indexHidden - 1) * nHidden(indexHidden);
    end
    nParams = nParams + nHidden(end) * nLabel;
    weightLinear = randn(nParams, 1);
end