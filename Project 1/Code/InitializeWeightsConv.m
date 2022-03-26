function weightLinear = InitializeWeightsConv(nHidden, nLabel)
    nParams = nHidden(1);
    nHidden(1) = 16 * 16;
    for indexHidden = 2: length(nHidden)
        nParams = nParams + ...
            nHidden(indexHidden - 1) * nHidden(indexHidden);
    end
    nParams = nParams + nHidden(end) * nLabel;
    weightLinear = randn(nParams, 1);
end