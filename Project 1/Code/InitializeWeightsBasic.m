function weightLinear = InitializeWeightsBasic(dim, nHidden, nLabel)
    nParams = dim * nHidden(1);
    for h = 2: length(nHidden)
        nParams = nParams + nHidden(h - 1) * nHidden(h);
    end
    nParams = nParams + nHidden(end) * nLabel;
    weightLinear = randn(nParams, 1);
end