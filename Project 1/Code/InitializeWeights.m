function weights = InitializeWeights(dim, nHidden, nLabel)
    nParams = dim * nHidden(1);
    for h = 2: length(nHidden)
        nParams = nParams + nHidden(h - 1) * nHidden(h);
    end
    nParams = nParams + nHidden(end) * nLabel;
    weights = randn(nParams, 1);
end