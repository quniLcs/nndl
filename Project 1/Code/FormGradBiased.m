function grad = FormGradBiased(gradInput, gradHidden, gradOutput, ...
    weights, dim, nHidden, nLabel)

    grad = zeros(size(weights));
    grad(1: dim * (nHidden(1) - 1)) = gradInput(:);
    offset = dim * (nHidden(1) - 1);
    for indexHidden = 2: length(nHidden)
        grad(offset + 1: offset + ...
            nHidden(indexHidden - 1) * (nHidden(indexHidden) - 1)) = ...
            gradHidden{indexHidden - 1};
        offset = offset + ...
            nHidden(indexHidden - 1) * (nHidden(indexHidden) - 1);
    end
    grad(offset + 1: offset + nHidden(end) * nLabel) = gradOutput(:);
end