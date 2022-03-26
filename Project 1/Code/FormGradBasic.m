function gradLinear = FormGradBasic(gradInput, gradHidden, gradOutput, ...
    weightLinear, dim, nHidden, nLabel)

    gradLinear = zeros(size(weightLinear));
    gradLinear(1: dim * nHidden(1)) = gradInput(:);
    offset = dim * nHidden(1);
    for indexHidden = 2: length(nHidden)
        gradLinear(offset + 1:  offset + ...
            nHidden(indexHidden - 1) * nHidden(indexHidden)) = ...
            gradHidden{indexHidden - 1};
        offset = offset + ...
            nHidden(indexHidden - 1) * nHidden(indexHidden);
    end
    gradLinear(offset + 1: offset + nHidden(end) * nLabel) = gradOutput(:);
end