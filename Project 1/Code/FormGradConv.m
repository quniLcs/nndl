function gradLinear = FormGradConv(gradInput, gradHidden, gradOutput, ...
    weightLinear, nHidden, nLabel)

    gradLinear = zeros(size(weightLinear));
    gradLinear(1: nHidden(1)) = gradInput(:);
    offset = nHidden(1);
    nHidden(1) = 16 * 16 * 5;
    for indexHidden = 2: length(nHidden)
        gradLinear(offset + 1:  offset + ...
            nHidden(indexHidden - 1) * nHidden(indexHidden)) = ...
            gradHidden{indexHidden - 1};
        offset = offset + ...
            nHidden(indexHidden - 1) * nHidden(indexHidden);
    end
    gradLinear(offset + 1: offset + nHidden(end) * nLabel) = gradOutput(:);
end