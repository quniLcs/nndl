function [S, mu, sigma] = Standardize(M, mu, sigma)
    % mu and sigma are computed from M if omitted
    
    [nrows, ~] = size(M);
    
    if nargin == 1
        mu = mean(M);
        sigma = std(M);
        sigma(sigma < eps) = 1;
    end
    
    S = M - repmat(mu, [nrows, 1]);
    S = S ./ repmat(sigma, [nrows, 1]);
end
