function y = sigmaFunc(x)
    % range input to 0-1
    y = 1 ./ (1 + exp(-x));
end