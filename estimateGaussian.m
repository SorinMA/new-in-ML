function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

[m, n] = size(X);


mu = zeros(n, 1);
sigma2 = zeros(n, 1);

for i = 1 : n

    mu(i) = 0;
    for j = 1 : m
        mu(i) = mu(i) + X(j,i);
    end
    mu(i) = mu(i) / m;

    for j = 1 : m
        sigma2(i) = sigma2(i) + (X(j,i) - mu(i))^2;
    end
    sigma2(i) = sigma2(i) / m;
end




end
