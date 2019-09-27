function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); 

J = 0;
grad = zeros(size(theta));

h_theta = sigmoid(X*theta);

J = 1 / m * ((-1*y)' * log(h_theta) - (ones(size(y)) - y)'*log(ones(size(h_theta)) - h_theta)) + lambda / (2*m) * theta(2:end)' * theta(2:end);

regularized_theta = lambda / m * theta;
regularized_theta(1) = 0;

grad = 1 / m * X' * (h_theta - y) + regularized_theta;

grad = grad(:);

end
