function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');
a2 = [ones(size(a2, 1 ), 1) a2];
h_theta = sigmoid(a2 *Theta2');

Y = [];
for i = 1 : size(y)
    aux = zeros(1,10);
    aux(y(i)) = 1;
    Y = [Y ; aux];
end


for i = 1 : m
    for k = 1 : num_labels
        J = J - Y(i,k)*log(h_theta(i, k)) - (1 - Y(i,k))*log(1-h_theta(i, k));
    end
end

J = J / m;



regularization_term = 0;

for i = 1 : size(Theta1,1)
    for j = 2 : size(Theta1, 2)
        regularization_term = regularization_term + Theta1(i,j)*Theta1(i,j);
    end
end

for i = 1 : size(Theta2,1)
    for j = 2 : size(Theta2, 2)
        regularization_term = regularization_term + Theta2(i,j)*Theta2(i,j);
    end
end

regularization_term = regularization_term * lambda / (2*m);

J = J + regularization_term;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
