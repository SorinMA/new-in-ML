function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

sigmoid_z = sigmoid(z);

for i = 1 : size(sigmoid_z, 1)
    for j = 1 : size(sigmoid_z, 2)
        if z(i,j) == 0
            g(i,j) = 0.25;
        else
            g(i,j) = sigmoid_z(i,j)*(1-sigmoid_z(i,j));
        end
    end
end


end
