function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%


X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

    
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


J = sum(sum(R.*((( X*Theta' - Y).^2)./2)));

 

for i = 1: size(X,1)
    for j = 1 : size(R,2)
        if R(i,j) == 1
            X_grad(i,:) = X_grad(i,:) + (X(i,:)*Theta(j,:)' - Y(i,j))*Theta(j,:);
        end
    end
    X_grad(i,:) = X_grad(i,:) + lambda.*X(i,:);
end

for j = 1 : size(R,2)
    for i = 1 : size(R,1)
        if R(i,j) == 1
            Theta_grad(j,:) = Theta_grad(j,:) + (X(i,:)*Theta(j,:)' - Y(i,j))*X(i,:);
        end
    end
     Theta_grad(j,:) = Theta_grad(j,:) + lambda.*Theta(j,:);
end


%  regularized

reg1_term_j = 0;
reg2_term_j = 0;

for i = 1 : num_users
    for j = 1 : num_features
        reg1_term_j = reg1_term_j + (Theta(i,j)^2) * lambda / 2;
    end
end

for i = 1 : num_movies
    for j = 1 : num_features
        reg2_term_j = reg2_term_j + (X(i,j)^2) * lambda / 2;
    end
end

J = J + reg1_term_j + reg2_term_j;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
