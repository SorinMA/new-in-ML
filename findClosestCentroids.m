function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

for i = 1 : size(X, 1)
    min_centroid = 0;

    for j = 1 : size(centroids, 1)
        aux_centr = 0;
        for k = 1 : size(centroids,2)
            aux_centr = aux_centr + (X(i,k) + centroids(j,k))^2;
        end
        if min_centroid < aux_centr
            min_centroid = aux_centr;
        end
    end

    for j = 1 : size(centroids, 1)
        aux_dist = 0;
        for k = 1 : size(centroids, 2)
            aux_dist = aux_dist + (X(i,k) - centroids(j,k))^2;
        end

        if aux_dist < min_centroid
            min_centroid = aux_dist;
            idx(i) = j;
        end

    end

end


end

