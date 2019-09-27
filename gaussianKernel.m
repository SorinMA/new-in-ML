function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

x1 = x1(:); x2 = x2(:);
sim = 0;
sim_sum = 0;

for i = 1 : size(x1)
    sim_sum = sim_sum + (x1(i) - x2(i)) * (x1(i) - x2(i));
end

sim = exp(- sim_sum / (2 * sigma * sigma));

end
