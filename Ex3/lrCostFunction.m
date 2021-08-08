function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%

% Trick to avoid regularizing \theta_0 (0-th element is always 0)

theta_0 = theta;
theta_0(1) = 0;


h_theta = sigmoid(X * theta);  % vector (5000x400 x 400x1 = 5000x1)
J1 = -1 * y' * log(h_theta);   % scalar (1x5000 x 5000x1 = 1x1)
J2 = -1 * (ones(m, 1) -y)' * log(ones(length(h_theta), 1) - h_theta);  % scalar (1x5000 x 5000x1 = 1x1)
R = (lambda / (2 * m)) * (theta_0' * theta_0);
J = (1 / m) * (J1 + J2) + R;

% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

grad = (1/m) * (X' * (h_theta - y)) + (lambda / m) * theta_0;









% =============================================================

grad = grad(:);

end
