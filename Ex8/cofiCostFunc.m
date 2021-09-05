function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta

%% Cost function
j = (X * Theta' - Y);  %% nmxnf x nfxnu - nmxnu == nmxnu
j2 = (1/2) * (j .^2);
J = sum(sum(j2 .* R));

%% Adding regularization
rTheta = sum(sum(Theta .* Theta, 1));
rX = sum(sum(X .* X, 1));

J = J + (lambda / 2) * (rTheta + rX);

% =============================================================

%% X_grad (vectorized)
%% Loop -over movies-
for i=1:size(X,1)
    %% Users who rated -this- movie
    idx = find(R(i,:) == 1);

    %% Preferences (for these users)
    Theta_tmp = Theta(idx,:);

    %% Ratings (for movie i-th and selected users)
    Y_tmp = Y(i, idx);

    X_grad(i,:) = (X(i,:) * Theta_tmp' - Y_tmp) * Theta_tmp;

    %% Adding regularization
    X_grad(i,:) += lambda * X(i,:);
end

%% Theta_grad (vectorized)
%% Loop -over users-
for j=1:size(Theta,1)
    %% Movies rated by -this- user
    idx = find(R(:,j) == 1);

    %% Movies information about -these- features
    %% size: idx x nf
    X_tmp = X(idx,:);

    %% Movie ratings by j-th user
    %% size: idx x 1
    Y_tmp = Y(idx, j);

    %% size Theta(j,:) 1xnf
    Theta_grad(j,:) = X_tmp' * (X_tmp * Theta(j,:)' - Y_tmp);

    %% Adding regularization
    Theta_grad(j,:) += lambda * Theta(j,:);
end


grad = [X_grad(:); Theta_grad(:)];

end
