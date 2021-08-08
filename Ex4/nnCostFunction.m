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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

# Add a column of ones (the bias)
Xbiased = [ ones(m, 1), X ];

% --------------------------------------------------------
% Let's define an auxiliary function that implements
% forward propagation for ONE input
% --------------------------------------------------------
function [h_theta_i, z_2, a_2, z_3, a_3] = fpHypothesis (Xi, theta_1, theta_2)
    %
    % The inputX is a single row (from X); transpose it
    z_2 = theta_1 * Xi';  % 25x401 x 401x1 == 25x1 (vector)
    a_2 = sigmoid(z_2);  % 25x1 (vector)
    a_2_biased = [ 1; a_2 ]; 
    z_3 = theta_2 * a_2_biased;  % 10x26  x 26x1 == 10x1 (vector)
    a_3 = sigmoid(z_3);

    h_theta_i = a_3;

endfunction


% --------------------------------------------------------
% This function translates between y \in \mathbb{R}
% and its vector form
% --------------------------------------------------------

function y_vec = y_to_vec(y_val)
    a = 1:num_labels;
    y_vec = a' == y_val;
endfunction


% --------------------------------------------------------
% This function computes the component of \Theta_i to
% the regularization factor
% --------------------------------------------------------

function Rl = Regularization(T)
    Rl = 0;
    [sl, slp] = size(T);

    printf("Regularization dimensions: %d x %d\n", sl, slp)
    for j=1:sl
        for k=1:slp
            Rl += T(j, k) ** 2;
        endfor
    endfor

endfunction


% Trick to avoid regularizing \Theta_{0j}^{(l)} (the first column is always 0)
theta_1_0 = Theta1;
theta_1_0(:, 1) = 0;
theta_2_0 = Theta2;
theta_2_0(:, 1) = 0;

% Regularization factor
R = (lambda / (2 * m)) * (Regularization(theta_1_0) + Regularization(theta_2_0));

for i=1:m
    yi = y_to_vec(y(i));  % 10x1  (K=10)
    xi = Xbiased(i, :);  % 1x401  (input size = 20x20)
    [hi, ~] = fpHypothesis(xi, Theta1, Theta2);  % 10x1 (output vector)
    J1 = -1 * yi' * log(hi); % 1x10 x 10x1 == 1x1 (\in \mathbb{R})
    J2 = -1 * (ones(num_labels, 1) - yi)' * log(ones(num_labels, 1) - hi);  % 1xK x Kx1 = 1x1 (\in \mathbb{R})
    Ji = (1 / m) * (J1 + J2);
    J += Ji;

endfor

% Add regularization
J += R;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

Delta_1 = 0;
Delta_2 = 0;

for t =1:m
    % Using fpHypothesis we get all NN parameters we need
    a_1 = Xbiased(t,:);  % This contains the bias already
    [hi, z_2, a_2, z_3, a_3] = fpHypothesis(a_1, Theta1, Theta2);
    a_2 = [1; a_2];  % I need to add the bias to this
    z_2 = [1; z_2];

    % Calculate the error at the output layer. This one is easy ;)
    delta_3 = a_3 - y_to_vec(y(t));  % vectorized  Kx1

    % Calculate the delta for the hidden layer using these values
    delta_2 = Theta2' * delta_3 .* sigmoidGradient(z_2);  % 26xK x Kx1 (26x1) .* 26x1 == 26x1
    delta_2 = delta_2(2:end);  % remove \delta_0^{(2)}

    Delta_2 += delta_3 * a_2';  % Kx1 x 1x26 == K x 26  size(Theta2)

    % Same regarding the influence of Theta1
    delta_1 = Theta1' * delta_2 .* sigmoidGradient(a_1);  % 401x25 x 25x1 (401x1) .* 401x1 == 401x1
    delta_1 = delta_1(2:end);   % remove \delta_0^{(1)}

    Delta_1 += delta_2 * a_1;  % 25x1 x 1x401 == 25x401 size(Theta1)

endfor

% Calculate the gradients
Theta1_grad = (1 / m) * Delta_1;
Theta2_grad = (1 / m) * Delta_2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad += (lambda / m) * theta_1_0;
Theta2_grad += (lambda / m) * theta_2_0;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


endfunction

