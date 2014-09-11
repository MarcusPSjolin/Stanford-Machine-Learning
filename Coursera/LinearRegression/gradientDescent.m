function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    prediction1 = 0.0;
    prediction2 = 0.0;

    for j = 1:m
        h = theta(1) + theta(2) * X(j, 2);
        prediction1 = prediction1 + (h - y(j)) * X(j, 1);
        prediction2 = prediction2 + (h - y(j)) * X(j, 2);
    end
    
    expression = alpha * (1/m);
    temp_theta1 = theta(1) - expression * prediction1;
    temp_theta2 = theta(2) - expression * prediction2;
    
    theta(1) = temp_theta1;
    theta(2) = temp_theta2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
