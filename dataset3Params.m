function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vals = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10, 30];
pred_err = zeros(length(vals));

for C = vals
    for sigma = vals
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        pred_err(find(vals == C),find(vals == sigma)) = mean(double(predictions ~= yval));
    end
end

best_soln = min(pred_err(:));
[C_ind, sigma_ind] = find(pred_err == best_soln);
C = vals(C_ind);
sigma = vals(sigma_ind);

% =========================================================================

end
