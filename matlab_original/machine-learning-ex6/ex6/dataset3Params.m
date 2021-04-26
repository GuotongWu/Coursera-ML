function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

digit = [0.01;0.03;0.1;0.3;1;3;10;30];
error = zeros(length(digit), length(digit));
for k = 1:length(digit)
    for m = 1:length(digit)
        C = digit(k);
        sigma = digit(m);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error(k,m) = mean(double(predictions ~= yval));
    end
end
[~, I] = min(error(:));
[I_row, I_col] = ind2sub(size(error),I); % 将向量索引转换为矩阵索引
C = digit(I_row);
sigma = digit(I_col);

% =========================================================================

end
