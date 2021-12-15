%   __________________________
%   COMP6011: Machine Learning
%   19129576
%   MLP MNIST CW
%   _____________________________



% Note: this file merely specifies the MLP class. It is not meant to be
% executed as a stand-alone script. The MLP needs to be instantiated and
% then used elsewhere, see e.g. 'testMLP131train.m'.

% A Multi-layer perceptron class
classdef MLP < handle
    % Member data
    properties (SetAccess=private)
        inputDimension % Number of inputs
        hiddenDimension % Number of hidden neurons
        outputDimension % Number of outputs
        
        hiddenLayerWeights % Weight matrix for the hidden layer, format (hiddenDim)x(inputDim+1) to include bias terms
        outputLayerWeights % Weight matrix for the output layer, format (outputDim)x(hiddenDim+1) to include bias terms

    end
    
    methods
        % Constructor: Initialize to given dimensions and set all weights
        % zero.
        % inputD ~ dimensionality of input vectors
        % hiddenD ~ number of neurons (dimensionality) in the hidden layer 
        % outputD ~ number of neurons (dimensionality) in the output layer 
        function mlp=MLP(inputD,hiddenD,outputD)
            mlp.inputDimension=inputD;
            mlp.hiddenDimension=hiddenD;
            mlp.outputDimension=outputD;
            mlp.hiddenLayerWeights=zeros(hiddenD,inputD+1);
            mlp.outputLayerWeights=zeros(outputD,hiddenD+1);
            
          
            
        end
        
        % TODO Implement a randomized initialization of the weight
        % matrices.
        % Use the 'stdDev' parameter to control the spread of initial
        % values.
        function mlp=initializeWeightsRandomly(mlp,stdDev)
            % Note: 'mlp' here takes the role of 'this' (Java/C++) or
            % 'self' (Python), refering to the object instance this member
            % function is run on.
            
            
            % use zeroes function which creates a scalar on the
            % corresponding layers (zeros
            
             mlp.hiddenLayerWeights=zeros(mlp.hiddenDimension,mlp.inputDimension+stdDev);% TODO
             mlp.outputLayerWeights=zeros(mlp.outputDimension,mlp.hiddenDimension+stdDev);% TODO
            
            %initialize a set of weights for the hidden and output layer
             mlp.hiddenLayerWeights = (rand(mlp.hiddenDimension, mlp.inputDimension +1) -1)*stdDev;
             mlp.outputLayerWeights = (rand(mlp.outputDimension, mlp.hiddenDimension +1) -1)*stdDev;
            
        
            
             %create new set of hidden weights using averages
            mlp.hiddenLayerWeights = mlp.hiddenLayerWeights./size( mlp.hiddenLayerWeights, 2);
            
            %create new set of output weights using averages
            mlp.outputLayerWeights = mlp.outputLayerWeights./size( mlp.outputLayerWeights, 2); 
            
        end
        
        % TODO Implement the forward-propagation of values algorithm in
        % this method.
        % 
        % inputData ~ a vector of data representing a single input to the
        % network in column format. It's dimension must fit the input
        % dimension specified in the contructor.
        % 
        % hidden ~ output of the hidden-layer neurons
        % output ~ output of the output-layer neurons
        % 
        % Note: the return value is automatically fit into a array
        % containing the above two elements
        function [hidden,output]=compute_forward_activation(mlp, inputData)

            % Choose activation function, logistic sigmoid
            activationFunction = @logisticSigmoid;

                %mlp.hiddenDimension = mlp.hiddenLayerWeights * hiddenVector;
                %creates a new hidden dimension using the hidden weights
                %with input. returns a vector using the activiation func
                hiddenInput  = (mlp.hiddenLayerWeights * [inputData;1]);
                hidden = activationFunction(hiddenInput);
            
                %mlp.outputDimension = mlp.outputLayerWeights.* hiddenOutputVector;
                %creates a new output dimension using the output weights
                %and hidden vector
                
                outputInput  = (mlp.outputLayerWeights.* [hidden;1]);
                
                output = activationFunction(outputInput);
                
                    
                % Adjust inputData including bias
                inputData(mlp.inputDimension+1,1) = 1;
                hidin = mlp.hiddenLayerWeights * inputData;
                % Apply activation function
                hidden = 1./(1 + exp(-hidin));
            
                % Adjust hidden neurons with respect to bias
                hidden(mlp.hiddenDimension+1,1) = 1;
                hidout = mlp.outputLayerWeights * hidden;
                output = 1./(1 + exp(-hidout));

                        
        end
        
        
        % This function calls the forward propagation and extracts only the
        % overall output. It does not have to be altered.
        function output=compute_output(mlp,input)
            [~,output] = mlp.compute_forward_activation(input);
            
        end
        
        
        % TODO Implement the backward-propagation of errors (learning) algorithm in
        % this method.
        %
        % This method implements MLP learning by means on backpropagation
        % of errors on a single data point.
        %
        % inputData ~  a vector of data representing a single input to the
        %   network in column format.
        % targetOutputData ~ a vector of data representing the
        %   desired/correct output the network should generate for the given
        %   input (this is the supervision signal for learning)
        % learningRate ~ step width for gradient descent
        %
        % This method is expected to update mlp.hiddenLayerWeights and
        % mlp.outputLayerWeights.
        function mlp=train_single_data(mlp, inputData, targetOutputData, learningRate)
            
            % sigmoid derivative is sigmoid(x) * (1-sigmoid(x))
            [hidden,output] = mlp.compute_forward_activation(inputData);
            
            outputWeights = mlp.outputLayerWeights(:,1:mlp.hiddenDimension);
            
            %Error with respect to the output layer activation
            errorOutput = 2*(targetOutputData - output);

            %Backprop output weights
            backpropOutputWeights = hidden'.*errorOutput.*(output.*(1-output));
            backpropOutputBias = output.*errorOutput.*(output.*(1-output));
            
            %hidden layer error
            errorHidden = (outputWeights'*errorOutput).*(1./(1+exp(-hidden)));
            
            %Backprop hidden weights
            backpropHiddenWeights = inputData'.*errorHidden.*(hidden.*(1-hidden));
            backpropHiddenBias = hidden'*errorHidden.*(hidden.*(1-hidden));
            
            % Update weights   
            mlp.hiddenLayerWeights = mlp.hiddenLayerWeights + ([backpropHiddenWeights backpropHiddenBias] * learningRate);     
            
            mlp.outputLayerWeights = mlp.outputLayerWeights + ([backpropOutputWeights backpropOutputBias] * learningRate);
        
        end
        
        
    end
    
end
