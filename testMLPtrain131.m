%   ____________________________
%   COMP6011: Machine Learning
%   19129576 
%   MNIST coursework
%   ____________________________



%   ____________________
%   main function
%   to run test mlp with 1 - 3 - 1 test data
%   ____________________
function testMLPtrain131

    images = loadMNISTImages('train-images-idx3-ubyte');    % Load MNIST Images
    labels = loadMNISTLabels('train-labels-idx1-ubyte');    % Load MNIST Labels
    
    display_network(images(:,1:100));   % display network using images
    disp(labels(1:100));                % display labels

    
    numberOfHiddenUnits = 700;  % Choose form of MLP:      
    learningRate = 0.07; % Learning rate: appropriate parameters.

    batchSize = 100;    % batch size, Remember there are 60k input values.    
    epochs = 1000;       % number of epochs
    
    % printing relevant data
    fprintf('Hidden units: %d.\n', numberOfHiddenUnits);
    fprintf('Learning rate: %d.\n', learningRate);
    fprintf('Batch size: %d.\n', batchSize);
    fprintf('Epochs: %d.\n', epochs);
    
    m = MLP(1, 3, 1);  % declare an MLP with one input, three hidden neurons, and one output

    m = m.initializeWeightsRandomly(1.0);   % randomize weights
    
   
    % repeat training on all data for 10000 epocs
    for x=1:epochs
        
        fprintf('epochs: %d', x);
        % first input is [0], target output is [0]
        m.train_single_data([0], [0], learningRate);
        % second input is [1], target output is [1]
        m.train_single_data([1], [1], learningRate);
        % third input is [2], target output is [0]
        m.train_single_data([2], [0], learningRate);
         
        % let's have a look at what the network currently produces.
        % this should approach the target outputs [0 1 0] as training
        % progresses.
        [m.compute_output([0]) m.compute_output([1]) m.compute_output([2])]
    

         
    end
    
   
    fprintf('fin');
    
    
end



