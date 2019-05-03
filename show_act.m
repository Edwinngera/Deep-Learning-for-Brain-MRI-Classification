%% Function to display activations of each layer
function y= show_act(image,L)
nett=load('network.mat')%load the saved model
network=nett.network
%% Display activations of each layer
image = imresize(image,network.Layers(1).InputSize(1:2));% Resize this image to the network size
activation = activations(network,image,L,'OutputAs','channels');%% Activatin of the first convolution layer
activation = reshape(activation,size(activation,1),size(activation,2),1,size(activation,3)); 
act_scaled = mat2gray(activation);
tmpact = act_scaled(:);
tmpact = imadjust(tmpact,stretchlim(tmpact));
stecthed_ac = reshape(tmpact,size(act_scaled));
y=stecthed_ac %% Display Stretched activations
end