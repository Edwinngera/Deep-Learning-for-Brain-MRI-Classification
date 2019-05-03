%% MATLAB GUI
function varargout = braingui(varargin)
% BRAINGUI MATLAB code for braingui.fig
% Begin initialization code 
%% Matlab guide generated code
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @braingui_OpeningFcn, ...
                   'gui_OutputFcn',  @braingui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% --- Executes just before braingui is made visible.
function braingui_OpeningFcn(hObject, eventdata, handles, varargin)

% hObject    handle to figure
% handles    structure with handles and user data (see GUIDATA)

handles.output = hObject;

% create an axes that spans the whole gui

ah = axes('unit', 'normalized', 'position', [0 0 1 1]);

% import the background image 

bg = imread('images.jpg'); imagesc(bg);

% prevent plotting over the background and turn the axis off

set(ah,'handlevisibility','off','visible','off')

% making sure the background is behind all the other uicontrols
uistack(ah, 'bottom');


bg = imread('index.jpg');
bg= imresize(bg,[100,100])
set(handles.classify_im,'CData',bg)

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes braingui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = braingui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure

% Get default command line output from handles structure
varargout{1} = handles.output;
% --- Executes on button press in classify_im.
function classify_im_Callback(hObject, eventdata, handles)
% hObject    handle to classify_im (see GCBO)

%% Load The Trained Model
nett=load('network.mat')%load the saved model
net2=nett.network
%classifier=file_data.classifier;
 global im
  [path,user_cancel]=imgetfile();
  if user_cancel
      msgbox(sprintf('Invalid Selection'),'Error','Warn');
      return
  end 
     %%load image
      im=imread(path)
      setappdata(0,'im1',im)
      %%Resize Image
      imr = imresize(im,[50 50])
      %%Covert resize image to double for visualization purposes
      im2=im2double(im)
      axes(handles.disp_image);
      % Display the image to be classified
      imshow(im2)% 
      % Classify the image
      [label,scores] = classify(net2,imr);% Classify the image
      x=label;% Classification Cartegory
      y=max(scores);% Probability
      %Display the cartegory
      set(handles.d_pred,'string',y);
      a=cellstr(x)
      %Display the probability
      set(handles.confidence,'String',a);
      %Plot The probavbility in a bar graph
      axes(handles.prob)
      bar(y,'r')
      b=char(a)% Convert the cell array into a char  array
      c=string(b)% Convert the char array into a string
      
      %If The Image is a Tumour exract the tumour
      %% TUMOUR EXTRACTION IF THE CLASSIFIED IMAGE IS HAS A TUMOUR
      if c=='TUMOUR'
        %% Otsu Binarization for segmentation
        pause(5)
        level = graythresh(im);
        img = im2bw(im,.6);
        img = bwareaopen(img,100); 
        axes(handles.disp_image)
        imshow(im2)
        pause(2)
        imshow(img)
        title('SEGMENTED IMAGE(OTSU BINARIZATION)');
        pause(5)
        %% Use  Solidity characteristic to extract tumour  
        y=extractor(im)
        axes(handles.disp_image)
        imshow(y,[])% display image
        pause(2) 
        %% plot tumour boundaries in the brain MRI
        axes(handles.disp_image)
        imshow(im)  
        hold on
        [B,L]=bwboundaries(y,'noholes')
        for i=1:length(B)
            plot(B{i}(:,2),B{i}(:,1),'m','linewidth',2.0)%Plot boundaries of the tumor region    
        end
        title('TUMOUR BOUNDARY')
        hold off
       
      end
      
           
      
     
  return


function d_pred_Callback(hObject, eventdata, handles)
% hObject    handle to d_pred (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB

function d_pred_CreateFcn(hObject, eventdata, handles)
% hObject    handle to d_pred (see GCBO)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function confidence_Callback(hObject, eventdata, handles)

function confidence_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in test.
function test_Callback(hObject, eventdata, handles)


% --- Executes on selection change in popupmenu1.
%% This function investigates the activations of Convolution and Non Linearity Layers
function popupmenu1_Callback(hObject, eventdata, handles)
 contents = cellstr(get(hObject,'String')) 
 pop_choice=contents{get(hObject,'Value')}% Get contents of the pop up menu 
 image = getappdata(0,'im1')
 %% Activations of the first convolutional layer
 if(strcmp(pop_choice,'CONVOLUTION1'))
  layer='C1' 
  activation=show_act(image,layer)
  axes(handles.disp_act)
  montage(activation)
 %% Activations of the first ReLu layer
 elseif(strcmp(pop_choice,'RELU1'))
  layer='R1' 
  activation=show_act(image,layer)
  axes(handles.disp_act)
  montage(activation)
 %% Activations of the second convolutional layer 
 elseif(strcmp(pop_choice,'CONVOLUTION2'))
  layer='C2' 
  activation=show_act(image,layer)
  axes(handles.disp_act)
  montage(activation)
  %% Activations of the second ReLU layer
  elseif(strcmp(pop_choice,'RELU2'))
  layer='R2' 
  activation=show_act(image,layer)
  axes(handles.disp_act)
  montage(activation)
  %% Activations of the third convolutional layer
  elseif(strcmp(pop_choice,'CONVOLUTION3'))
  layer='C3' 
  activation=show_act(image,layer)
  axes(handles.disp_act)
  montage(activation)
  %% Activations of the third convolutional layer
  elseif(strcmp(pop_choice,'RELU3'))
  layer='R3' 
  activation=show_act(image,layer)
  axes(handles.disp_act)
  montage(activation)
  %% Activations of the final convlutional layer
  elseif(strcmp(pop_choice,'CONVOLUTION4'))
  layer='skipConv' 
  activation=show_act(image,layer)
  axes(handles.disp_act)
  montage(activation)
   
    
 end

% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
