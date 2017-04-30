img = cifar10load();

%display image as array[32,32,1]
%[X,MAP]=gray2ind(img{3}.image);
%image(ind2rgb(X,MAP));
%title(img{3}.label);

%create model
%x=zeros(32*32);
Weights=rand([10,3072],'double');
Weights=Weights.-0.5;
bias=rand([10,1],'double');
bias=bias-0.5;

SIZE_OF_TRAINING_DATA=100;
TRAINING_STEPS=100;

loss=zeros(SIZE_OF_TRAINING_DATA,10);
accuracy=zeros(SIZE_OF_TRAINING_DATA,1);
gradWeights=zeros(10,3072);
newWeights=zeros(10,3072);
learningRate=power(10,-8);

delta=zeros(SIZE_OF_TRAINING_DATA,TRAINING_STEPS);

%show random weights as image
S1=visualizeWeight(Weights(1,:));
image(S1);
title("Random weight");

%compute loss before training
loss=SVMlossVect(Weights,img,bias);
%printf("Original loss= %f\n", loss);
%loss(1:SIZE_OF_TRAINING_DATA)

%train weights at each image
for i=1:SIZE_OF_TRAINING_DATA
  %compute loss with updated weights
  loss(i)=SVMloss(Weights,bias,img{i}.image,img{i}.label);
  
  %traing each image TRAINIG_STEPS times
  for j=1:TRAINING_STEPS
  
    %set learningRate and compute calculus
    calc=calculus(Weights,img{i}.image,img{i}.label,bias);
    
    %compute gradients of weights' rows
    
    %for k=1:10
    %  gradWeights(k)=sum(double(Weights(k,:)).*calc(k));
    %  end
    
    
    %each class weights multiply by calculus
    %gradWeights=double(Weights).*calc;
    
    %label class: - sum over all classes calculus except label calss multiply by label class weights
    %gradWeights(img{i}.label,:)=-((double(Weights(img{i}.label))).*(sum(calc)-calc(img{i}.label)));
    imageVector=img{i}.image';
    for k=1:10
      if(k != img{i}.label)
        gradWeights(k,:)=imageVector.*calc(k);
      else
        gradWeights(k,:)=imageVector.*-(sum(calc)-calc(k));
        endif
      end
  
    gradWeights=gradWeights.*learningRate;
     
    %update weights 
    
    %for k=1:10
    %  if(k != img{i}.label)
    %    newWeights(k,:)=Weights(k,:).-gradWeights(k);
    %    endif
    %  end
    
    newWeights=Weights.-gradWeights;
    
    %compute loss with updated weights  
    newLoss=SVMloss(newWeights,bias,img{i}.image,img{i}.label);
    Weights=newWeights;
    
    %update training progress
    delta(i,j)=loss(i)-newLoss;
    end
  end
disp(delta(1:SIZE_OF_TRAINING_DATA,TRAINING_STEPS))
loss2=SVMlossVect(Weights,img,bias);
delta2=loss-loss2;

S2=visualizeWeight(Weights(1,:));
image(S2);
title("Trained weight");
%loss=SVMloss(img{1},score);
%gradient(softmax(Weights,x,bias),[Weights,bias]);
%y=sum(W.*x)+b;