img = cifar10load();

%display image as array[32,32,1]
%[X,MAP]=gray2ind(img{3}.image);
%image(ind2rgb(X,MAP));
%title(img{3}.label);

%create model
%x=zeros(32*32);
Weights=rand([10,3072],'double');
Weights=Weights.*0.001;
bias=rand([10,1],'double');
bias=bias.*0.001;

SIZE_OF_TRAINING_DATA=1;
TRAINING_STEPS=5;
REG=0.001;

loss=zeros(SIZE_OF_TRAINING_DATA,10);
accuracy=zeros(SIZE_OF_TRAINING_DATA,1);
gradWeights=zeros(10,3072);
newWeights=zeros(10,3072);
learningRate=power(10,-6);

delta=zeros(SIZE_OF_TRAINING_DATA,TRAINING_STEPS);

%show random weights as image
%S1=visualizeWeight(Weights(1,:));
%image(S1);
%title("Random weight");

%compute loss before training
%loss=SVMlossVect(Weights,img,bias);
prob=softmaxVect(Weights,img,bias);
prob(1,:)'

dataLoss=sum(-log2(prob))/SIZE_OF_TRAINING_DATA;
regLoss=0.5*REG*sum(sum(Weights.*Weights));
loss=dataLoss+regLoss;

%printf("Original loss= %f\n", loss);
%loss(1:SIZE_OF_TRAINING_DATA)

oldWeights=Weights;
%train weights at each image
for i=1:SIZE_OF_TRAINING_DATA
  %compute loss with updated weights
  %loss(i)=SVMloss(Weights,bias,img{i}.image,img{i}.label);
  loss(i,:)=softmax(Weights,img{i}.image,bias);
  
  %traing each image TRAINIG_STEPS times
  for j=1:TRAINING_STEPS
  
    %set learningRate and compute calculus
    %calc=calculus(Weights,bias,img{i}.image, img{i}.label);
    p=softmax(Weights,img{i}.image,bias);
    calc=-log2(p);
    
    %experimental backpropagation
    
    %over each image example
    %dscores=softmax(Weights, img{i}.image, bias);
    %dscores(img{i}.label)-=1;
    
    %dscores/=SIZE_OF_TRAINING_DATA;
    
    %dW=Weights'*dscores;
    %dW=dW';
    %dB=sum(dscores);
    %dW=dW.+Weights.*REG;
    
    
    %experimental backpropagation
    
    %compute gradients of weights' rows and biases
    imageVector=img{i}.image';
    
    for k=1:10
      if(k != img{i}.label)
        gradWeights(k,:)=imageVector.*calc(k);
        gradBias(k)=bias(k).*calc(k);
      else
        gradWeights(k,:)=imageVector.*(-(sum(calc)-calc(k)));
        gradBias(k)=bias(k).*(-sum(calc)-calc(k));
        endif
      end
  
    %softmax
    %gradWeights=imageVector.*calc;  
  
    gradWeights=gradWeights.*learningRate;
    gradBias=gradBias.*learningRate;
     
    %update weights 
    newWeights=Weights.-gradWeights;
    newBias=bias.-gradBias;
    %compute regularized loss with updated weights  
    
    %SVM
    %newLoss=SVMloss(newWeights,bias,img{i}.image,img{i}.label);
    
    %softmax
    newProb=softmax(newWeights,img{i}.image,newBias);
    newDataLoss=sum(-log2(newProb(img{i}.label)))/SIZE_OF_TRAINING_DATA;
    newRegLoss=0.5*REG*sum(sum(Weights.*Weights));
    newLoss=newDataLoss+newRegLoss
    
    Weights=newWeights;
    bias=newBias;
    
    prob=softmaxVect(Weights,img,bias);
    prob(1,:)';
    loss=-log2(prob);
    
    %update training progress
    %delta(i,j)=loss(i)-newLoss;
    end
    img{i}.label;
    s=softmax(Weights,img{i}.image,bias);
    
  end
%disp(delta(1:SIZE_OF_TRAINING_DATA,TRAINING_STEPS))
%loss2=-log2(softmaxVect(Weights,img,bias));
%delta2=loss-loss2;

z=softmaxVect(Weights,img,bias);
z(1,:)'
%S2=visualizeWeight(Weights(1,:));
%image(S2);
%title("Trained weight");