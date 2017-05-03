[img,label] = cifar10load();

%hyperparameters
SIZE_OF_TRAINING_DATA=10;
SIZE_OF_NETWORK=10;
CLASSES=10;
TRAINING_STEP=10;
REG=0.01;
learningRate=power(10,-3);

%create model
%first layer
Weights=rand([3072,SIZE_OF_NETWORK],'double');
Weights=Weights.*0.001;
bias=rand([1,SIZE_OF_NETWORK],'double');
bias=bias.*0.001;


%model of hidden layer
hiddenWeights=rand([SIZE_OF_NETWORK,CLASSES],'double');
hiddenWeights=hiddenWeights.*0.001;
hiddenBias=rand([SIZE_OF_TRAINING_DATA,CLASSES],'double');
hiddenBias=hiddenBias.*0.001;

%train weights at each image
for i=1:SIZE_OF_TRAINING_DATA

    %accuracy before training
    hidden=max(0,linearMultiplication(Weights,img,bias));
    score=linearMultiplication(hiddenWeights,hidden,hiddenBias);
    probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
    
    exp_scores=exp(score);
    for l=1:SIZE_OF_TRAINING_DATA
      probs(l,:)=exp_scores(l,:)/sum(exp_scores(l,:));
      end
    probs(i,:)'
    
  for j=1:TRAINING_STEP
    %hidden layer
    %result is matrix [SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK] 
    hidden=max(0,linearMultiplication(Weights,img,bias));
    
    %output layer
    score=linearMultiplication(hiddenWeights,hidden,hiddenBias);

    
    %compute probability of belonging to each class
    probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
    exp_scores=exp(score);
    for l=1:SIZE_OF_TRAINING_DATA
      probs(l,:)=exp_scores(l,:)/sum(exp_scores);
      end
    
    %overall loss
    data_loss=sum(sum(-log(probs')));
    
    %reg_loss=0.5*REG*sum(sum(Weights.*Weights))+0.5*REG*sum(sum(hiddenWeights.*hiddenWeights));
    %loss=data_loss+reg_loss
    loss=data_loss;
    %if(mod(j,5)==0)
    %  printf("loss %f\n",loss);
    %  endif
    
    
    %compute backpropagation
    %derivates of output; for right class it's negative number
    dscores=probs;
    dscores(:,label(i))-=1;
    
    %[SIZE_OF_NETWORK,10] backpropagate through hidden layer
    dHiddenWeights=hidden'*dscores;
    dHiddenBias=sum(dscores);
    
    
    %[3072,10] backpropagate to first layer
    dHidden=dscores*hiddenWeights';
    
    %non linearity ReLU
    dHidden=max(0,dHidden);
      
      
    %derivates of first layer weights
    dW=zeros(3072,SIZE_OF_NETWORK);
    dW=img'*dHidden;
    
    dB=sum(dHidden);
    
    %update trainable variables
    dHiddenWeights+=REG*hiddenWeights;
    dW+=REG*Weights;
  
    Weights-=dW;
    bias-=learningRate.*dB;
    hiddenWeights-=learningRate.*dHiddenWeights;
    hiddenBias-=learningRate.*dHiddenBias; 
    end
    
  %check accuracy after training
  hidden=max(0,linearMultiplication(Weights,img,bias));
  score=linearMultiplication(hiddenWeights,hidden,hiddenBias);
  probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
    
  exp_scores=exp(score);
  for l=1:SIZE_OF_TRAINING_DATA
    probs(l,:)=exp_scores(l,:)/sum(exp_scores(l,:));
    end 
  probs(i,:)'
  end