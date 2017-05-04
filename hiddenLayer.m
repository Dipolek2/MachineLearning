[img,label] = cifar10load();

%hyperparameters
SIZE_OF_TRAINING_DATA=10;
SIZE_OF_NETWORK=100;
CLASSES=10;
TRAINING_STEP=2000;
REG=0.001;
learningRate=power(10,-3);

%create model
%first layer
Weights=rand([3072,SIZE_OF_NETWORK],'double');
Weights=Weights.*REG;
bias=rand([SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK],'double');
bias=bias.*REG;


%model of hidden layer
hiddenWeights=rand([SIZE_OF_NETWORK,CLASSES],'double');
hiddenWeights=hiddenWeights.*REG;
hiddenBias=rand([SIZE_OF_TRAINING_DATA,CLASSES],'double');
hiddenBias=hiddenBias.*REG;

%train weights at each image parallel

%accuracy before training
hidden=max(0,linearMultiplication(Weights,img,bias));
score=linearMultiplication(hiddenWeights,hidden,hiddenBias);
probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
for l=1:SIZE_OF_TRAINING_DATA
  score(l,:)-=min(score(l,:));
  probs(l,:)=score(l,:)/sum(score(l,:));
  end
probs';
    
for j=1:TRAINING_STEP
  %hidden layer
  %result is matrix [SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK]
  hidden=max(0,linearMultiplication(Weights,img,bias));
    
  %output layer
  score=linearMultiplication(hiddenWeights,hidden,hiddenBias);

    
  %compute probability of belonging to each class
  probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
  for l=1:SIZE_OF_TRAINING_DATA
      score(l,:)-=min(score(l,:));
      probs(l,:)=score(l,:)/sum(score(l,:));
    end
      
  %overall loss
  data_loss=0;
  for l=1:SIZE_OF_TRAINING_DATA
    data_loss+=-log(probs(l, label(l)));
    end
  data_loss/=SIZE_OF_TRAINING_DATA;
    
  %data_loss=sum(sum(-log(probs')));
  %reg_loss=0.5*REG*sum(sum(Weights.*Weights))+0.5*REG*sum(sum(hiddenWeights.*hiddenWeights));
  %loss=data_loss+reg_loss;
    
  loss=data_loss;
  if(mod(j,100)==0)
    printf("loss %.5f\n",loss);
    endif
    
    
  %compute backpropagation
  %derivates of output; for right class it's negative number
  dscores=probs;
  for l=1:SIZE_OF_TRAINING_DATA
    dscores(l,label(l))-=1;
    end
  dscores/=SIZE_OF_TRAINING_DATA;
      
  %[SIZE_OF_NETWORK,10] backpropagate through hidden layer
  dHiddenWeights=hidden'*dscores;
  dHiddenBias=sum(dscores);
    
    
  %[3072,10] backpropagate to first layer
  dHidden=dscores*hiddenWeights';

  %non linearity ReLU
  %crucial part; all 'magic' happens here
  for l=1:size(dHidden)
    for k=1:SIZE_OF_TRAINING_DATA
      if(hidden(l,k)<=0)
        dHidden(l,k)=0;
        endif
      end
    end
      
  %derivates of first layer weights
  dW=img'*dHidden;
  dB=sum(dHidden);
    
  %update trainable variables
  dHiddenWeights+=REG.*hiddenWeights;
  dW+=REG.*Weights;
  
  Weights-=learningRate.*dW;
  bias-=learningRate.*dB;
  hiddenWeights-=learningRate.*dHiddenWeights;
  hiddenBias-=learningRate.*dHiddenBias; 
  end
    
%check accuracy after training
hidden=max(0,linearMultiplication(Weights,img,bias));
score=linearMultiplication(hiddenWeights,hidden,hiddenBias);
probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
for l=1:SIZE_OF_TRAINING_DATA
  score(l,:)-=min(score(l,:));
  probs(l,:)=score(l,:)/sum(score(l,:));
  end
label';
probs=probs';

%accuracy
acc=0;
for l=1:SIZE_OF_TRAINING_DATA
  if(probs(label(l),l) == max(probs(:,l)))
    acc+=1;  
    endif
  end
precision=acc*100/SIZE_OF_TRAINING_DATA;
printf("Precision %2.2f%%\n",precision);