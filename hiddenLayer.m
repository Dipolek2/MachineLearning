%[img,label,testImg,testLabel] = cifar10load();
[img,label,testImg,testLabel] = loadMNIST();

DATA_SIZE=28*28;

%hyperparameters
SIZE_OF_TRAINING_DATA=2000;
SIZE_OF_NETWORK=800;
CLASSES=10;
TRAINING_STEP=20;
REG=power(10,0);
K=power(10,0);
learningRate=power(10,-1);


%create model
%first layer
Weights=randn([DATA_SIZE,SIZE_OF_NETWORK],'double').*sqrt(2.0/SIZE_OF_NETWORK);
Weights=Weights.*K;
bias=zeros([SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK],'double').*sqrt(2.0/SIZE_OF_NETWORK);


%model of hidden layer
hiddenWeights=randn([SIZE_OF_NETWORK,CLASSES],'double').*sqrt(2.0/CLASSES);
hiddenWeights=hiddenWeights.*K;
hiddenBias=zeros([SIZE_OF_TRAINING_DATA,CLASSES],'double').*sqrt(2.0/CLASSES);

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
  data_loss=0.0;
  delta=power(10,-10);
  for l=1:SIZE_OF_TRAINING_DATA
    data_loss+=-log(probs(l, label(l))+delta);
    end
  data_loss/=SIZE_OF_TRAINING_DATA;
  %data_loss=sum(sum(-log(probs')));
  %reg_loss=0.5*REG*sum(Weights*Weights')+0.5*REG*sum(hiddenWeights*hiddenWeights');
  %loss=data_loss+reg_loss
    
  loss=data_loss;
  if(mod(j,5)==0)
    printf("loss %f\n",loss);
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
    
    
  %[DATA_SIZE,10] backpropagate to first layer
  dHidden=dscores*hiddenWeights';


  %non linearity ReLU
  %crucial part; all 'magic' happens here
  for l=1:size(dHidden)
    for k=1:SIZE_OF_NETWORK
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
hidden=max(0,linearMultiplication(Weights,double(testImg),bias));
score=linearMultiplication(hiddenWeights,hidden,hiddenBias);
probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
for l=1:SIZE_OF_TRAINING_DATA
  score(l,:)-=min(score(l,:));
  probs(l,:)=score(l,:)/sum(score(l,:));
  end
probs=probs';

%accuracy
acc=0;
for l=1:SIZE_OF_TRAINING_DATA
  if(probs(testLabel(l),l) == max(probs(:,l)))
    acc+=1;  
    endif
  end
precision=double(acc*100/SIZE_OF_TRAINING_DATA);
printf("Precision %2.2f%%\n",precision);

%show classificated images with predictions
%dispImg=zeros([28,28,3],'double');

%p=zeros([28;28],'uint8');
%for l=1:SIZE_OF_TRAINING_DATA
%  for j=1:28
%    p(j,:)=testImg(l,1+(j-1)*28:j*28);
%    end
%  p=p';
  
  %dispImg(:,:,1)=p;
  %dispImg(:,:,2)=p;
  %dispImg(:,:,3)=p;
  %image(dispImg)
  %title("Predicted class: ",testLabel(l)-1);
  %pause(0.5);
  %end