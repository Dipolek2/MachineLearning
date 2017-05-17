%[img,label,testImg,testLabel] = cifar10load();
[img,label,testImg,testLabel] = loadMNIST();

DATA_SIZE=28*28;

%hyperparameters
SIZE_OF_DATA_BATCH=10000;
SIZE_OF_TRAINING_DATA=1000;
DATA_BATCH=2;
SIZE_OF_NETWORK=80;
CLASSES=10;
REG=power(10,0);
K=power(10,0);
learningRate=power(10,-2);
EPOCHS=500;

testImg=testImg(1:SIZE_OF_TRAINING_DATA,:);
%create model
%first layer
Weights=randn([DATA_SIZE,SIZE_OF_NETWORK],'double').*sqrt(2.0/SIZE_OF_NETWORK);
Weights=Weights.*K;
bias=zeros([SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK],'double').*sqrt(2.0/SIZE_OF_NETWORK);


%model of hidden layer
hiddenWeights=randn([SIZE_OF_NETWORK,CLASSES],'double').*sqrt(2.0/CLASSES);
hiddenWeights=hiddenWeights.*K;
hiddenBias=zeros([SIZE_OF_TRAINING_DATA,CLASSES],'double').*sqrt(2.0/CLASSES);

%accuracy before training
for i=1:DATA_BATCH
  hidden=max(0,linearMultiplication(Weights,img((i-1)*SIZE_OF_DATA_BATCH+1:(i-1)*SIZE_OF_DATA_BATCH+SIZE_OF_TRAINING_DATA,:),bias));
  score=linearMultiplication(hiddenWeights,hidden,hiddenBias);
  probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
  for l=1:SIZE_OF_TRAINING_DATA
    score(l,:)-=min(score(l,:));
    probs(l,:)=score(l,:)/sum(score(l,:));
    end
  probs';
  end
    
    
%train weights at each image parallel
precision=zeros(SIZE_OF_TRAINING_DATA);
maxPrecision=0.0;
iterations=1;   
maxIterations=0; 

%precision=zeros(SIZE_OF_TRAINING_DATA*DATA_BATCH);

%plot([1:SIZE_OF_TRAINING_DATA],0,'g')

axis([0 EPOCHS 0 100]);
xlabel("Iteracje");
ylabel("Prezycja [%]");
set(gca, "linewidth", 3, "fontsize", 12);

while (precision(iterations) < 85.0 && iterations < EPOCHS)
for i=1:DATA_BATCH
  %hidden layer
  %result is matrix [SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK]
  hidden=max(0,linearMultiplication(Weights,img((i-1)*SIZE_OF_DATA_BATCH+1:(i-1)*SIZE_OF_DATA_BATCH+SIZE_OF_TRAINING_DATA,:),bias));
    
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
    data_loss+=-log(probs(l, label((DATA_BATCH-1)*SIZE_OF_TRAINING_DATA+l))+delta);
    end
  data_loss/=SIZE_OF_TRAINING_DATA;
  %data_loss=sum(sum(-log(probs')));
  %reg_loss=0.5*REG*sum(Weights*Weights')+0.5*REG*sum(hiddenWeights*hiddenWeights');
  %loss=data_loss+reg_loss
  
  %loss(iterations)=data_loss;
  %if(mod(iterations,5)==0)
  %  printf("loss %f\n",loss);
  %  endif
    
    
  %compute backpropagation
  %derivates of output; for right class it's negative number
  
  %Conjugate gradient
  %
  %Weights=conjgrad(score,bias,Weights);
  %hiddenWeights=conjgrad(hidden,hiddenBias,hiddenWeights);
  % 
  %end Conjugate gradient
  
  dscores=probs;

  for l=1:SIZE_OF_TRAINING_DATA
    dscores(l,label(l))-=1;
    end
  dscores/=SIZE_OF_TRAINING_DATA;
  %dscores=gradient(W)    
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
  dW=img((i-1)*SIZE_OF_DATA_BATCH+1:(i-1)*SIZE_OF_DATA_BATCH+SIZE_OF_TRAINING_DATA,:)'*dHidden;
  dB=sum(dHidden);
    
  %update trainable variables
  dHiddenWeights+=REG.*hiddenWeights;
  dW+=REG.*Weights;
  
  Weights-=learningRate.*dW;
  bias-=learningRate.*dB;
  hiddenWeights-=learningRate.*dHiddenWeights;
  hiddenBias-=learningRate.*dHiddenBias; 
  end
  
  %update learning rate - momentum
  %learningRate-=power(10,-4);  
  
%check accuracy after training
hidden=max(0,linearMultiplication(Weights,double(testImg),bias));
score=linearMultiplication(hiddenWeights,hidden,hiddenBias);
probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
for l=1:SIZE_OF_TRAINING_DATA
  score(l,:)-=min(score(l,:));
  probs(l,:)=score(l,:)/sum(score(l,:));
  end
probs=probs';

acc=0;
for l=1:SIZE_OF_TRAINING_DATA
  if(probs(testLabel(l),l) == max(probs(:,l)))
    acc+=1;  
    endif
  end
precision(iterations)=double(acc*100/SIZE_OF_TRAINING_DATA);

plot([1:iterations],precision(1:iterations),"linewidth",3,'r');
set(gca, "linewidth", 3, "fontsize", 12);
axis([0 EPOCHS 0 100]);
xlabel("Iteracje");
ylabel("Prezycja [%]");
drawnow;

if(precision(iterations) > maxPrecision)
  maxPrecision=precision(iterations);
  maxIterations=iterations;
  endif
  
%printf("Iterations: %d\n", iterations);
%printf("Precision %2.2f%%\n",precision(iterations);

iterations+=1;
endwhile

printf("MAXIterations: %d\n", maxIterations);
printf("MAXPrecision %2.2f%%\n",maxPrecision);

%show classificated images with predictions
%dispImg=zeros([28,28,3],'double');
%
%p=zeros([28;28],'uint8');
%for l=1:SIZE_OF_TRAINING_DATA
%  for j=1:28
%    p(j,:)=testImg(l,1+(j-1)*28:j*28);
%    end
%  p=p';
%  
%  dispImg(:,:,1)=p;
%  dispImg(:,:,2)=p;
%  dispImg(:,:,3)=p;
%  image(dispImg)
%  
%  predictedClass=0;
%  for m=1:CLASSES
%    if(probs(m,l) == max(probs(:,l)))
%      predictedClass=m-1;
%      endif
%    end
%
%  string=sprintf("Predicted class: %d, actual class: %d",predictedClass,testLabel(l)-1);
%  title(string);
%  pause(1);
%  end