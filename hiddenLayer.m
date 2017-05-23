  function hiddenLayer()
  %[img,label,testImg,testLabel] = cifar10load();
  [img,label,testImg,testLabel] = loadMNIST();

  %hyperparameters
  DATA_SIZE=28*28;
  SIZE_OF_DATA_BATCH=10000;
  SIZE_OF_TRAINING_DATA=64;
  DATA_BATCH=100;
  SIZE_OF_NETWORK=100;
  CLASSES=10;
  REG=power(10,0);
  K=power(10,2);
  learningRate=0.001;
  EPOCHS=500;

 % testImg=testImg(1:SIZE_OF_TRAINING_DATA,:);

  %create model
  %first layer
  Weights=randn([DATA_SIZE,SIZE_OF_NETWORK],'double').*sqrt(2.0/SIZE_OF_NETWORK);
  Weights=Weights.*K;
  bias=zeros([SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK],'double').*sqrt(2.0/SIZE_OF_NETWORK);

  dW=zeros([DATA_SIZE,SIZE_OF_NETWORK],'double');
  dB=zeros([SIZE_OF_TRAINING_DATA,SIZE_OF_NETWORK],'double');
  
  %model of hidden layer
  hiddenWeights=randn([SIZE_OF_NETWORK,CLASSES],'double').*sqrt(2.0/CLASSES);
  hiddenWeights=hiddenWeights.*K;
  hiddenBias=zeros([SIZE_OF_TRAINING_DATA,CLASSES],'double').*sqrt(2.0/CLASSES);
 
  dHiddenWeights=zeros([SIZE_OF_NETWORK,CLASSES],'double');
  dHiddenBias=zeros([SIZE_OF_TRAINING_DATA,CLASSES],'double');
 
  precision=zeros(SIZE_OF_TRAINING_DATA);
  maxPrecision=0.0;
  iterations=1;   
  maxIterations=0; 

  %set graph parameters
  axis([0 EPOCHS 0 100]);
  xlabel("Iteracje");
  ylabel("Prezycja [%]");
  set(gca, "linewidth", 3, "fontsize", 12);

  %train weights at each image parallel
  while (iterations <= EPOCHS)
  for i=1:DATA_BATCH
  
    %hidden layer
    hidden=max(0,linearMultiplication(Weights,img((i-1)*SIZE_OF_TRAINING_DATA+1:i*SIZE_OF_TRAINING_DATA,:),bias));
   
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
      data_loss+=-log(probs(l, label((i-1)*SIZE_OF_TRAINING_DATA+l))+delta);
      end
    data_loss/=SIZE_OF_TRAINING_DATA;
   
    %compute backpropagation
    %derivates of output; for right class it's negative number
    
    dscores=probs;
    for l=1:SIZE_OF_TRAINING_DATA
      dscores(l,label((i-1)*SIZE_OF_TRAINING_DATA+l))-=1;
      end
    dscores/=SIZE_OF_TRAINING_DATA;
   
    %Stochastic Gradient Descent
    %[Weights,bias,hiddenWeights,hiddenBias,dW,dB,dHiddenWeights,dHiddenBias]=gradientDescent(Weights,bias,hiddenWeights,hiddenBias,dscores,hidden,img((i-1)*SIZE_OF_TRAINING_DATA+1:i*SIZE_OF_TRAINING_DATA,:),SIZE_OF_TRAINING_DATA,SIZE_OF_DATA_BATCH,SIZE_OF_NETWORK,learningRate,REG);
    
    %Momentum Gradient Descent
    [Weights,bias,hiddenWeights,hiddenBias,dW,dB,dHiddenWeights,dHiddenBias]=momentum(Weights,bias,hiddenWeights,hiddenBias,dscores,hidden,img((i-1)*SIZE_OF_TRAINING_DATA+1:i*SIZE_OF_TRAINING_DATA,:),SIZE_OF_TRAINING_DATA,SIZE_OF_DATA_BATCH,SIZE_OF_NETWORK,learningRate,REG,dW,dB,dHiddenWeights,dHiddenBias);
   
    end  
    
  accuracy=0.0;
  %check accuracy after training
  for k=1:(SIZE_OF_DATA_BATCH/SIZE_OF_TRAINING_DATA)
    hidden=max(0,linearMultiplication(Weights,double(testImg((k-1)*SIZE_OF_TRAINING_DATA+1:k*SIZE_OF_TRAINING_DATA,:)),bias));
    score=linearMultiplication(hiddenWeights,hidden,hiddenBias);

    probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
    for l=1:SIZE_OF_TRAINING_DATA
      score(l,:)-=min(score(l,:));
      probs(l,:)=score(l,:)/sum(score(l,:));
      end
    probs=probs';

    acc=0;
    for l=1:SIZE_OF_TRAINING_DATA
      if(probs(testLabel((k-1)*SIZE_OF_TRAINING_DATA+l),l) == max(probs(:,l)))
        acc+=1;  
        endif
     end
    accuracy+=double(acc*100);
    end
  precision(iterations)=accuracy/(SIZE_OF_DATA_BATCH);

  %plot graph
  plot([1:iterations],precision(1:iterations),"linewidth",3,'r');
  set(gca, "linewidth", 3, "fontsize", 12);
  axis([0 EPOCHS 0 100]);
  xlabel("Iteracje");
  ylabel("Prezycja [%]");
  drawnow;

  %save best precision value
  if(precision(iterations) > maxPrecision)
    bestWeights=Weights;
    bestBias=bias;
    bestHiddenWeights=hiddenWeights;
    bestHiddenBias=hiddenBias;
    maxPrecision=precision(iterations);
    maxIterations=iterations;
    endif

  iterations+=1;
  endwhile

  printf("MAXIterations: %d\n", maxIterations);
  printf("MAXPrecision %2.2f%%\n",maxPrecision);

  accuracy=0;
  %evaluate net with best Weights
  for i=1:(SIZE_OF_DATA_BATCH/SIZE_OF_TRAINING_DATA)
  hidden=max(0,linearMultiplication(bestWeights,double(testImg((i-1)*SIZE_OF_TRAINING_DATA+1:i*SIZE_OF_TRAINING_DATA,:)),bestBias));
  score=linearMultiplication(bestHiddenWeights,hidden,bestHiddenBias);

  probs=zeros(SIZE_OF_TRAINING_DATA,CLASSES);
  for l=1:SIZE_OF_TRAINING_DATA
    score(l,:)-=min(score(l,:));
    probs(l,:)=score(l,:)/sum(score(l,:));
    end
  probs=probs';

  acc=0;
  for l=1:SIZE_OF_TRAINING_DATA
   if(probs(testLabel((i-1)*SIZE_OF_TRAINING_DATA+l),l) == max(probs(:,l)))
        acc+=1;  
      endif
    end
  accuracy+=double(acc*100);
  end
  accuracy/=(SIZE_OF_DATA_BATCH);
  
  %showPredictions(probs,testImg,testLabel,SIZE_OF_TRAINING_DATA,CLASSES);
  end