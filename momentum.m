function [Weights,bias,hiddenWeights,hiddenBias,dW,dB,dHiddenWeights,dHiddenBias]=momentum(Weights,bias,hiddenWeights,hiddenBias,dscores,hidden,img,SIZE_OF_TRAINING_DATA,SIZE_OF_DATA_BATCH,SIZE_OF_NETWORK,learningRate,REG,prevDW,prevDB,prevDHiddenWeights,prevDHiddenBias)
    momentum=1.5;
    
    %backpropagate through hidden layer
    dHiddenWeights=hidden'*dscores;
    dHiddenBias=sum(dscores);
            
    %backpropagate to first layer
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
    
    dHiddenWeights+=REG.*hiddenWeights;
    dW+=REG.*Weights;
    dW*=learningRate;
    dB*=learningRate;
    dHiddenWeights*=learningRate;
    dHiddenBias*=learningRate;
    
    Weights-=dW+momentum*prevDW;
    bias-=dB+momentum*prevDB;
    hiddenWeights-=dHiddenWeights+momentum*prevDHiddenWeights;
    hiddenBias-=dHiddenBias+momentum*prevDHiddenBias; 
   
end