function showPredictions(probs,testImg,testLabel,SIZE_OF_TRAINING_DATA,CLASSES)
  %show classificated images with predictions
  dispImg=zeros([28,28,3],'double');

  p=zeros([28;28],'uint8');
  for l=1:SIZE_OF_TRAINING_DATA
    for j=1:28
      p(j,:)=testImg(l,1+(j-1)*28:j*28);
      end
    p=p';
    
    dispImg(:,:,1)=p;
    dispImg(:,:,2)=p;
    dispImg(:,:,3)=p;
    image(dispImg)
    
    predictedClass=0;
    for m=1:CLASSES
      if(probs(m,l) == max(probs(:,l)))
        predictedClass=m-1;
        endif
      end

    string=sprintf("Predicted class: %d, actual class: %d",predictedClass,testLabel(l)-1);
    title(string);
    pause(1);
    end
 end