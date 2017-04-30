function RGBImage = visualizeWeight(w)
%load R,G,B vectors
R=w(1:1024);
G=w(1025:2048);
B=w(2049:3072);

%load R,G,B channels
Img(:,:,1)=reshape(R,32,32);
Img(:,:,2)=reshape(G,32,32);
Img(:,:,3)=reshape(B,32,32);

%rotate image 90 degree
RGBImage = permute(Img,[2,1,3]);
end