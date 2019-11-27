function If = nonLocalMeans(I, patchSize, filtSigma, patchSigma)
% NONLOCALMEANS - Non local means CPU implementation
%   
% SYNTAX
%
%   IF = NONLOCALMEANS( IN, FILTSIGMA, PATCHSIGMA )
%
% INPUT
%
%   IN          Input image                     [m-by-n]
%   PATCHSIZE   Neighborhood size in pixels     [1-by-2]
%   FILTSIGMA   Filter sigma value              [scalar]
%   PATCHSIGMA  Patch sigma value               [scalar]
%
% OUTPUT
%
%   IF          Filtered image after nlm        [m-by-n]
%
% DESCRIPTION
%
%   IF = NONLOCALMEANS( IN, PATCHSIZE, FILTSIGMA, PATCHSIGMA ) applies
%   non local means algorithm with sigma value of FILTSIGMA, using a
%   Gaussian patch of size PATCHSIZE with sigma value of PATCHSIGMA.
%
%
  
%% USEFUL FUNCTIONS
  
 % create 3-D cube with local patches
 % patchCube = @(X,w) ...
    %  permute( ...
       %   reshape( ...
          %    im2col( ...
             %     padarray( ...
                %      X, ...
                   %   (w-1)./2, 'symmetric'), ...
                 % w, 'sliding' ), ...
              %[prod(w) size(X)] ), ...
          % [2 3 1] );
  
  % create 3D cube
  %B = patchCube(I, patchSize);
  %[m, n, d] = size( B );
  %B = reshape(B, [ m*n d ] );
  
  % gaussian patch
      H = fspecial('gaussian',patchSize, patchSigma);
      H = H(:) ./ max(H(:));

  % apply gaussian patch on 3D cube
  %B = bsxfun( @times, B, H' );
  
  % padding our image
  B= padarray(I, (patchSize-1)./2, 'symmetric');
  % kernel 
    kernel = parallel.gpu.CUDAKernel( '../cuda/nonLocalMeansKernel.ptx', ...
                               '../cuda/nonLocalMeansKernel.cu');
  
    threadsPerBlock = [32 32];
    Nthreads= sqrt(prod(size(I)));
    numberOfBlocks = ceil([Nthreads Nthreads] ./ threadsPerBlock );
    
    kernel.ThreadBlockSize = threadsPerBlock;
    kernel.GridSize        = numberOfBlocks;
    kernel.SharedMemorySize= (16+2*floor(patchSize(1)/2))*(16+2*floor(patchSize(1)/2))*4;
  %Data 
      imgGPU=gpuArray(B);
      gaussian = gpuArray(H);
      D = zeros([Nthreads Nthreads], 'gpuArray');
      tic;
      D = gather( feval(kernel, single(imgGPU) , single(gaussian), patchSize(1),filtSigma , Nthreads, single(D)) );
      toc
  
  % reshape for image
  If=D;
 

end


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.2 - January 05, 2017
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%   0.2 (Jan 05, 2017) - Dimitris
%       * minor fix (distance squared)
%
% ------------------------------------------------------------

