export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_USE_CUDA_DSA=1
python train.py "$@"