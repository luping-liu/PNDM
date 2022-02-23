for method in DDIM S-PNDM F-PNDM FON PF;
do
  echo $method
  mkdir -p ./temp/sample
  mpiexec -np 4 python main.py --runner sample --method $method --config pf_deep_cifar10.yml --model_path temp/models/pf_deep_cifar10.ckpt
  pytorch-fid ./temp/sample ~/llp/Datasets/fid_cifar10_train.npz --device cuda:3
  mv ./temp/sample ./temp/pf_deep/$method
done