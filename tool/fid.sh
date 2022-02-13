for method in DDIM S-PNDM F-PNDM FON PF;
do
    echo $method
    mkdir -p ./temp/sample
    mpiexec -np 4 python main.py --runner sample --method $method
    pytorch-fid ./temp/sample ~/llp/Datasets/fid_cifar10_train.npz --device cuda:3
    mv ./temp/sample ./temp/$method
done